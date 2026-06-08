import numpy as np
import tvm
from tvm import relax, topi
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tirx as T
import pickle as pkl
import torch
import torchvision
import matplotlib.pyplot as plt

# =============================================================================
# Chapter 7: Computational Graph Optimization
# https://book.mlc.ai/chapter_graph_optimization/index.html
#
# Previous chapters optimized individual ops (loop tiling, vectorization, GPU).
# Chapter 7 steps back and optimizes at the GRAPH level — looking at multiple
# ops together to find fusion opportunities before lowering to loops.
#
# Key insight: fuse at high level first (matmul + add obvious as Relax ops),
# then lower the fused version to TensorIR (loop-level optimization).
# After lowering, the fusion opportunity is buried in loop indices and harder to see.
# =============================================================================

# -----------------------------------------------------------------------------
# 7.1. The IRModule as an AST (Abstract Syntax Tree)
#
# An IRModule is a tree. Every binding, call, and op is a node you can walk.
# Understanding the structure is what makes pattern-matching rewrites possible.
#
# IRModule
# └── Function ("main")
#     └── SeqExpr (body)
#         └── DataflowBlock
#             ├── VarBinding: lv0 = multiply(x, y)   ← binding.var / binding.value
#             └── VarBinding: gv0 = add(lv0, y)
#
# binding.value is a Call node:
#   call.op   = the operation (add, multiply, ewise_fma, ...)
#   call.args = list of inputs
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyModule:
    @R.function
    def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
        with R.dataflow():
            lv0 = relax.op.multiply(x, y)
            gv0 = relax.op.add(lv0, y)
            R.output(gv0)
        return gv0

# Walk the AST to see its structure
relax_func = MyModule["main"]

type(relax_func)

relax_func.params

func_body = relax_func.body
type(func_body)

func_body.blocks

dataflow_block = func_body.blocks[0]

# lv0 = relax.op.multiply(x, y)
# gv0 = relax.op.add(lv0, y)

dataflow_block.bindings

binding = dataflow_block.bindings[0]

binding.var      # → lv0
binding.value    # → Call(multiply, [x, y])

# -----------------------------------------------------------------------------
# 7.2. Visitor Pattern — EwiseFMARewriter
#
# PyExprMutator handles tree walking for us. We just override visit_call_() and
# write the transformation logic for the nodes we care about.
#
# Pattern we want to find and replace:
#   add(multiply(x, y), z)  →  ewise_fma(x, y, z)
#
# visit_expr_post_order: processes children before the current node, so by the
#   time we check call.args[0], it's already been transformed if needed.
# lookup_binding: resolves a variable reference back to the Call that produced it.
#   Needed because call.args[0] is lv0 (a Var), not the multiply Call directly.
# remove_all_unused: after rewriting, lv0 (the intermediate multiply) is no longer
#   referenced — this call prunes dead bindings from the function.
# -----------------------------------------------------------------------------

@relax.expr_functor.mutator
class EwiseFMARewriter(relax.PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        add_op = tvm.ir.Op.get("relax.add")
        multiply_op = tvm.ir.Op.get("relax.multiply")
        ewise_fma_op = tvm.ir.Op.get("relax.ewise_fma")

        if call.op != add_op:
            return call

        # look through the variable to the Call that produced it
        value = self.lookup_binding(call.args[0])
        if not isinstance(value, relax.Call) or value.op != multiply_op:
            return call

        # found add(multiply(x,y), z) → replace with ewise_fma(x,y,z)
        fma_call = relax.Call(
            ewise_fma_op, [value.args[0], value.args[1], call.args[1]], None, None
        )
        return fma_call


updated_fn = EwiseFMARewriter().visit_expr(MyModule["main"])
# updated_fn.show()
# relax.analysis.remove_all_unused(updated_fn).show()   # prunes dead lv0 binding

# -----------------------------------------------------------------------------
# 7.3. Building the MLP model as a Relax IRModule
#
# Same FashionMNIST MLP from Chapter 5, but built directly with BlockBuilder
# (no torch.fx import). Weights are baked in as constants via relax.const.
# permute_dims transposes the weight matrix (matmul convention: x @ w.T).
# This is the "before fusion" representation — separate matmul and add ops.
# -----------------------------------------------------------------------------

mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))

def create_model():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((1, 784), "float32"))
    w0 = relax.const(mlp_params["w0"], "float32")
    b0 = relax.const(mlp_params["b0"], "float32")
    w1 = relax.const(mlp_params["w1"], "float32")
    b1 = relax.const(mlp_params["b1"], "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.matmul(x, relax.op.permute_dims(w0)))
            lv1 = bb.emit(relax.op.add(lv0, b0))
            lv2 = bb.emit(relax.op.nn.relu(lv1))
            lv3 = bb.emit(relax.op.matmul(lv2, relax.op.permute_dims(w1)))
            lv4 = bb.emit(relax.op.add(lv3, b1))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    return bb.get()

MLPModel = create_model()
# MLPModel.show()

# -----------------------------------------------------------------------------
# 7.4. Graph-Level Fusion — MatmulAddFusor
#
# Why sub-functions instead of a dedicated op (like ewise_fma)?
# A dedicated fused op requires writing a new registered op for every combination.
# Sub-functions are a flexible container: ANY ops can be grouped into one.
# Primitive=1 attribute marks it as already fused → downstream passes skip it.
#
# transform() walks all functions in the module, skips already-primitive ones,
# runs visit_expr on each, then removes dead bindings.
#
# visit_call_ detects the pattern matmul(x, w) → add(..., b) and:
#   1. Creates a fresh BlockBuilder with a new fused_matmul_addN function inside
#   2. Sets Primitive=1 on it (don't fuse again)
#   3. Adds it to the module via builder_.add_func
#   4. Returns a Call to that new function — replacing the two separate ops
# -----------------------------------------------------------------------------

@relax.expr_functor.mutator
class MatmulAddFusor(relax.PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__()
        self.mod_ = mod
        # cache pre-defined ops
        self.add_op = tvm.ir.Op.get("relax.add")
        self.matmul_op = tvm.ir.Op.get("relax.matmul")
        self.counter = 0

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            # skip already-fused primitive functions — don't fuse twice
            if func.attrs is not None and "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
                continue
            updated_func = self.visit_expr(func)
            updated_func = relax.analysis.remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            return node.op == op

        # pattern match matmul => add
        if not match_call(call, self.add_op):
            return call

        value = self.lookup_binding(call.args[0])
        if value is None:
            return call

        if not match_call(value, self.matmul_op):
            return call

        x = value.args[0]
        w = value.args[1]
        b = call.args[1]

        # build the fused sub-function: matmul(x, w) + b
        param_x = relax.Var("x" ,relax.TensorStructInfo(x.struct_info.shape, x.struct_info.dtype))
        param_w = relax.Var("w" ,relax.TensorStructInfo(w.struct_info.shape, w.struct_info.dtype))
        param_b = relax.Var("b" ,relax.TensorStructInfo(b.struct_info.shape, b.struct_info.dtype))

        bb = relax.BlockBuilder()

        fn_name = "fused_matmul_add%d" % (self.counter)
        self.counter += 1
        with bb.function(fn_name, [param_x, param_w, param_b]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.matmul(param_x, param_w))
                gv = bb.emit_output(relax.op.add(lv0, param_b))
            bb.emit_func_output(gv)

        # Primitive=1: marks this sub-function as a fused op — don't fuse again
        fused_fn = bb.get()[fn_name].with_attr("Primitive", 1)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        # replace the two separate calls with one call to the fused sub-function
        return relax.Call(global_var, [x, w, b], None, None)

# Wrap as a reusable module pass — can be chained with other passes
@tvm.ir.transform.module_pass(opt_level=2, name="MatmulAddFuse")
class FuseDenseAddPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return MatmulAddFusor(mod).transform()


MLPFused = FuseDenseAddPass()(MLPModel)
# MLPFused.show()   # now has fused_matmul_add0 and fused_matmul_add1 sub-functions

# -----------------------------------------------------------------------------
# 7.5. Lowering High-Level Relax Ops to TensorIR — LowerToTensorIR
#
# After graph fusion, we still have high-level ops (relax.matmul, relax.add, etc.).
# This pass lowers them to actual TensorIR prim_funcs using topi (TVM's pre-built TE).
#
# op_map: maps each Relax op string → a handler function
#   handler(bb, call) extracts the call's args and emits a topi call via bb.call_te
#
# Why fuse FIRST, then lower?
#   At the Relax level, "matmul + add" is obvious — two consecutive Call nodes.
#   After lowering to TensorIR, both become separate loop nests; the pattern is
#   buried in index arithmetic and much harder to detect.
# -----------------------------------------------------------------------------

@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: IRModule, op_map) -> None:
        super().__init__()
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }


    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        if call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()


# Handler functions: extract args from the Relax Call, emit TensorIR via topi
def map_matmul(bb, call):
    x, w = call.args
    return bb.call_te(topi.nn.matmul, x, w)

def map_add(bb, call):
    a, b = call.args
    return bb.call_te(topi.add, a, b)

def map_relu(bb, call):
    return bb.call_te(topi.nn.relu, call.args[0])

def map_transpose(bb, call):
    return bb.call_te(topi.transpose, call.args[0], )

op_map = {
  "relax.matmul": map_matmul,
  "relax.add": map_add,
  "relax.nn.relu": map_relu,
  "relax.permute_dims": map_transpose
}

@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return LowerToTensorIR(mod, op_map).transform()


MLPModelTIR = LowerToTensorIRPass()(MLPFused)
# MLPModelTIR.show()   # fused sub-functions now call two separate TensorIR prim_funcs

# -----------------------------------------------------------------------------
# 7.6. FuseTIR — merge the two TensorIR prim_funcs into one
#
# After lowering, each fused_matmul_addN function calls TWO TensorIR prim_funcs
# (one for matmul, one for add) via separate call_tir nodes.
#
# FuseTIR is a built-in TVM pass that merges those into a single TensorIR function.
# The result: one kernel with both ops' loops together — the compiler can now
# optimize across the boundary (e.g. keep matmul output in cache for the add).
#
# Passes compose: custom fusion (ours) → lower to TIR (ours) → fuse TIR (TVM's)
# We write the pattern-matching logic; TVM handles the loop-level merge.
# -----------------------------------------------------------------------------

MLPModelFinal = relax.transform.FuseTIR()(MLPModelTIR)
# MLPModelFinal.show()   # fused_matmul_add0 is now ONE TensorIR function

# Run on FashionMNIST to verify the full pipeline produces correct predictions
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img, label = next(iter(test_loader))
img = img.reshape(1, 28, 28).numpy()

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
# plt.show()

print("Class:", class_names[label[0]])

target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(MLPModelFinal, target)
vm = relax.VirtualMachine(ex, tvm.cpu())
data_nd = tvm.runtime.tensor(img.reshape(1, 784))

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MLPModule Prediction:", class_names[pred_kind[0]])
