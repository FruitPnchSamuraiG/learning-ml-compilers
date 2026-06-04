import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tirx as T
import torch
import torchvision
import torch.nn as nn
from torch import fx
from torch.nn import functional as F
from tvm import te
import matplotlib.pyplot as plt
import pickle as pkl
from tvm import topi

# =============================================================================
# Chapter 5: Integration with Machine Learning Frameworks
# https://book.mlc.ai/chapter_integration/index.html
#
# Problem: in previous chapters we wrote TensorIR/Relax by hand (TVMScript).
# Real models are written in PyTorch — we need a way to import them into TVM.
# Solution: Tensor Expression (TE) + BlockBuilder + torch.fx tracing.
# =============================================================================

# -----------------------------------------------------------------------------
# Tensor Expression (TE) — declarative shorthand for TensorIR
#
# Instead of writing explicit loops + T.sblock + T.axis by hand, TE lets you
# describe *what* to compute with lambdas and reduce_axis. TVM generates the
# TensorIR loops automatically from this description.
#
# te.placeholder: declares a symbolic input tensor (no actual data)
# te.reduce_axis: declares an accumulation dimension (the k in matmul)
# te.compute:     describes the output — shape + a lambda for each element
# te.create_prim_func: turns the TE graph into a real TensorIR prim_func
# -----------------------------------------------------------------------------

A = te.placeholder((128, 128), name="A", dtype="float32")
B = te.placeholder((128, 128), name="B", dtype="float32")

# print(type(A))
# print(A.shape)

def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")

C = te_matmul(A, B)
# te.create_prim_func([A, B, C]).show()

# lambda *i — works for ANY shape (1D, 2D, ND). lambda i,j would only work for 2D.
def te_relu(A: te.Tensor) -> te.Tensor:
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")

X1 = te.placeholder((10,), name="X1", dtype="float32")
Y1 = te_relu(X1)
# te.create_prim_func([X1, Y1]).show()

X2 = te.placeholder((10, 20), name="X1", dtype="float32")
Y2 = te_relu(X2)
# te.create_prim_func([X2, Y2]).show()

# -----------------------------------------------------------------------------
# Operator Fusion via TE
#
# When you call te.create_prim_func([A, B, D]) and skip C, TVM sees that C is
# not in the output list and makes it an *internal* buffer — it stays in cache
# between the matmul and relu instead of writing back to main memory.
# This is fusion: two ops become one fused kernel with no memory roundtrip.
#
# te.create_prim_func([A, B, D])    → fused matmul+relu (C is internal)
# te.create_prim_func([A, B, C, D]) → two separate ops (C materializes in memory)
# -----------------------------------------------------------------------------

C = te_matmul(A, B)
D = te_relu(C)

# te.create_prim_func([A, B, D]).show()    # fused: C stays in cache
# te.create_prim_func([A, B, C, D]).show() # separate: C written to memory

# -----------------------------------------------------------------------------
# BlockBuilder — programmatically build an IRModule
#
# Instead of writing TVMScript by hand, BlockBuilder lets you construct an
# IRModule in Python code. This is what enables the automatic PyTorch import
# path later — we walk the PyTorch graph and emit ops one by one.
#
# emit_te: wraps a TE function, generates the TensorIR prim_func automatically,
#          adds a call_tir into the Relax graph, and returns a DataflowVar.
#
# DataflowVar: intermediate result inside a dataflow block — only visible inside.
# emit_output(D) → R: promotes D (DataflowVar, trapped) to R (regular Var, can exit).
# -----------------------------------------------------------------------------

A = relax.Var("A", relax.TensorStructInfo((128, 128), "float32"))
B = relax.Var("B", relax.TensorStructInfo((128, 128), "float32"))

bb = relax.BlockBuilder()

with bb.function("main"):
    with bb.dataflow():
        C = bb.emit_te(te_matmul, A, B)   # generates TensorIR + call_tir automatically
        D = bb.emit_te(te_relu, C)
        R = bb.emit_output(D)             # D is DataflowVar → R is regular Var
    bb.emit_func_output(R, params=[A, B])

MyModule = bb.get()
# MyModule.show()

# -----------------------------------------------------------------------------
# Importing a PyTorch model using torch.fx
#
# torch.fx.symbolic_trace runs a forward pass with symbolic inputs — it records
# every operation as a node in a computation graph. Each node has an op type:
#
#   placeholder  → function input (relax.Var)
#   get_attr     → weight tensor from the model (relax.const)
#   call_function → standalone function like torch.matmul (→ bb.emit_te)
#   call_module  → nn.Module layer like nn.Linear (→ bb.emit_te with weights)
#   output       → function return value
#
# node_map: dictionary from fx.Node → TVM relax.Var, used to look up inputs
#   when translating each node (each op reads from previously translated nodes).
# -----------------------------------------------------------------------------

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(128, 128))

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)
        return x

model = MyModel()
fx_module = fx.symbolic_trace(model)
# print(type(fx_module))
# print(fx_module.graph.print_tabular())

# map_param: converts a PyTorch weight to a TVM constant.
# detach() removes gradient tracking — we don't need autograd for inference.
def map_param(param: nn.Parameter):
    t = param.detach() if hasattr(param, "detach") else param
    arr = t.cpu().numpy().astype("float32")
    return relax.const(arr, "float32")

# fetch_attr: navigates dotted attribute paths like "layer1.weight" to get
# the actual tensor from a nested nn.Module structure.
def fetch_attr(fx_mod, target: str):
    """Helper function to fetch an attr"""
    target_atoms = target.split('.')
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

# from_fx: walks the fx graph node by node and builds an IRModule.
# call_function_map: maps PyTorch functions (torch.matmul) → TVM emit functions
# call_module_map:  maps nn.Module types (nn.Linear) → TVM emit functions
def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
    input_index = 0
    node_map = {}
    named_modules = dict(fx_mod.named_modules())

    bb = relax.BlockBuilder()

    fn_inputs = []
    fn_output = None
    with bb.function("main"):
        with bb.dataflow():
            for node in fx_mod.graph.nodes:
                if node.op == "placeholder":
                    # create input placeholder
                    shape = input_shapes[input_index]
                    input_index += 1
                    input_var = relax.Var(
                        node.target, relax.TensorStructInfo(shape, "float32")
                    )
                    fn_inputs.append(input_var)
                    node_map[node] = input_var
                elif node.op == "get_attr":
                    node_map[node] = map_param(fetch_attr(fx_mod, node.target))
                elif node.op == "call_function":
                    node_map[node] = call_function_map[node.target](bb, node_map, node)
                elif node.op == "call_module":
                    named_module = named_modules[node.target]
                    node_map[node] = call_module_map[type(named_module)](bb, node_map, node, named_module)
                elif node.op == "output":
                    output = node_map[node.args[0]]
                    assert fn_output is None
                    fn_output = bb.emit_output(output)
        # output and finalize the function
        bb.emit_func_output(output, fn_inputs)
    return bb.get()

# Handler for torch.matmul → looks up the translated inputs via node_map
def map_matmul(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    B = node_map[node.args[1]]
    return bb.emit_te(te_matmul, A, B)

def map_relu(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    return bb.emit_te(te_relu, A)

MyModule = from_fx(
    fx_module,
    input_shapes = [(1, 128)],
    call_function_map = {
      torch.matmul: map_matmul,
      torch.relu: map_relu,
    },
    call_module_map={},
)

# MyModule.show()

# -----------------------------------------------------------------------------
# Applying from_fx to the FashionMNIST MLP
# -----------------------------------------------------------------------------

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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear0 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.linear0(x)
        x = self.relu(x)
        x = self.linear1(x)
        return x

mlp_model = MLP()

mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
mlp_model.linear0.weight.data = torch.from_numpy(mlp_params["w0"])
mlp_model.linear0.bias.data = torch.from_numpy(mlp_params["b0"])
mlp_model.linear1.weight.data = torch.from_numpy(mlp_params["w1"])
mlp_model.linear1.bias.data = torch.from_numpy(mlp_params["b1"])

torch_res = mlp_model(torch.from_numpy(img.reshape(1, 784)))

pred_kind = np.argmax(torch_res.detach().numpy(), axis=1)
# print("Torch Prediction:", class_names[pred_kind[0]])

# -----------------------------------------------------------------------------
# topi — TVM's pre-built TE functions
#
# Instead of writing te_matmul by hand, topi.nn.dense and topi.add already exist.
# topi.nn.dense(x, w) computes x @ w.T (transposed weight, which is what nn.Linear does).
# Using topi saves writing TE definitions for common ops.
#
# map_nn_linear handles nn.Linear: extracts weight and bias from the nn.Module
# object (nn_mod), converts them to constants, and emits dense + add.
# map_nn_relu delegates to map_relu (which uses our te_relu).
# -----------------------------------------------------------------------------

def map_nn_linear(bb, node_map, node, nn_mod):
    x = node_map[node.args[0]]
    w = map_param(nn_mod.weight)
    if nn_mod.bias is not None:
        b = map_param(nn_mod.bias)
    y = bb.emit_te(topi.nn.dense, x, w)   # topi.nn.dense = x @ w.T
    return bb.emit_te(topi.add, y, b)      # bias add

def map_nn_relu(bb, node_map, node, nn_mod):
    return map_relu(bb, node_map, node)


MLPModule = from_fx(
    fx.symbolic_trace(mlp_model),
    input_shapes = [(1, 784)],
    call_function_map={
    },
    call_module_map={
        torch.nn.Linear: map_nn_linear,
        torch.nn.ReLU: map_nn_relu,
    },
)

# MLPModule.show()

target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(MLPModule, target)
vm = relax.VirtualMachine(ex, tvm.cpu())
data_nd = tvm.runtime.tensor(img.reshape(1, 784))

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MLPModule Prediction:", class_names[pred_kind[0]])
