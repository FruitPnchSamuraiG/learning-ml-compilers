import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tirx as T
import IPython
from tvm.s_tir import meta_schedule as ms
import torch
import torchvision
import matplotlib.pyplot as plt
import pickle as pkl

# =============================================================================
# Chapter 4: Automatic Program Optimization
# https://book.mlc.ai/chapter_auto_program_optimization/index.html
#
# Problem: in Chapter 2/3 we manually chose transformations (split j by 4).
# But how do we know 4 is best? What if 8 is faster? Impossible to try manually.
# Solution: define a *search space* of programs, let TVM search it automatically.
# =============================================================================

def code2html(code):
    """Helper function to use pygments to turn the code string into highlighted html."""
    import pygments
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import Python3Lexer
    formatter = HtmlFormatter()
    html = pygments.highlight(code, Python3Lexer(), formatter)
    return "<style>%s</style>%s\n" % (formatter.get_style_defs(".highlight"), html)

# -----------------------------------------------------------------------------
# The program we want to optimize: 128x128 matrix multiply
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm = a_np @ b_np

a_nd = tvm.runtime.tensor(a_np)
b_nd = tvm.runtime.tensor(b_np)
c_nd = tvm.runtime.tensor(np.empty((128, 128), dtype="float32"))

lib = tvm.compile(MyModule, target="llvm")
f_timer_before = lib.mod.time_evaluator("main", tvm.cpu())
# print("Time cost of MyModule: %.3f ms" % (f_timer_before(a_nd, b_nd, c_nd).mean * 1000))

# -----------------------------------------------------------------------------
# Deterministic schedule (Chapter 2 style) — fixed jfactor, same result always
# -----------------------------------------------------------------------------

def schedule_mm(sch: tvm.s_tir.Schedule, jfactor=4):
    block_C = sch.get_sblock("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_0, j_1 = sch.split(loop=j, factors=[None, jfactor])
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch

sch = tvm.s_tir.Schedule(MyModule)
sch = schedule_mm(sch)
# print(sch.mod.script())

lib = tvm.compile(sch.mod, target="llvm")
f_timer_after = lib.mod.time_evaluator("main", tvm.cpu())
# print("Time cost of MyModule=>schedule_mm: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))

# print(sch.trace)

# -----------------------------------------------------------------------------
# Stochastic schedule — uses sample_perfect_tile instead of a fixed jfactor.
#
# sample_perfect_tile randomly picks factors that multiply to the loop size.
# For j=128, n=2: could pick [4,32], [8,16], [16,8], [32,4], etc.
# This single function now defines an entire *space* of valid programs.
#
# The returned j_factors are SYMBOLIC variables (tvm.tir.expr.Var), not real
# numbers. The actual chosen values are stored in the trace's "decision" field.
# This lets TVM replay any choice exactly, or search over many choices.
# -----------------------------------------------------------------------------

def stochastic_schedule_mm(sch: tvm.s_tir.Schedule):
    block_C = sch.get_sblock("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_factors = sch.sample_perfect_tile(loop=j, n=2)   # symbolic, not a real number yet
    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch

sch = tvm.s_tir.Schedule(MyModule)
sch = stochastic_schedule_mm(sch)

# print(sch.mod.script())
# print(sch.trace)   # trace shows: decision=[8,16] — the actual values chosen

# Stepping through the stochastic schedule manually to inspect types/trace:
sch = tvm.s_tir.Schedule(MyModule)
block_C = sch.get_sblock("C", "main")
i, j, k = sch.get_loops(block=block_C)
j_factors = sch.sample_perfect_tile(loop=j, n=2)

# print(type(j_factors[0]))   # → tvm.tir.expr.Var (symbolic, not a number!)

# print(sch.trace)
# print(sch.mod.script())   # module unchanged — decisions not applied until split

j_0, j_1 = sch.split(loop=j, factors=j_factors)
sch.reorder(i, j_0, k, j_1)

# print(sch.trace)
# print(sch.mod.script())

sch.decompose_reduction(block_C, k)

# print(sch.mod.script())

# -----------------------------------------------------------------------------
# Random search — simplest possible search over the stochastic space.
# Each call to stochastic_schedule_mm picks random j_factors, compile, benchmark.
# Keep the fastest variant found across num_trials attempts.
# This is the baseline; meta_schedule (below) does this much smarter.
# -----------------------------------------------------------------------------

def random_search(mod: tvm.IRModule, num_trials=5):
    best_result = None
    best_sch = None

    for i in range(num_trials):
        sch = stochastic_schedule_mm(tvm.s_tir.Schedule(mod))
        lib = tvm.compile(sch.mod, target="llvm")
        f_timer_after = lib.mod.time_evaluator("main", tvm.cpu())
        result = f_timer_after(a_nd, b_nd, c_nd).mean

        print("=====Attempt %d, time-cost: %.3f ms====" % (i, result * 1000))
        print(sch.trace)

        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch

    return best_sch

# sch = random_search(MyModule)
# print(sch.trace)


# -----------------------------------------------------------------------------
# MetaSchedule tune_tir — smarter search than random_search:
#
# 1. Parallel benchmarking: compiles/runs many variants simultaneously
# 2. XGBoost cost model: ML model predicts performance, skips bad variants
#    without running them — only benchmarks the promising ones
# 3. Evolutionary search: learns from results; good variant → try similar ones
#
# max_trials_global = total search budget (number of variants to try)
# num_trials_per_iter = how many per round (more rounds = smarter search)
#
# NOTE: tune_tir with manual ScheduleFn space has a known runner issue in
# this version of mlc-ai-nightly — all trials return N/A latency. The
# auto-scheduling version below (no space= parameter) works correctly.
# -----------------------------------------------------------------------------

# database = ms.tune_tir(
#     mod=MyModule,
#     target=tvm.target.Target({"kind": "llvm", "num-cores": 1}),
#     max_trials_global=64,
#     num_trials_per_iter=64,
#     space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
#     work_dir="./tune_tmp",
# )

# sch = ms.tir_integration.compile_tir(database, MyModule, tvm.target.Target({"kind": "llvm", "num-cores": 1}))
# print(sch.trace.show())

# -----------------------------------------------------------------------------
# Auto-scheduling (no manual space) — TVM analyzes the loop structure itself
# and generates the search space automatically.
#
# It looks at spatial/reduce axes, data access patterns, and loop structure,
# then tries splits on ALL loops (i, j, k), parallelization, vectorization,
# unrolling, and loop fusion — thousands of variants.
#
# The resulting tiling structure is SSRSRS (Spatial,Spatial,Reduce,Spatial,
# Reduce,Spatial) — splits each loop into multiple levels and interleaves
# spatial and reduce loops for better cache behavior than manual split.
#
# Result: auto-scheduling outperforms manual space (e.g. 0.158ms vs 0.181ms).
# -----------------------------------------------------------------------------

# database = ms.tune_tir(
#     mod=MyModule,
#     target=tvm.target.Target({"kind": "llvm", "num-cores": 1}),
#     max_trials_global=64,
#     num_trials_per_iter=64,
#     work_dir="./tune_tmp",
# )

# sch = ms.tir_integration.compile_tir(database, MyModule, tvm.target.Target({"kind": "llvm", "num-cores": 1}))
# print(sch.trace.show())
# print(sch.mod.script())

# -----------------------------------------------------------------------------
# Putting it back into the end-to-end model (FashionMNIST MLP)
#
# Strategy: extract linear0 from the module, tune it in isolation, then
# plug the optimized version back into the full model.
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
mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))

data_nd = tvm.runtime.tensor(img.reshape(1, 784))
nd_params = {k: tvm.runtime.tensor(v) for k, v in mlp_params.items()}

# Mixed module: linear0 is our TensorIR, relu and second linear use PyTorch
@tvm.script.ir_module
class MyModuleMixture:
    @T.prim_func
    def linear0(X: T.Buffer((1, 784), "float32"),
                W: T.Buffer((128, 784), "float32"),
                B: T.Buffer((128,), "float32"),
                Z: T.Buffer((1, 128), "float32")):
        T.func_attr({"global_symbol": "linear0", "tirx.noalias": True})
        Y = T.alloc_buffer((1, 128), "float32")
        for i, j, k in T.grid(1, 128, 784):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]

        for i, j in T.grid(1, 128):
            with T.sblock("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] =  Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w0: R.Tensor((128, 784), "float32"),
             b0: R.Tensor((128,), "float32"),
             w1: R.Tensor((10, 128), "float32"),
             b1: R.Tensor((10,), "float32")):
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0,), R.Tensor((1, 128), dtype="float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out

@tvm.register_global_func("env.linear", override=True)
def torch_linear(x: tvm.runtime.Tensor,
                 w: tvm.runtime.Tensor,
                 b: tvm.runtime.Tensor,
                 out: tvm.runtime.Tensor):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)

@tvm.register_global_func("env.relu", override=True)
def lnumpy_relu(x: tvm.runtime.Tensor,
                out: tvm.runtime.Tensor):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)

# Baseline: bind weights and time the unoptimized model
MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)

target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(MyModuleWithParams, target)
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
# print("MyModuleWithParams Prediction:", class_names[pred_kind[0]])

ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=100)
# print("MyModuleWithParams time-cost: %g ms" % (ftimer(data_nd).mean * 1000))

# Step 1: extract linear0 and rename to "main" so tune_tir can treat it
# as a standalone module — tune_tir expects a module with a "main" function
mod_linear = tvm.IRModule.from_expr(MyModuleMixture["linear0"].with_attr("global_symbol", "main"))
# print(mod_linear.script())

# Step 2: auto-tune linear0 (no manual space — TVM generates it automatically)
database = ms.tune_tir(
    mod=mod_linear,
    target=tvm.target.Target({"kind": "llvm", "num-cores": 1}),
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
)
sch = ms.tir_integration.compile_tir(database, mod_linear, tvm.target.Target({"kind": "llvm", "num-cores": 1}))

# Step 3: rename back to "linear0" and plug into the full module
MyModuleWithParams2 = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
new_func = sch.mod["main"].with_attr("global_symbol", "linear0")  # rename main → linear0
gv = MyModuleWithParams2.get_global_var("linear0")                 # pointer to original
MyModuleWithParams2.update_func(gv, new_func)                      # swap in optimized version
print(MyModuleWithParams2.script())

target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(MyModuleWithParams2, target)
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleWithParams2 Prediction:", class_names[pred_kind[0]])

ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=50)

print("MyModuleWithParams2 time-cost: %g ms" % (ftimer(data_nd).mean * 1000))
