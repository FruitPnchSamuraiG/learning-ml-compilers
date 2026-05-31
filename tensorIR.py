import numpy as np
import tvm
from tvm import te
from tvm.ir.module import IRModule
from tvm.script import tirx as T

# =============================================================================
# Chapter 2: Tensor Program Abstraction
# https://book.mlc.ai/chapter_tensor_program/index.html
# =============================================================================

# -----------------------------------------------------------------------------
# 2.4.2. Prelude
# Reference data: random 128x128 matrices, ground truth computed with numpy
# -----------------------------------------------------------------------------

dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)

# a @ b is equivalent to np.matmul(a, b)
c_mm_relu = np.maximum(a_np @ b_np, 0)

# -----------------------------------------------------------------------------
# 2.4.3. Learning one Tensor Program Abstraction – TensorIR
#
# lnumpy_mm_relu is a low-level numpy reference implementation of:
#   C = ReLU(A @ B)
# Written with explicit loops to mirror what TensorIR expresses underneath.
# -----------------------------------------------------------------------------

def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)


c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)


# TensorIR version of mm_relu.
# T.Buffer, T.grid, T.axis, T.sblock are TensorIR constructs that express
# the same loop structure but with annotations the compiler can reason about.
# spatial = output dimension (can be parallelized), reduce = accumulation axis.
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tirx.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.sblock("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


# T.axis.remap is syntactic sugar: "SSR" means spatial, spatial, reduce.
# Equivalent to MyModule but more concise — remap infers axes from loop vars.
@tvm.script.ir_module
class MyModuleWithAxisRemapSugar:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tirx.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.sblock("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


# A module can hold multiple functions.
# Here mm and relu are separated into two prim_funcs instead of one combined.
@tvm.script.ir_module
class MyModuleWithTwoFunctions:
    @T.prim_func
    def mm(A: T.Buffer((128, 128), "float32"),
           B: T.Buffer((128, 128), "float32"),
           Y: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm", "tirx.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

    @T.prim_func
    def relu(A: T.Buffer((128, 128), "float32"),
             B: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "relu", "tirx.noalias": True})
        for i, j in T.grid(128, 128):
            with T.sblock("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0))


# -----------------------------------------------------------------------------
# 2.4.4. Transformation
#
# lnumpy_mm_relu_v2: same computation as v1 but j is split into j0 (outer,
# steps of 4) and j1 (inner, 0–3). This is exactly what Schedule.split() does
# programmatically on TensorIR — no rewriting required.
# -----------------------------------------------------------------------------

def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j0 in range(32):
            for k in range(128):
                for j1 in range(4):
                    j = j0 * 4 + j1
                    if k == 0:
                        Y[i, j] = 0
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v2(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)

# Schedule applies a sequence of transformations to a TensorIR module.
# Each step below mirrors a manual loop rewrite — but done programmatically.
sch = tvm.s_tir.Schedule(MyModuleWithAxisRemapSugar)

block_Y = sch.get_sblock("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)

# Split j into (j0, j1) where j1 has size 4 — equivalent to v2 above
j0, j1 = sch.split(j, factors=[None, 4])
# print(sch.mod.script())

# Reorder loops: move k between j0 and j1 for better cache behavior
sch.reorder(j0, k, j1)
# print(sch.mod.script())

# Move the C (relu) block to be computed right after j0, fusing it closer
sch.reverse_compute_at(block_C := sch.get_sblock("C", "mm_relu"), j0)
# print(sch.mod.script())

# Separate Y's init (zeroing) from the update (accumulation) into distinct loops
sch.decompose_reduction(block_Y, k)
# print(sch.mod.script())

# lnumpy_mm_relu_v3: numpy equivalent of the fully transformed schedule above.
# Matches the loop structure after split + reorder + reverse_compute_at + decompose.
def lnumpy_mm_relu_v3(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j0 in range(32):
            # Y_init: zero out this tile before accumulating
            for j1 in range(4):
                j = j0 * 4 + j1
                Y[i, j] = 0
            # Y_update: accumulate over k
            for k in range(128):
                for j1 in range(4):
                    j = j0 * 4 + j1
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
            # C: apply relu to this tile immediately (fused)
            for j1 in range(4):
                j = j0 * 4 + j1
                C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v3(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)

# -----------------------------------------------------------------------------
# 2.4.5. Build and Run
#
# tvm.compile turns a TensorIR module into executable machine code.
# tvm.runtime.tensor wraps a numpy array for the TVM runtime to use.
# -----------------------------------------------------------------------------

rt_lib = tvm.compile(MyModule, target="llvm")

a_nd = tvm.runtime.tensor(a_np)
b_nd = tvm.runtime.tensor(b_np)
c_nd = tvm.runtime.tensor(np.empty((128, 128), dtype="float32"))
# print(type(c_nd))

func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# Compile the transformed schedule and verify it still produces correct output
rt_lib_after = tvm.compile(sch.mod, target="llvm")
rt_lib_after["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# time_evaluator measures execution time — useful for comparing transformations
f_timer_before = rt_lib.mod.time_evaluator("mm_relu", tvm.cpu())
# print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.mod.time_evaluator("mm_relu", tvm.cpu())
# print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)

# -----------------------------------------------------------------------------
# 2.4.7. TensorIR Functions as Results of Transformations
#
# Wrapping the schedule steps in a function lets us try different jfactors
# and compare their performance — this is the core of the MLC search loop.
# -----------------------------------------------------------------------------

def transform(mod, jfactor):
    sch = tvm.s_tir.Schedule(mod)
    block_Y = sch.get_sblock("Y", func_name="mm_relu")
    i, j, k = sch.get_loops(block_Y)
    j0, j1 = sch.split(j, factors=[None, jfactor])
    sch.reorder(j0, k, j1)
    block_C = sch.get_sblock("C", "mm_relu")
    sch.reverse_compute_at(block_C, j0)
    return sch.mod

mod_transformed = transform(MyModule, jfactor=8)

rt_lib_transformed = tvm.compile(mod_transformed, "llvm")
f_timer_transformed = rt_lib_transformed.mod.time_evaluator("mm_relu", tvm.cpu())
# print("Time cost of transformed mod_transformed %g sec" % f_timer_transformed(a_nd, b_nd, c_nd).mean)
# print(mod_transformed.script())

# -----------------------------------------------------------------------------
# 2.4.6. Ways to Create and Interact with TensorIR
#
# Besides writing TensorIR by hand, you can generate it from Tensor Expression (te).
# te describes *what* to compute (declarative); TensorIR describes *how* (imperative).
# te.create_prim_func converts a te computation into a TensorIR prim_func.
# -----------------------------------------------------------------------------

A = te.placeholder((128, 128), "float32", name="A")
B = te.placeholder((128, 128), "float32", name="B")
k = te.reduce_axis((0, 128), "k")
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name="C")

te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
MyModuleFromTE = tvm.IRModule({"mm_relu": te_func})
print(MyModuleFromTE.script())
