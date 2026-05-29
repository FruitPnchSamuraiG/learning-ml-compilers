import IPython
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tirx as T
import torch

# =============================================================================
# Chapter 2 Exercises — TensorIR
# https://book.mlc.ai/chapter_tensor_program/tensorir_exercises.html
# =============================================================================

# -----------------------------------------------------------------------------
# Exercise 1: Element-wise Add (same shape)
# Write TensorIR for C[i,j] = A[i,j] + B[i,j]
# -----------------------------------------------------------------------------

# High-level numpy reference
a = np.arange(16).reshape(4, 4)
b = np.arange(16, 0, -1).reshape(4, 4)
c_np = a + b
# print(c_np)

# Low-level numpy: explicit loops to show what TensorIR will express
def lnumpy_add(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    for i in range(4):
        for j in range(4):
            c[i, j] = a[i, j] + b[i, j]
c_lnumpy = np.empty((4, 4), dtype=np.int64)
lnumpy_add(a, b, c_lnumpy)
# print(c_lnumpy)


# TensorIR version: both A and B are (4,4), straightforward spatial iteration
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4, 4), "int64"),
          B: T.Buffer((4, 4), "int64"),
          C: T.Buffer((4, 4), "int64")):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.sblock("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

rt_lib = tvm.compile(MyAdd, target="llvm")
a_tvm = tvm.runtime.tensor(a)
b_tvm = tvm.runtime.tensor(b)
c_tvm = tvm.runtime.tensor(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)

# -----------------------------------------------------------------------------
# Exercise 2: Element-wise Add with Broadcasting
# B is shape (4,) — it gets broadcast across each row of A (shape 4x4)
# C[i,j] = A[i,j] + B[j]
# -----------------------------------------------------------------------------

a = np.arange(16).reshape(4,4)
b = np.arange(4, 0, -1).reshape(4)   # 1D, will broadcast along j
c_np = a + b


# Key difference from MyAdd: B is a 1D buffer indexed only by vj, not vi
@tvm.script.ir_module
class MyAddEx:
    @T.prim_func
    def add(A: T.Buffer((4, 4), "int64"),
            B: T.Buffer((4, ), "int64"),
            C: T.Buffer((4, 4), "int64")):
        T.func_attr({"global_symbol": "add", "tirx.noalias": True})
        for i, j in T.grid(4, 4):
            with T.sblock("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vj]   # B[vj] — no row index


rt_lib = tvm.compile(MyAddEx, target="llvm")
a_tvm = tvm.runtime.tensor(a)
b_tvm = tvm.runtime.tensor(b)
c_tvm = tvm.runtime.tensor(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)


# -----------------------------------------------------------------------------
# Exercise 3a: 2D Convolution
# Compute conv2d using TensorIR — verified against PyTorch's conv2d output
#
# Shape conventions:
#   A (input):  (N, CI, H, W)   — batch, input channels, height, width
#   B (kernel): (CO, CI, K, K)  — output channels, input channels, kernel size
#   C (output): (N, CO, OUT_H, OUT_W)
#
# The reduction axes (q, di, dj) sum over input channels and kernel positions.
# SSSSRRR in remap = 4 spatial + 3 reduce axes.
# -----------------------------------------------------------------------------

N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1     # valid (no padding) convolution output size
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

# PyTorch reference output to verify against
data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
# print(conv_torch)

@tvm.script.ir_module
class MyConv:
  @T.prim_func
  def conv(
            A: T.Buffer((N, CI, H, W), "int64"),
            B: T.Buffer((CO, CI, K, K), "int64"),
            C: T.Buffer((N, CO, OUT_H, OUT_W), "int64")):
        T.func_attr({"global_symbol": "conv", "tirx.noalias": True})
        # Outer loops: iterate over every output element
        for n, c, i, j in T.grid(N, CO, OUT_H, OUT_W):
            # Inner loops: reduce over input channels (q) and kernel window (di, dj)
            for q, di, dj in T.grid(CI, K, K):
                with T.sblock("C"):
                    vn, vc, vi, vj, vq, vdi, vdj = T.axis.remap("SSSSRRR", [n, c, i, j, q, di, dj])
                    with T.init():
                        C[vn, vc, vi, vj] = T.int64(0)
                    # Slide the kernel: input position = output position + kernel offset
                    C[vn, vc, vi, vj] += A[vn, vq, vi+vdi, vj+vdj] * B[vc, vq, vdi, vdj]

rt_lib = tvm.compile(MyConv, target="llvm")
data_tvm = tvm.runtime.tensor(data)
weight_tvm = tvm.runtime.tensor(weight)
conv_tvm = tvm.runtime.tensor(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
# print(conv_tvm.numpy())
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)


# -----------------------------------------------------------------------------
# Exercise 3b: Understanding Schedule Transformations on a Simple Add
# Applying split, parallel, unroll, vectorize to a 4x4 add as a warm-up
# before tackling the full bmm_relu transformation below.
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4, 4), "int64"),
          B: T.Buffer((4, 4), "int64"),
          C: T.Buffer((4, 4), "int64")):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.sblock("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

sch = tvm.s_tir.Schedule(MyAdd)
block = sch.get_sblock("C", func_name="add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[2, 2])  # split i into 2 outer * 2 inner
sch.parallel(i0)    # run outer i loop across CPU threads
sch.unroll(i1)      # unroll inner i loop (compiler expands iterations manually)
sch.vectorize(j)    # use SIMD instructions for the j loop
# print(sch.mod.script())


# -----------------------------------------------------------------------------
# Exercise 4: Transform Batch Matrix Multiply + ReLU (bmm_relu)
#
# Goal: transform MyBmmRelu to match TargetModule using Schedule primitives.
# Transformations applied:
#   split j into (j0=16, j1=8) and k into (k0=32, k1=4)
#   reorder loops so j1 is innermost for vectorization
#   parallelize over batch dimension n
#   move C (relu) block next to j0 so it's computed tile-by-tile
#   decompose_reduction separates Y_init (zeroing) from Y_update (accumulation)
#   vectorize the init and relu loops, unroll k1
# -----------------------------------------------------------------------------

# Numpy reference for bmm_relu: batched matmul then ReLU
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)


# Unoptimized TensorIR — this is the starting point before transformation
@tvm.script.ir_module
class MyBmmRelu:
    @T.prim_func
    def bmm_relu(
        A: T.Buffer((16, 128, 128), "float32"),
        B: T.Buffer((16, 128, 128), "float32"),
        C: T.Buffer((16, 128, 128), "float32" )):
        T.func_attr({"global_symbol": "bmm_relu", "tirx.noalias": True})
        Y = T.alloc_buffer((16, 128, 128), dtype="float32")
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.sblock("Y"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                with T.init():
                    Y[vn, vi, vj] = T.float32(0)
                Y[vn, vi, vj] += A[vn, vi, vk] * B[vn, vk, vj]
        for n, i, j in T.grid(16, 128, 128):
            with T.sblock("C"):
                vn, vi, vj = T.axis.remap("SSS", [n, i, j])
                C[vn, vi, vj] = T.max(Y[vn, vi, vj], 0)

a = np.arange(262144).reshape(16, 128, 128).astype(np.float32)
b = np.arange(262144, 0, -1).reshape(16, 128, 128).astype(np.float32)
c_numpy = np.empty((16, 128, 128), dtype=np.float32)

lnumpy_mm_relu_v2(a, b, c_numpy)

rt_lib = tvm.compile(MyBmmRelu, target="llvm")
a_tvm = tvm.runtime.tensor(a)
b_tvm = tvm.runtime.tensor(b)
c_tvm = tvm.runtime.tensor(np.empty((16, 128, 128), dtype=np.float32))
rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_numpy, rtol=1e-5)

# Target: what the schedule should produce after all transformations
@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tirx.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.sblock("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.sblock("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.sblock("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))


sch = tvm.s_tir.Schedule(MyBmmRelu)
# print(sch.mod.script())


# Step 1. Get blocks
Y = sch.get_sblock("Y", func_name="bmm_relu")
C = sch.get_sblock("C", func_name="bmm_relu")

# Step 2. Get loops
n, i, j, k = sch.get_loops(Y)

# Step 3. Organize the loops
j0, j1 = sch.split(j, factors=[16, 8])   # j0 tiles j into 16 groups of 8
k0, k1 = sch.split(k, factors=[32, 4])   # k0 tiles k into 32 groups of 4

sch.reorder(n, i, j0, k0, k1, j1)        # j1 innermost so it can be vectorized
sch.parallel(n)                           # each batch item runs on its own thread
sch.reverse_compute_at(C, j0)            # fuse relu into the j0 tile loop

# Step 4. decompose reduction
Y_init = sch.decompose_reduction(Y, k0)  # split zeroing out from accumulation

# Step 5. vectorize / parallel / unroll
n_init, i_init, j_init0, j_init1 = sch.get_loops(Y_init)
n_in, i_in, j_in0, j_in1 = sch.get_loops(C)

sch.vectorize(j_init1)   # SIMD for the init (zero-fill) inner loop
sch.vectorize(j_in1)     # SIMD for the relu inner loop
sch.unroll(k1)           # unroll k1 (4 iterations) — compiler inlines them

# print(sch.mod.script())

# Verify the transformed schedule exactly matches the target structure
tvm.ir.assert_structural_equal(sch.mod, TargetModule)
# print("Pass")

# Compare runtime performance before and after transformation
before_rt_lib = tvm.compile(MyBmmRelu, target="llvm")
after_rt_lib = tvm.compile(sch.mod, target="llvm")
a_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.mod.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.mod.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))
