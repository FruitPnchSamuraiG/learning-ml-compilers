import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.s_tir import meta_schedule as ms

# =============================================================================
# Chapter 6: GPU Acceleration (Part 1)
# https://book.mlc.ai/chapter_gpu_acceleration/index.html
#
# Previous chapters ran everything on CPU. GPU has thousands of cores that
# execute in parallel — the trick is mapping our loop structure to GPU threads.
#
# NOTE: tvm.compile(target="cuda") generates real CUDA kernels and works here.
# Executing them requires GPU tensors (tvm.cuda(0) device) and a live CUDA
# runtime. Execution blocks are commented out — compilation is the learning.
# =============================================================================

# -----------------------------------------------------------------------------
# 6.1. Vector Addition
#
# GPU programming model:
#   thread  = smallest unit, runs one element
#   block   = group of threads that share an SM (Stream Multiprocessor)
#   grid    = all blocks together (the whole kernel launch)
#
# threadIdx.x = which thread within a block (fine-grained)
# blockIdx.x  = which block in the grid (coarse-grained)
#
# Splitting 1024 → [None, 128] gives 8 blocks × 128 threads.
# bind() tells TVM which loop variable maps to which GPU index.
# Every element runs in its own thread — all 1024 run simultaneously.
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyModuleVecAdd:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"),
             B: T.Buffer((1024,), "float32"),
             C: T.Buffer((1024,), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i in T.grid(1024):
            with T.sblock("C"):
                vi = T.axis.remap("S", [i])
                C[vi] = A[vi] + B[vi]


sch = tvm.s_tir.Schedule(MyModuleVecAdd)
block_C = sch.get_sblock("C")
i, = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, 128])
# sch.mod.show()

sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
# sch.mod.show()

rt_mod = tvm.compile(sch.mod, target="cuda")
print("Compiled vector add successfully")

# Commented out: execution requires tensors on GPU memory (tvm.cuda(0) device)
# A_nd = tvm.runtime.tensor(A_np, device=tvm.cuda(0))
# B_nd = tvm.runtime.tensor(B_np, device=tvm.cuda(0))
# C_nd = tvm.runtime.tensor(np.zeros((1024,), dtype="float32"), device=tvm.cuda(0))
# rt_mod["main"](A_nd, B_nd, C_nd)

# -----------------------------------------------------------------------------
# 6.2. Window Sum with Shared Memory
#
# B[i] = A[i] + A[i+1] + A[i+2]
#
# Problem: neighboring threads read overlapping elements of A from global
# memory — slow, repeated reads.
#
# Solution: shared memory. Load A into a shared buffer once cooperatively
# (all threads in the block participate), then every thread reads from fast
# shared memory instead of slow global memory.
#
# cache_read(block, idx, "shared") → creates a new block that loads A into
#   a shared buffer. Must be placed at the right loop level with compute_at().
# compute_at(block, loop) → schedules the load to run inside that loop level.
# bind(ax1, "threadIdx.x") → the load is spread across threads (cooperative).
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyModuleWindowSum:
    @T.prim_func
    def main(A: T.Buffer((1027,), "float32"),
             B: T.Buffer((1024,), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i in T.grid(1024):
            with T.sblock("B"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi] + A[vi + 1] + A[vi + 2]


nthread = 128
sch = tvm.s_tir.Schedule(MyModuleWindowSum)
block_B = sch.get_sblock("B")
i, = sch.get_loops(block=block_B)
i0, i1 = sch.split(i, [None, nthread])

sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")

# Load A tile into shared memory — placed at the block level (i0) so each
# thread block cooperatively loads its chunk before computing B
A_shared = sch.cache_read(block_B, read_buffer_index=0, storage_scope="shared")
sch.compute_at(A_shared, i0)

# Spread the shared memory load across threads — each thread loads one element
ax = sch.get_loops(A_shared)[-1]
ax0, ax1 = sch.split(ax, [None, nthread])
sch.bind(ax1, "threadIdx.x")
sch.mod.show()

rt_mod = tvm.compile(sch.mod, target="cuda")
print("Compiled window sum successfully")
# print(rt_mod.mod.imported_modules[0].get_source())   # would show raw CUDA C — API changed in this version

# -----------------------------------------------------------------------------
# 6.3. Matrix Multiplication — Optimization 1: Local Blocking (Registers)
#
# Split the output into tiles: each thread computes an 8×8 tile and accumulates
# into local registers (cache_write "local" = register-level storage).
# This avoids writing partial sums back to slow global memory on every k step.
#
# cache_write(block, 0, "local") → C_local: each thread owns its tile in regs
# reverse_compute_at(C_local, j1) → write C_local → C happens inside j1 tile
# decompose_reduction(block_C, k0) → separates zeroing from accumulation
#
# Thread/block mapping:
#   i0, j0 → blockIdx.y, blockIdx.x  (which output tile)
#   i1, j1 → threadIdx.y, threadIdx.x (which thread within block)
#   i2, j2 → local tile (each thread's private 8×8 output)
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"),
             B: T.Buffer((1024, 1024), "float32"),
             C: T.Buffer((1024, 1024), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def blocking(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_sblock("C")
    C_local = sch.cache_write(block_C, 0, "local")   # accumulate in registers

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)   # write registers → global after j1 tile

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")
    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch

sch = tvm.s_tir.Schedule(MyModuleMatmul)
sch = blocking(sch, 8, 8, 8, 8, 4)
sch.mod.show()

rt_mod = tvm.compile(sch.mod, target="cuda")
print("Compiled matmul blocking successfully")

# Commented out: needs live CUDA device to run and benchmark
# dev = tvm.cuda(0)
# A_nd = tvm.runtime.tensor(A_np, device=dev)
# B_nd = tvm.runtime.tensor(B_np, device=dev)
# C_nd = tvm.runtime.tensor(np.zeros((1024, 1024), dtype="float32"), device=dev)
# num_flop = 2 * 1024 * 1024 * 1024
# evaluator = rt_mod.mod.time_evaluator("main", dev, number=10)
# print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))

# -----------------------------------------------------------------------------
# 6.4. Matrix Multiplication — Optimization 2: Shared Memory Blocking
#
# Optimization 1 still reads A and B from slow global memory on every k step.
# Optimization 2: load a tile of A and B into shared memory cooperatively
# before computing — all threads in the block then read from fast shared memory.
#
# cache_read_and_coop_fetch:
#   cache_read(block, idx, "shared") → shared buffer for A or B
#   compute_at(read_cache, k0) → load happens once per k0 tile
#   fuse + split → spread the load across all nthread threads
#   vectorize(vec) → each thread loads 4 elements at once
#   bind(tx, "threadIdx.x") → cooperative: all threads participate in loading
#
# fuse(i1, j1) → single threadIdx.x for both i and j thread dims, so all
#   nthread = tile_block_y * tile_block_x threads cooperate on the load.
# -----------------------------------------------------------------------------

def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
    sch.compute_at(block=read_cache, loop=read_loc)
    # vectorized cooperative fetch: fuse spatial dims, split into thread × vec lanes
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    sch.vectorize(vec)
    sch.bind(tx, "threadIdx.x")


def blocking_with_shared(
    sch,
    tile_local_y,
    tile_local_x,
    tile_block_y,
    tile_block_x,
    tile_k):
    block_C = sch.get_sblock("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    # Fuse thread dims so all threads cooperate on the shared memory load
    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)   # load A tile
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)   # load B tile
    sch.decompose_reduction(block_C, k0)

    return sch

sch = tvm.s_tir.Schedule(MyModuleMatmul)
sch = blocking_with_shared(sch, 8, 8, 8, 8, 8)
sch.mod.show()

rt_mod = tvm.compile(sch.mod, target="cuda")
print("Compiled matmul shared memory blocking successfully")

# Commented out: needs live CUDA device to run and benchmark
# dev = tvm.cuda(0)
# evaluator = rt_mod.mod.time_evaluator("main", dev, number=10)
# print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))

# -----------------------------------------------------------------------------
# 6.5. Auto-Tuning for GPU with MetaSchedule
#
# Same tune_tir from Chapter 4, now targeting a specific GPU.
# The search space covers tile sizes, thread counts, shared memory usage,
# vectorization widths, and unroll factors — all GPU-specific knobs.
# TVM finds the best configuration for the exact GPU hardware automatically.
#
# Commented out: tune_tir benchmarks on real hardware, needs a live GPU.
# The target "nvidia/tesla-p100" is a preset with known hardware parameters.
# -----------------------------------------------------------------------------

# database = ms.tune_tir(
#     mod=MyModuleMatmul,
#     target="nvidia/tesla-p100",
#     max_trials_global=64,
#     num_trials_per_iter=64,
#     work_dir="./tune_tmp",
# )
# sch = ms.tir_integration.compile_tir(database, MyModuleMatmul, "nvidia/tesla-p100")
# sch.mod.show()

# rt_mod = tvm.compile(sch.mod, target="nvidia/tesla-p100")
# dev = tvm.cuda(0)
# evaluator = rt_mod.mod.time_evaluator("main", dev, number=10)
# print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
