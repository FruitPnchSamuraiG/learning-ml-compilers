import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tirx as T

# =============================================================================
# Chapter 6: GPU Acceleration (Part 2) — Hardware Specialization & Tensorization
# https://book.mlc.ai/chapter_gpu_acceleration/index.html
#
# Part 1 mapped loops to GPU thread/block indices (general parallelism).
# Part 2 goes further: mapping ops to *specialized* hardware instructions
# like tensor cores, which do an entire 16×16×16 matmul in ONE instruction.
#
# The trend: scalar (1 element) → vector (8-16 elements) → tensor (16×16 region).
# ML workloads are mostly matmul, so hardware vendors built dedicated units for it.
# =============================================================================

# -----------------------------------------------------------------------------
# 6.2.1. lnumpy_tmm — numpy simulation of tensor-core style hardware
#
# Simulates the three special operations a tensor accelerator exposes:
#   accel_fill_zero → zero the accumulator register before a new output tile
#   accel_dma_copy  → fast memory transfer: DRAM → hardware register
#   accel_tmm_add   → ONE tensor instruction: C += A @ B.T over a 16×16×16 tile
#
# The key insight: the hardware works on 16×16 *regions*, not individual elements.
# So we must reorganize the matmul into 16×16 tiles to match what the hardware does.
# The outer loops (i,j,k over 64 tiles each) are software; the inner 16×16×16 is HW.
# -----------------------------------------------------------------------------

def accel_fill_zero(C):
    C[:] = 0

def accel_tmm_add(C, A, B):
    C[:] += A @ B.T

def accel_dma_copy(reg, dram):
    reg[:] = dram[:]

def lnumpy_tmm(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    C_accumulator = np.empty((16, 16), dtype="float32")
    A_reg = np.empty((16, 16), dtype="float32")
    B_reg = np.empty((16, 16), dtype="float32")

    for i in range(64):       # 64 × 16 = 1024 rows of output
        for j in range(64):   # 64 × 16 = 1024 cols of output
            accel_fill_zero(C_accumulator[:,:])
            for k in range(64):   # 64 × 16 = 1024 reduction steps
                accel_dma_copy(A_reg[:], A[i * 16 : i * 16 + 16, k * 16 : k * 16 + 16])
                accel_dma_copy(B_reg[:], B[j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
                accel_tmm_add(C_accumulator[:,:], A_reg, B_reg)
            accel_dma_copy(C[i * 16 : i * 16 + 16, j * 16 : j * 16 + 16], C_accumulator[:,:])

dtype = "float32"
a_np = np.random.rand(1024, 1024).astype(dtype)
b_np = np.random.rand(1024, 1024).astype(dtype)
c_tmm = a_np @ b_np.T

c_np = np.empty((1024, 1024), dtype="float32")
lnumpy_tmm(a_np, b_np, c_np)
np.testing.assert_allclose(c_np, c_tmm, rtol=1e-5)
print("lnumpy_tmm passed")

# -----------------------------------------------------------------------------
# 6.2.2. MatmulBlockModule — TensorIR version of the tiled structure above
#
# The outer sblock "tmm-16x16" operates over tile indices (i0, j0, k0).
# The inner loops (i1, j1, k1) over 16×16×16 are what the hardware will replace.
# This is the starting structure before blockize() collapses the inner loops.
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MatmulBlockModule:
    @T.prim_func
    def main(
        A: T.Buffer((1024, 1024), "float32"),
        B: T.Buffer((1024, 1024), "float32"),
        C: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i0, j0, k0 in T.grid(64, 64, 64):
            with T.sblock("tmm-16x16"):
                vi0, vj0, vk0 = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    for i1, j1 in T.grid(16, 16):
                        with T.sblock("tmm_init"):
                            vi1, vj1 = T.axis.remap("SS", [i1, j1])
                            C[vi0 * 16 + vi1, vj0 * 16 + vj1] = T.float32(0)

                for i1, j1, k1 in T.grid(16, 16, 16):
                    with T.sblock("tmm"):
                        vi1, vj1, vk1 = T.axis.remap("SSR", [i1, j1, k1])
                        C[vi0 *16 + vi1, vj0 * 16 + vj1] += \
                            A[vi0 * 16 + vi1, vk0 * 16 + vk1] * B[vj0 * 16 + vj1, vk0 * 16 + vk1]

# MatmulBlockModule.show()

a_nd = tvm.runtime.tensor(a_np)
b_nd = tvm.runtime.tensor(b_np)
c_nd = tvm.runtime.tensor(np.empty((1024, 1024), dtype="float32"))

lib = tvm.compile(MatmulBlockModule, target="llvm")
lib["main"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_nd.numpy(), c_tmm, rtol=1e-5)
print("MatmulBlockModule passed")

# Reorder to prepare for blockize: outer tile loops first, inner 16×16 loops last
sch = tvm.s_tir.Schedule(MatmulBlockModule)
block_mm = sch.get_sblock("tmm-16x16")
i, j, k = sch.get_loops(block_mm)
i0, i1 = sch.split(i, [None, 4])
sch.reorder(i0, j, i1, k)
# sch.mod.show()

# -----------------------------------------------------------------------------
# 6.2.3. Tensorization — the full pipeline on MatmulModule
#
# Starting from a flat scalar matmul, we apply:
# 1. split (i,j,k) into 16-wide tiles so inner loops cover exactly 16×16×16
# 2. reorder: outer tile loops first, inner 16×16×16 loops last
# 3. blockize(ii): collapse the inner loops into a single tensorized block
#    → tells TVM "treat this 16×16×16 region as ONE operation"
# 4. add special memory scopes: A_reg, B_reg, accumulator
#    → "global.A_reg" means: accessible globally, but tagged as hardware register
#    → cache_read / cache_write create load/store blocks; compute_at places them
# 5. decompose_reduction: separate zeroing from accumulation
# 6. tensorize(block_mm, "tmm16"): replace the tensorized block with the hardware call
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer((1024, 1024), "float32"),
        B: T.Buffer((1024, 1024), "float32"),
        C: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.sblock("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vj, vk]


sch = tvm.s_tir.Schedule(MatmulModule)
i, j, k = sch.get_loops("matmul")

# Step 1 & 2: tile into 16-wide chunks, reorder so inner loops are i1,j1,k1
i, ii = sch.split(i, factors=[None, 16])
j, ji = sch.split(j, factors=[None, 16])
k, ki = sch.split(k, factors=[None, 16])
sch.reorder(i, j, k, ii, ji, ki)
# sch.mod.show()

# Step 3: blockize — collapse inner 16×16×16 into one tensorized block
# After this, block_mm represents a 16×16 *region* not a single element
block_mm = sch.blockize(ii)
# sch.mod.show()

# Step 4: special memory — load A and B into hardware registers before compute
# "global.A_reg" scope tags the buffer so the backend knows to use HW registers
A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
sch.compute_at(A_reg, k)   # load A tile once per k step
sch.compute_at(B_reg, k)   # load B tile once per k step

# accumulator: C's partial sums stay in hardware register until tile is done
write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
sch.reverse_compute_at(write_back_block, j)   # write accumulator → C after j tile
# sch.mod.show()

# -----------------------------------------------------------------------------
# TensorIntrin — bridges TVM's IR and the actual hardware instruction
#
# tmm16_desc: describes WHAT the instruction computes (16×16×16 matmul).
#   TVM uses this to find and match tensorized blocks in the IR.
#
# tmm16_impl: describes HOW to execute it — calls T.call_extern("tmm16", ...)
#   which calls the actual hardware instruction (or here, a C simulation).
#   strides (sa, sb, sc) let the implementation handle non-contiguous memory.
# -----------------------------------------------------------------------------

@T.prim_func
def tmm16_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16, scope="global.accumulator")

    with T.sblock("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j, k in T.grid(16, 16, 16):
            with T.sblock(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def tmm16_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    sa = T.int32()
    sb = T.int32()
    sc = T.int32()
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, strides=[sa, 1], scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, strides=[sb, 1], scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16, strides=[sc, 1], scope="global.accumulator")

    with T.sblock("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.call_extern(
                "tmm16",
                C.access_ptr("w"),
                A.access_ptr("r"),
                B.access_ptr("r"),
                sa,
                sb,
                sc,
                dtype="int32",
            )
        )

tvm.s_tir.TensorIntrin.register("tmm16", tmm16_desc, tmm16_impl)

# Step 5: decompose reduction — separates zeroing accumulator from update
sch.decompose_reduction(block_mm, k)
# sch.mod.show()

# Step 6: tensorize — replaces the matched block with the hardware instruction call
sch.tensorize(block_mm, "tmm16")
# sch.mod.show()

# -----------------------------------------------------------------------------
# External C kernel — CPU simulation of the tensor instruction
#
# On real hardware this would be a wmma (warp matrix multiply accumulate) call
# or similar. Here we compile a C function via clang into LLVM IR and import it.
# pragma_import_llvm tells TVM to link this LLVM IR into the compiled module.
# -----------------------------------------------------------------------------

def tmm_kernel():
    cc_code = """
      extern "C" int tmm16(float *cc, float *aa, float *bb, int stride_a, int stride_b, int stride_c) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                for (int k = 0; k < 16; ++k) {
                    cc[i * stride_c + j] += aa[i * stride_a + k] * bb[j * stride_b + k];
                }
            }
        }
        return 0;
      }
    """
    from tvm.contrib import clang, utils

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    ll_code = clang.create_llvm(cc_code, output=ll_path)
    return ll_code

sch.annotate(i, "pragma_import_llvm", tmm_kernel())
