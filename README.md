# Learning ML Compilers

Working through the [MLC (Machine Learning Compilation) course](https://mlc.ai/chapter_introduction/index.html) step by step.

> **Note:** The course is fairly old and the setup instructions are outdated. If you're following along, here's what actually works:
>
> - Install via uv, not plain pip. The course tells you to run `python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu` but with uv you need to configure the find-links source in `pyproject.toml` (see below).
> - `pytest` is required as a dependency even if you're not running tests — tvm imports it internally and will crash without it.
> - Pin your Python version to `>=3.11,<3.13` — the package doesn't support 3.13 yet and uv will fail to resolve without this constraint.

## Key Concepts

**TVM** — The overall ML compiler framework. Takes a model and compiles it to run efficiently on hardware. Everything else below lives inside TVM.

**TensorIR** — The intermediate representation TVM works with internally. Every computation is expressed as three things: buffers (the data), loops (how to iterate), and compute statements (the math).

**TVMScript** — The Python-like syntax used to write TensorIR by hand. Human-readable way to express TensorIR directly.
```python
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer(...), ...):
        ...
```

**Tensor Expression (TE)** — A higher-level, declarative way to describe computations without writing loops. TVM generates the TensorIR loops automatically.
```python
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
```

**IRModule** — A container that holds one or more tensor functions. TVMScript and TE both produce an IRModule.

**Schedule** — A transformation tool that wraps an IRModule and lets you restructure its loops without changing the result.

**Schedule Primitives** — The individual transformation operations applied through a Schedule:
```
sch.split()                → split one loop into two
sch.reorder()              → change the order of loops
sch.reverse_compute_at()   → move a block closer to where its output is used
sch.decompose_reduction()  → separate the init (zeroing) from the update (accumulation)
```

**How it all fits together:**
```
WRITE    TVMScript or TE  →  TensorIR stored in IRModule
TRANSFORM  Schedule + primitives  →  optimized IRModule
COMPILE  tvm.compile(mod, target="llvm")  →  runnable machine code
RUN      rt_lib["mm_relu"](a, b, c)  →  executes on hardware
VERIFY   np.testing.assert_allclose(...)  →  checks correctness
```

## Chapter 2 Exercises — What I Learned

### Writing TensorIR

Every TensorIR function has three parts: buffers (the data), loops (iteration), and compute statements (the math). Axes are annotated as spatial (`S`) or reduce (`R`) — spatial axes are output dimensions that can be parallelized, reduce axes are accumulation dimensions that cannot.

```python
vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # j loops = spatial, k = reduce
with T.init():
    Y[vi, vj] = 0       # runs once per (i,j) before the k reduction
Y[vi, vj] += A[vi, vk] * B[vk, vj]
```

### Schedule Primitives

| Primitive | What it does |
|---|---|
| `split(loop, factors=[a, b])` | Break one loop into two (outer × inner) |
| `reorder(l1, l2, ...)` | Change the order of loops |
| `parallel(loop)` | Run this loop across CPU cores |
| `vectorize(loop)` | Use SIMD instructions for this loop |
| `unroll(loop)` | Eliminate loop overhead by inlining iterations |
| `reverse_compute_at(block, loop)` | Move a consumer block inside a producer's loop |
| `decompose_reduction(block, loop)` | Separate init (zeroing) from update (accumulation) |

### `get_loops` always returns all surrounding loops

```python
n, i, j, k = sch.get_loops(Y)
# returns every loop from outermost → innermost that wraps the block
```

After `reverse_compute_at(C, j0)`, calling `get_loops(C)` returns `[n, i, j0, ax0]` — all loops now surrounding C, not just its own.

### Ordering rules that matter

1. `parallel` must come **before** `decompose_reduction` — TVM needs the whole reduction block intact to verify the parallel is safe.
2. `reorder` must come **before** `reverse_compute_at` — the consumer block needs to see the final loop order when being moved.
3. `split` must come **before** `reorder` — the new loops need to exist before you can reorder them.

### Why each transformation helps performance

```
split j (chunks of 8)   → right size for SIMD vectorization
split k (chunks of 4)   → right size for loop unrolling
reorder (j1 innermost)  → cache-friendly memory access
parallel (n)            → spreads batches across CPU cores
vectorize (j1)          → processes 8 elements per SIMD instruction
unroll (k1)             → removes 4-iteration loop overhead
reverse_compute_at      → relu reads Y while it's still in cache
decompose_reduction     → clean separation of init and update phases
```

**Core insight:** The same computation expressed with different loop structure produces very different performance. TensorIR makes the loop structure explicit so the compiler — or you — can transform it to match what the hardware is good at (multiple cores, SIMD units, cache hierarchy).

---

## Code

- `tensorIR.py` — Chapter 2 notes and examples (coded by me, annotated by Claude)
- `tensorIR_EX.py` — Chapter 2 exercises: element-wise add, broadcasting, 2D convolution, bmm_relu transformation (coded by me, annotated by Claude)


