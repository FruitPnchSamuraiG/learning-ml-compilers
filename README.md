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

## Code

- `main.py` — Chapter 2 notes and examples (coded by me, annotated by Claude)
- `tensorIR_EX.py` — Chapter 2 exercises: element-wise add, broadcasting, 2D convolution, bmm_relu transformation (coded by me, annotated by Claude)


