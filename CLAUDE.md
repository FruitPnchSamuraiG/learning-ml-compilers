# CLAUDE.md

This is a learning project for the MLC (Machine Learning Compilation) course. The user is new to systems programming and ML compilers — explanations should be approachable, not terse.

## Project

- Working through https://mlc.ai course exercises chapter by chapter

## Environment

- Python managed with `uv` (Python 3.11)
- Run scripts with `uv run <script.py>`
- Install packages with `uv add <package>`
- Main ML package: `mlc-ai-nightly-cpu` from https://mlc.ai/wheels

## Key package imports

```python
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
```

## Style

- Keep code simple and close to the course examples
- Prefer clarity over cleverness — this is for learning
- Add short comments explaining the *why* when something is non-obvious
