import IPython
import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tirx as T
import torch
import torchvision
import matplotlib.pyplot as plt
import pickle as pkl

# =============================================================================
# Chapter 3: End-to-End Model Execution
# https://mlc.ai/chapter_end_to_end/index.html
#
# Goal: run a full MLP (linear → relu → linear) on FashionMNIST using TVM.
# We build up from raw numpy → low-level numpy → TensorIR + Relax → compiled.
# =============================================================================

# -----------------------------------------------------------------------------
# Dataset setup — FashionMNIST, single image inference
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

# plt.figure()
# plt.imshow(img[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
print("Class:", class_names[label[0]])

# -----------------------------------------------------------------------------
# High-level numpy MLP — the reference implementation
# data @ w.T + b is a linear layer; np.maximum(..., 0) is relu
# -----------------------------------------------------------------------------

def numpy_mlp(data, w0, b0, w1, b1):
    lv0 = data @ w0.T + b0
    lv1 = np.maximum(lv0, 0)
    lv2 = lv1 @ w1.T + b1
    return lv2

mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
res = numpy_mlp(img.reshape(1, 784),
                mlp_params["w0"],
                mlp_params["b0"],
                mlp_params["w1"],
                mlp_params["b1"])
pred_kind = res.argmax(axis=1)
print("NumPy-MLP Prediction:", class_names[pred_kind[0]])


# -----------------------------------------------------------------------------
# Low-level numpy MLP — explicit loops, mirroring what TensorIR will express
# DPS (Destination Passing Style): output buffer Z is passed in, not returned
# -----------------------------------------------------------------------------

def lnumpy_linear0(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 128), dtype="float32")
    for i in range(1):
        for j in range(128):
            for k in range(784):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(128):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_relu0(X: np.ndarray, Y: np.ndarray):
     for i in range(1):
        for j in range(128):
            Y[i, j] = np.maximum(X[i, j], 0)

def lnumpy_linear1(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 10), dtype="float32")
    for i in range(1):
        for j in range(10):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(10):
            Z[i, j] = Y[i, j] + B[j]


# Orchestrates the three primitive functions manually — each writes into a buffer
def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    lnumpy_linear1(lv1, w1, b1, out)
    return out

result =lnumpy_mlp(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"])

pred_kind = result.argmax(axis=1)
print("Low-level Numpy MLP Prediction:", class_names[pred_kind[0]])

# -----------------------------------------------------------------------------
# TensorIR + Relax module
#
# TensorIR (@T.prim_func): the individual low-level ops (linear, relu)
# Relax (@R.function): the high-level orchestrator — connects ops as a graph
#
# T.handle + T.match_buffer: used instead of T.Buffer when shapes are dynamic
#   (the sizes m, n, k are not known until runtime)
#
# call_dps_packed: bridges DPS primitive functions into the Relax graph.
#   It allocates the output buffer and calls the function — making the result
#   a trackable value in the dataflow graph.
#
# R.dataflow(): marks a region of pure computation (no side effects).
#   TVM can freely optimize inside this block.
#   R.output() declares what leaves the block.
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def relu0(x: T.handle, y: T.handle):
        # T.handle = dynamic shape — n is determined at runtime
        n = T.int64()
        X = T.match_buffer(x, (1, n), "float32")
        Y = T.match_buffer(y, (1, n), "float32")
        for i, j in T.grid(1, n):
            with T.sblock("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

    @T.prim_func
    def linear0(x: T.handle,
                w: T.handle,
                b: T.handle,
                z: T.handle):
        m, n, k = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (1, m), "float32")
        W = T.match_buffer(w, (n, m), "float32")
        B = T.match_buffer(b, (n, ), "float32")
        Z = T.match_buffer(z, (1, n), "float32")
        Y = T.alloc_buffer((1, n), "float32")
        for i, j, k in T.grid(1, n, m):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.sblock("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            # call_dps_packed: calls a DPS prim_func and returns its output as a value
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("relu0", (lv0, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("linear0", (lv1, w1, b1), R.Tensor((1, k), "float32"))
            R.output(out)   # marks what exits the dataflow block
        return out


# -----------------------------------------------------------------------------
# Numpy equivalent of call_dps_packed — shows what it does under the hood:
# allocates output buffer, calls the function with it, returns the result
# -----------------------------------------------------------------------------

def lnumpy_call_dps_packed(prim_func, inputs, shape, dtype):
    res = np.empty(shape, dtype=dtype)
    prim_func(*inputs, res)
    return res

def lnumpy_mlp_with_call_dps_packed(data, w0, b0, w1, b1):
    lv0 = lnumpy_call_dps_packed(lnumpy_linear0, (data, w0, b0), (1, 128), dtype="float32")
    lv1 = lnumpy_call_dps_packed(lnumpy_relu0, (lv0, ), (1, 128), dtype="float32")
    out = lnumpy_call_dps_packed(lnumpy_linear1, (lv1, w1, b1), (1, 10), dtype="float32")
    return out

result = lnumpy_mlp_with_call_dps_packed(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"])

pred_kind = np.argmax(result, axis=1)
print("Low-level Numpy with CallTIR Prediction:", class_names[pred_kind[0]])

# print(MyModule.script())

# Compile and run MyModule using the Relax VirtualMachine
target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(MyModule, target)
type(ex)

vm = relax.VirtualMachine(ex, tvm.cpu())

data_nd = tvm.runtime.tensor(img.reshape(1, 784))
nd_params = {k: tvm.runtime.tensor(v) for k, v in mlp_params.items()}


nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])
# print(nd_res)
pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModule Prediction:", class_names[pred_kind[0]])


# -----------------------------------------------------------------------------
# Integrating external libraries via env.* calls
#
# Instead of our own TensorIR prim_funcs, we register PyTorch functions under
# "env.linear" / "env.relu" and call them from Relax the same way.
# Zero-copy via DLPack: torch.from_dlpack shares memory — no data duplication.
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyModuleWithExternCall:
    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        # block 0
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            # Same call_dps_packed pattern — but now the names resolve to
            # PyTorch functions registered below, not TensorIR prim_funcs
            lv0 = R.call_dps_packed("env.linear", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, k), "float32"))
            R.output(out)
        return out

# Register PyTorch implementations under TVM global function names
@tvm.register_global_func("env.linear", override=True)
def torch_linear(x: tvm.runtime.Tensor,
                 w: tvm.runtime.Tensor,
                 b: tvm.runtime.Tensor,
                 out: tvm.runtime.Tensor):
    x_torch = torch.from_dlpack(x)     # zero-copy: shares memory with TVM tensor
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

target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(MyModuleWithExternCall, target)
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleWithExternCall Prediction:", class_names[pred_kind[0]])

# -----------------------------------------------------------------------------
# Mixing TensorIR and external libraries in the same module
# linear0 (first layer) = our own TensorIR prim_func
# relu and linear1 = PyTorch via env.*
# This shows MLC's flexibility: optimize specific ops while reusing others
# -----------------------------------------------------------------------------

@tvm.script.ir_module
class MyModuleMixture:
    @T.prim_func
    def linear0(x: T.handle,
                w: T.handle,
                b: T.handle,
                z: T.handle):
        m, n, k = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (1, m), "float32")
        W = T.match_buffer(w, (n, m), "float32")
        B = T.match_buffer(b, (n, ), "float32")
        Z = T.match_buffer(z, (1, n), "float32")
        Y = T.alloc_buffer((1, n), "float32")
        for i, j, k in T.grid(1, n, m):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.sblock("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))   # TensorIR
            lv1 = R.call_dps_packed("env.relu", (lv0, ), R.Tensor((1, n), "float32"))       # PyTorch
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, k), "float32")) # PyTorch
            R.output(out)
        return out

target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(MyModuleMixture, target)
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleMixture Prediction:", class_names[pred_kind[0]])

# -----------------------------------------------------------------------------
# BindParams: bakes the model weights into the IRModule
#
# Before: vm["main"](data, w0, b0, w1, b1) — weights passed every call
# After:  vm["main"](data)                 — weights embedded in the module
#
# Useful for inference: weights are fixed after training, no need to pass them.
# -----------------------------------------------------------------------------

MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
# print(MyModuleWithParams.script())
target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(MyModuleWithParams, target)
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd)   # only input data needed — weights are baked in

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleWithParams Prediction:", class_names[pred_kind[0]])
