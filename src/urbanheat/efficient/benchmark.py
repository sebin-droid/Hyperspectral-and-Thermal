"""Efficiency metrics on CPU or GPU (Member 2's train.efficiency is GPU-only), ONNX export and INT8 quantisation."""
import copy
import io
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from urbanheat.models import cnn

INPUTS = ["hs", "ls", "env", "thermal", "sensor"]


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure(model, batch, device="cpu", reps=20):
    """Contract metrics for one batch of patches: params, FLOPs per patch, size, latency per patch, peak memory."""
    model = model.to(device).eval()
    b = {k: torch.as_tensor(batch[k]).to(device) for k in INPUTS}
    n = b["ls"].shape[0]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        model(b)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        model(b)
    _sync(device)
    latency = 1000 * (time.perf_counter() - t0) / reps / n
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    # shortcut: no reliable per-model peak memory on CPU (torch allocations bypass tracemalloc); GPU runs give it
    peak = round(torch.cuda.max_memory_allocated() / 1e6, 1) if device == "cuda" else None
    return dict(params=cnn.count_params(model), flops=int(cnn.count_flops(model, b)),
                size_mb=round(buf.getbuffer().nbytes / 1e6, 3), latency_ms=round(latency, 4), peak_mem_mb=peak,
                device=device, bench_batch=n)


class _Positional(nn.Module):
    """ONNX needs tensors, not the dict batch HeatCNN takes."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hs, ls, env, thermal, sensor):
        return self.model(dict(hs=hs, ls=ls, env=env, thermal=thermal, sensor=sensor))


def export_onnx(model, batch, path):
    """FP32 ONNX with dynamic batch and scene size. Inputs a variant does not read are dropped from the graph."""
    spatial = {0: "batch", 2: "height", 3: "width"}
    axes = {**{k: spatial for k in INPUTS[:4]}, "sensor": {0: "batch"}, "lst": {0: "batch", 1: "height", 2: "width"}}
    args = tuple(torch.as_tensor(batch[k]) for k in INPUTS)
    model = copy.deepcopy(model).cpu().eval()
    for m in model.modules():                   # the TorchScript exporter mis-exports in-place ReLU (torch 2.15)
        if isinstance(m, nn.ReLU):
            m.inplace = False
    torch.onnx.export(_Positional(model), args, str(path), input_names=INPUTS, output_names=["lst"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    return Path(path)


def quantize_int8(src, dst):
    """Dynamic INT8 weights (onnxruntime). QUInt8: ConvInteger on CPU has no int8-weight kernel."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return Path(dst)


def onnx_run(path, batch, reps=20):
    """(latency ms per patch, prediction) of an ONNX model with onnxruntime on CPU."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {i.name: np.asarray(batch[i.name]) for i in sess.get_inputs()}
    out = sess.run(None, feed)[0]
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, feed)
    n = np.asarray(batch["ls"]).shape[0]
    return round(1000 * (time.perf_counter() - t0) / reps / n, 4), out
