"""Training and evaluation of the CNN heat models (masked loss, flips/rotations, early stopping on val)."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler

from urbanheat.models import baselines, cnn

KEYS = ["hs", "ls", "env", "thermal", "met", "y", "mask", "sensor"]


def to_device(batch, device):
    out = {}
    for k in KEYS:
        v = batch[k]
        v = torch.as_tensor(v) if not torch.is_tensor(v) else v
        out[k] = v.to(device, non_blocking=True)
    return out


def augment(b):
    """Same random flip / 90-degree rotation for every spatial tensor of the batch."""
    k, flip = np.random.randint(4), np.random.rand() < 0.5
    for key in ["hs", "ls", "env", "thermal", "y", "mask"]:
        x = torch.rot90(b[key], k, dims=(-2, -1))
        b[key] = torch.flip(x, dims=(-1,)) if flip else x
    return b


def masked_mse(pred, y, mask):
    m = mask.float()
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1)


def fit(model, train_ds, val_ds, epochs=30, batch_size=32, lr=2e-3, samples_per_epoch=4000, patience=6,
        device="cuda", log=print):
    """Train with AdamW + cosine schedule and mixed precision; keep the weights with the best val RMSE."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (samples_per_epoch // batch_size + 1))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, drop_last=True,
                        sampler=RandomSampler(train_ds, replacement=True, num_samples=samples_per_epoch))
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=0)
    best, best_state, history, wait = np.inf, None, [], 0
    for ep in range(1, epochs + 1):
        t0, losses = time.time(), []
        model.train()
        for b in loader:
            b = augment(to_device(b, device))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = masked_mse(model(b).float(), b["y"], b["mask"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        val_rmse = evaluate_patches(model, val_loader, device)
        history.append(dict(epoch=ep, train_rmse=float(np.sqrt(np.mean(losses))), val_rmse=val_rmse,
                            seconds=round(time.time() - t0, 1)))
        log(f"epoch {ep:2d}  train RMSE {history[-1]['train_rmse']:.3f}  val RMSE {val_rmse:.3f}  "
            f"({history[-1]['seconds']:.0f}s)")
        if val_rmse < best - 1e-4:
            best, best_state, wait = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                log(f"early stop; best val RMSE {best:.3f}")
                break
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_patches(model, loader, device):
    model.eval()
    se, n = 0.0, 0
    for b in loader:
        b = to_device(b, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            p = model(b).float()
        m = b["mask"]
        se += float(((p - b["y"]) ** 2)[m].sum())
        n += int(m.sum())
    return float(np.sqrt(se / max(n, 1)))


@torch.no_grad()
def predict_scene(model, ds, pid, device="cuda"):
    """Whole-scene prediction (B=1, the model is fully convolutional). Returns (pred, y, mask) as numpy."""
    model.eval()
    s = ds.scene(pid)
    b = {k: torch.as_tensor(np.asarray(s[k]))[None].to(device) for k in KEYS}
    return model(b)[0].float().cpu().numpy(), s["y"], s["mask"]


def evaluate_scenes(model, ds, info_of, device="cuda"):
    """Pixel metrics over every valid pixel of every scene in `ds` (+ per climate / sensor), and predictions."""
    ys, ps, groups, preds = [], [], {"climate_group": [], "sensor": []}, {}
    for pid in ds.pairs:
        p, y, m = predict_scene(model, ds, pid, device)
        preds[pid] = (p, y, m)
        ys.append(y[m]); ps.append(p[m])
        for k in groups:
            groups[k].append(np.full(m.sum(), info_of[pid][k]))
    info = pd.DataFrame({k: np.concatenate(v) for k, v in groups.items()})
    return baselines.evaluate(np.concatenate(ys), np.concatenate(ps), info), preds


def efficiency(model, ds, device="cuda", batch_size=32, reps=20):
    """Contract metrics: params, FLOPs per 32x32 patch, size, latency per patch, peak GPU memory."""
    b = to_device(next(iter(DataLoader(ds, batch_size=batch_size))), device)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            model(b)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            model(b)
        torch.cuda.synchronize()
    return dict(params=cnn.count_params(model), flops=int(cnn.count_flops(model, b)),
                size_mb=round(cnn.count_params(model) * 4 / 1e6, 2),
                latency_ms=round(1000 * (time.time() - t0) / reps / batch_size, 4),
                peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1))


def save(model, run_dir, metrics, history, config):
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (d / "history.json").write_text(json.dumps(history, indent=1))
    (d / "config.json").write_text(json.dumps(config, indent=1))
