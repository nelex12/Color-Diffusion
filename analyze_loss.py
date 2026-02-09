#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_loss_steps_ide.py

Анализ smoothed_loss по global_opt_step, готовый для запуска в IDE.
Файл CSV задаётся через переменную CSV_FILE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# ======================
# CONFIG
# ======================
CSV_FILE = r"loss_log.csv"  # <- укажи свой путь
LOSS_COL = "smoothed_loss"
STEP_COL = "global_opt_step"

TRIM_FIRST_N = 10
EMA_SPAN = 50
SLOPE_WINDOW = 20
PLOT = True
PLOT_FILE = "loss_analysis.png"
# ======================


def ema_skip_nan(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.full_like(values, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(values))[0]
    if finite_idx.size == 0:
        return out
    i0 = finite_idx[0]
    out[i0] = float(values[i0])
    for i in range(i0 + 1, len(values)):
        v = values[i]
        out[i] = out[i - 1] if not np.isfinite(v) else alpha * float(v) + (1 - alpha) * out[i - 1]
    return out


def rolling_linear_slope(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    slopes = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        xs = x[i - window + 1:i + 1].astype(float)
        ys = y[i - window + 1:i + 1].astype(float)
        A = np.vstack([xs, np.ones_like(xs)]).T
        m, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        slopes[i] = float(m)
    return slopes


def fit_exponential(t: np.ndarray, y: np.ndarray) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
    try:
        from scipy.optimize import curve_fit
        c0 = float(np.min(y))
        A0 = float(np.max(y) - c0)
        dt = float(np.max(t) - np.min(t) + 1e-12)
        k0 = 1.0 / dt if dt > 0 else 1.0

        def model(tt, A, k, c): return A * np.exp(-k * tt) + c

        popt, _ = curve_fit(model, t, y, p0=(A0, k0, c0),
                            maxfev=10000, bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
        yhat = model(t, *popt)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return (float(popt[0]), float(popt[1]), float(popt[2])), rmse
    except Exception:
        return None, None


def analyze(filename: str):
    df = pd.read_csv(filename)
    if STEP_COL not in df.columns or LOSS_COL not in df.columns:
        raise ValueError(f"CSV должен содержать колонки '{STEP_COL}' и '{LOSS_COL}'")

    steps = df[STEP_COL].to_numpy(dtype=float)
    loss = df[LOSS_COL].to_numpy(dtype=float)

    # сортировка на всякий случай
    order = np.argsort(steps)
    steps = steps[order]
    loss = loss[order]

    # TRIM
    trim = max(0, TRIM_FIRST_N)
    steps = steps[trim:]
    loss = loss[trim:]
    n = len(loss)

    # Instant slope
    dsteps = np.diff(steps)
    dloss = np.diff(loss)
    inst_slope = np.diff(loss) / (np.diff(steps) + 1e-12)
    inst_slope_padded = np.concatenate(([np.nan], inst_slope))

    # EMA slope
    ema_slope = ema_skip_nan(inst_slope_padded, EMA_SPAN)

    # Rolling linear slope
    roll_slope = rolling_linear_slope(steps, loss, max(3, SLOPE_WINDOW))

    # Acceleration
    acc = np.full_like(roll_slope, np.nan)
    valid_idx = np.where(~np.isnan(roll_slope))[0]
    if valid_idx.size >= 2:
        idxs = valid_idx
        ds = np.diff(roll_slope[idxs])
        dst = np.diff(steps[idxs])
        acc[idxs[1:]] = ds / (dst + 1e-12)

    # Exponential fit
    t_rel = steps - steps[0]
    exp_params, exp_rmse = fit_exponential(t_rel, loss)

    # Последние значения
    last_loss = float(loss[-1])
    last_inst_slope = float(inst_slope_padded[-1]) if np.isfinite(inst_slope_padded[-1]) else None
    last_ema_slope = float(ema_slope[-1]) if np.isfinite(ema_slope[-1]) else None
    last_roll_slope = float(roll_slope[-1]) if np.isfinite(roll_slope[-1]) else None
    last_acc = float(acc[-1]) if np.isfinite(acc[-1]) else None

    # Вывод
    print(f"file: {filename}")
    print(f"samples_after_trim: {n}")
    print(f"last_step: {int(steps[-1])}")
    print(f"last_loss: {last_loss:.12f}")
    print(f"last_inst_slope_per_step: {last_inst_slope:.12e}")
    print(f"last_EMA_slope_per_step: {last_ema_slope:.12e}  # EMA span={EMA_SPAN}")
    print(f"last_local_rolling_slope: {last_roll_slope:.12e}  # window={SLOPE_WINDOW}")
    print(f"last_acceleration_of_slope: {last_acc}")
    if exp_params is not None:
        A, k, c = exp_params
        print(f"exp_fit_A: {A:.12e}")
        print(f"exp_fit_k: {k:.12e}  # characteristic_steps={1.0/k:.6f}" if k > 0 else f"exp_fit_k: {k:.12e}")
        print(f"exp_fit_c: {c:.12e}")
        print(f"exp_fit_rmse: {exp_rmse:.12e}")
    else:
        print("exp_fit: unavailable")

    # Плоты
    if PLOT:
        fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axs[0].plot(steps, loss, marker=".", linewidth=0.7, label="loss")
        axs[0].set_ylabel("loss")
        axs[0].legend()

        axs[1].plot(steps, inst_slope_padded, marker=".", linewidth=0.6, label="instant slope")
        axs[1].plot(steps, ema_slope, linewidth=1.0, label=f"EMA slope span={EMA_SPAN}")
        axs[1].axhline(0.0, linestyle="--", linewidth=0.7)
        axs[1].set_ylabel("dloss/dstep")
        axs[1].legend()

        axs[2].plot(steps, roll_slope, marker=".", linewidth=0.7, label=f"rolling slope window={SLOPE_WINDOW}")
        axs[2].plot(steps, acc, linestyle="--", linewidth=0.8, label="acceleration of slope")
        axs[2].set_ylabel("rolling slope")
        axs[2].set_xlabel("global_opt_step")
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=150)
        print(f"plot_saved: {PLOT_FILE}")
        plt.show()


# ======================
# Запуск для IDE
# ======================
if __name__ == "__main__":
    analyze(CSV_FILE)
