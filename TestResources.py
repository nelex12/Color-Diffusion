# TestResources.py
import torch
import time
import numpy as np
from bitsandbytes.optim import AdamW8bit
from Architecture import ColorNet
import torch.nn.functional as F


def get_grad_stats(model):
    """
    Собирает общую статистику градиентов всей модели.
    Возвращает словарь с mean_abs, max_abs, min, l2_norm.
    """
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    if not grads:
        return {}

    all_grads = torch.cat(grads)

    stats = {
        "mean_abs": all_grads.abs().mean().item(),
        "max_abs": all_grads.abs().max().item(),
        "min": all_grads.min().item(),
        "l2_norm": all_grads.norm().item()
    }

    return stats

def test_memory(batch_size=4, device='cuda'):
    # Модель
    model = ColorNet().to(device)
    # model.convert_model_to_bf16()
    model.train()

    # Оптимизатор 8-bit AdamW
    optimizer = AdamW8bit(model.parameters(), lr=1e-3)

    # Генерация тестовых входов
    gray_input = torch.randn(batch_size, 1, 1600, 1600, device=device)
    rgb_input = torch.randn(batch_size, 3, 1600, 1600, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    # Сброс пиковой памяти
    torch.cuda.reset_peak_memory_stats(device)

    # Замер времени
    start_time = time.time()

    # Forward + backward с bf16
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        predicted_noise = model(gray_input, rgb_input, t)
        loss = F.mse_loss(predicted_noise, rgb_input)

    loss.backward()

    stats = get_grad_stats(model)


    optimizer.step()
    optimizer.zero_grad()

    elapsed = time.time() - start_time
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3

    return elapsed, peak_mem_gb, stats


if __name__ == "__main__":
    runs = 16
    times = []
    mems = []

    for i in range(runs):
        t, m, s = test_memory()
        times.append(t)
        mems.append(m)
        print(f"Run {i + 1}: Time = {t:.2f}s, Peak Memory = {m:.2f} GB, "
              f"Grad Stats: mean_abs={s.get('mean_abs', 0):.3e}, "
              f"max_abs={s.get('max_abs', 0):.3e}, "
              f"min={s.get('min', 0):.3e}, "
              f"l2_norm={s.get('l2_norm', 0):.3e}")
        print("\n")

    print("\nStatistics over all runs:")
    print(f"Average Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
    print(f"Average Peak Memory: {np.mean(mems):.2f} GB ± {np.std(mems):.2f} GB")
