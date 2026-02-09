import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from Dataset import NativeResolutionDataset, BucketManagerSampler
from Architecture import ColorNet
from torch.utils.data import DataLoader
from ScheduleAndSampleGen import linear_schedule
from ScheduleAndSampleGen import sample_validation
import time
from tqdm import tqdm
import os
import csv

class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * p.detach())

    def apply_ema(self):
        """Загружает веса EMA в модель, сохраняя текущие в backup"""
        self.backup = {name: p.clone().detach() for name, p in self.model.named_parameters() if p.requires_grad}
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore_raw(self):
        """Возвращает сырые веса обратно в модель"""
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def copy_to(self):
        """Копирует EMA веса в основную модель (для валидации)"""
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        for name, value in state_dict.items():
            self.shadow[name].copy_(value)


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

def save_checkpoint(model, optimizer, ema, global_opt_step, smoothed_loss, accumulated_time):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(), # Сохраняем EMA
        "global_opt_step": global_opt_step,
        "smoothed_loss": smoothed_loss,
        "accumulated_time": accumulated_time
    }, "checkpoint/checkpoint.pth")

def load_checkpoint(model, optimizer, ema):
    if os.path.exists("checkpoint/checkpoint.pth"):
        ckpt = torch.load("checkpoint/checkpoint.pth", map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        return ckpt["global_opt_step"], ckpt["smoothed_loss"], ckpt.get("accumulated_time", 0.0)
    return 0, 0.0, 0.0



def train(model, optimizer, train_loader, ema, optimization_steps=1000, accumulation_steps=1, global_opt_step=0, smoothed_loss=0.0, accumulated_time=0.0):
    model.train()
    step = 0
    train_iter = iter(train_loader)
    total_steps = optimization_steps * accumulation_steps

    schedule = linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02)

    pbar = tqdm(
        total=optimization_steps,
        desc="Optimization steps",
        unit="step"
    )

    start_time = time.time()

    # Хранения лоссов батчей перед optimizer.step()
    batch_losses = []
    # Коэффициент экспоненциального сглаживания
    smoothing_factor = 0.998


    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        gray = batch["gray"].cuda()
        rgb = batch["rgb"].cuda()

        # --- Шаг диффузии ---
        # t случайный из [0, 999]
        t = torch.randint(0, 1000, (rgb.size(0),), device="cuda")

        # Шум, который накладываем
        noise = torch.randn_like(rgb)

        # Коэфициенты, чтобы наложить шум на изображение
        sqrt_alpha_bar_t = schedule["sqrt_alpha_bar"][t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = schedule["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1)

        noised_rgb = sqrt_alpha_bar_t * rgb + sqrt_one_minus_alpha_bar_t * noise

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            predicted_noise = model(gray, noised_rgb, t)
            loss = F.mse_loss(predicted_noise, noise) / accumulation_steps

        batch_losses.append(loss.item() * accumulation_steps)

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            # Сбор статистики градиентов перед optimizer.step()
            grad_stats = get_grad_stats(model)

            current_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            if global_opt_step == 0:
                smoothed_loss = current_loss

            smoothed_loss = smoothing_factor * smoothed_loss + (1 - smoothing_factor) * current_loss
            batch_losses = []

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update()
            global_opt_step += 1


            # Запись в CSV
            # логируем только каждые 500 шагов
            if global_opt_step % 500 == 0:
                now = time.time()
                elapsed = now - start_time  # время, прошедшее с последнего сброса start_time
                accumulated_time += elapsed  # добавляем к накопленному времени из чекпоинта
                start_time = now  # сбрасываем отсчёт на следующий интервал


                # формат hh:mm:ss
                def fmt(t):
                    t = int(t)
                    return f"{t // 3600}h:{(t % 3600) // 60}m:{t % 60}s"

                with open("loss_log.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        global_opt_step,
                        smoothed_loss,
                        grad_stats.get("mean_abs", ""),
                        grad_stats.get("max_abs", ""),
                        grad_stats.get("min", ""),
                        grad_stats.get("l2_norm", ""),
                        fmt(accumulated_time)
                    ])



            # Обновление прогресс-бара
            pbar.set_postfix({
                "smoothed_loss": f"{smoothed_loss:.4e}",
                "current_loss": f"{current_loss:.4e}",
                "grad_mean": f"{grad_stats.get('mean_abs', 0):.3e}",
                "grad_l2": f"{grad_stats.get('l2_norm', 0):.3e}",
                "grad_max": f"{grad_stats.get('max_abs', 0):.3e}",
                "step": global_opt_step
            })

            pbar.update(1)

        step += 1



    pbar.close()

    return global_opt_step, smoothed_loss, accumulated_time


def main():
    # Константы
    lr = 1e-4
    optimization_steps = 10000
    batch_size = 4
    accumulation_steps = 32


    # Создание папки чекпоинтов и файла логов
    os.makedirs("checkpoint", exist_ok=True)

    if not os.path.exists("loss_log.csv"):
        with open("loss_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_opt_step",
                "smoothed_loss",
                "grad_mean_abs",
                "grad_max_abs",
                "grad_min",
                "grad_l2_norm",
                "accumulated_time"
            ])

    # Загрузчики изображений
    train_dataset = NativeResolutionDataset("metadata_train.json", train=True)
    val_dataset = NativeResolutionDataset("metadata_val.json", train=False)

    train_sampler = BucketManagerSampler(train_dataset.buckets, batch_size=batch_size, shuffle=True)
    val_sampler   = BucketManagerSampler(val_dataset.buckets, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    # Модель с оптимизатором
    model = ColorNet().cuda()
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.0001, betas=(0.9, 0.99))

    ema = EMAModel(model, decay=0.999)

    # Загрузка чекпоинта
    global_opt_step, smoothed_loss, accumulated_time = load_checkpoint(model, optimizer, ema)

    global_opt_step, smoothed_loss, accumulated_time = train(
        model=model,
        optimizer=optimizer,
        ema=ema,
        train_loader=train_loader,
        optimization_steps=optimization_steps,
        accumulation_steps=accumulation_steps,
        global_opt_step=global_opt_step,
        smoothed_loss=smoothed_loss,
        accumulated_time=accumulated_time
    )

    save_checkpoint(model, optimizer, ema, global_opt_step, smoothed_loss, accumulated_time)

    ema.apply_ema()
    sample_validation(model=model, val_loader=val_loader, step=global_opt_step)
    ema.restore_raw() # Бессмысленно, но для сохранения памяти о наличии метода оставлю, влияние слабое





if __name__ == "__main__":
    main()