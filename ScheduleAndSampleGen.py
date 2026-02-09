import os
import torch
import torchvision
from tqdm import tqdm
from PIL import Image


def linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    beta = torch.linspace(beta_start, beta_end, T, device='cuda')
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "one_over_sqrt_alpha": 1.0 / torch.sqrt(alpha),
        "coeff": beta / torch.sqrt(1.0 - alpha_bar)
    }


@torch.no_grad()
def sample_validation(model, val_loader, step):
    model.eval()
    save_dir = os.path.join("samples", str(step))
    os.makedirs(save_dir, exist_ok=True)

    schedule = linear_schedule()
    betas = schedule["beta"]
    one_over_sqrt_alpha = schedule["one_over_sqrt_alpha"]
    sqrt_one_minus_alpha_bar = schedule["sqrt_one_minus_alpha_bar"]
    T = betas.shape[0]

    sample_counter = 0
    for batch in val_loader:
        gray = batch["gray"].cuda()
        real_rgb = batch["rgb"].cuda()
        x = torch.randn_like(real_rgb)

        for t in reversed(range(T)):
            t_tensor = torch.full((gray.size(0),), t, dtype=torch.long).cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                noise_pred = model(gray, x, t_tensor)

            # ПОВЫШЕНИЕ ТОЧНОСТИ: переводим предсказание в fp32 перед математикой
            noise_pred = noise_pred.float()

            beta_t = betas[t]
            coeff = beta_t / sqrt_one_minus_alpha_bar[t]
            mean = one_over_sqrt_alpha[t] * (x - coeff * noise_pred)
            # Все расчеты ниже пойдут в fp32
            x = mean + torch.sqrt(beta_t) * torch.randn_like(x) if t > 0 else mean

            # Ограничение диапазона тензора для стабильности генерации
            x = x.clamp(-1, 1)

        def denorm(img):
            return (img.clamp(-1, 1) + 1) / 2

        gray3 = denorm(gray.repeat(1, 3, 1, 1))
        fake = denorm(x)
        real = denorm(real_rgb)

        for i in range(gray.size(0)):
            grid = torch.cat((gray3[i], fake[i], real[i]), dim=2)
            torchvision.utils.save_image(grid, os.path.join(save_dir, f"sample_{sample_counter}.png"))
            sample_counter += 1

    model.train()

if __name__ == "__main__":
    schedule = linear_schedule(T=1000)

    # Демонстрация: выводим каждый 100-й шаг
    print("Step | beta     | alpha_bar | sqrt_alpha_bar | sqrt_1-alpha_bar")
    print("-"*60)
    for t in range(0, 1000, 100):
        print(f"{t:4d} | {schedule['beta'][t]:.6f} | {schedule['alpha_bar'][t]:.6f} | "
              f"{schedule['sqrt_alpha_bar'][t]:.6f} | {schedule['sqrt_one_minus_alpha_bar'][t]:.6f}")
    # показываем последний шаг отдельно
    t = 999
    print(f"{t:4d} | {schedule['beta'][t]:.6f} | {schedule['alpha_bar'][t]:.6f} | "
          f"{schedule['sqrt_alpha_bar'][t]:.6f} | {schedule['sqrt_one_minus_alpha_bar'][t]:.6f}")