import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torchinfo import summary
import math
import random



class AdaLN(nn.Module):
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 6 * embed_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        with torch.no_grad():
            self.linear.bias.fill_(0)
            # Приоткрываем гейты (индексы 2 и 5 в chunk), чтобы градиент шел сразу
            self.linear.bias[2*embed_dim : 3*embed_dim] = 0.5
            self.linear.bias[5*embed_dim : 6*embed_dim] = 0.5

    def forward(self, emb):
        return self.linear(emb).to(emb.dtype).chunk(6, dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_value = dropout
        self.dropout = nn.Dropout(dropout)
        self.adaln = AdaLN(embed_dim, cond_dim=128)  # cond_dim должен совпадать с выходом time_projection

        # Линейные проекции для Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Линейная проекция выхода
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # FFN с SwiGLU и SiLU
        self.ff_linear = nn.Sequential(
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.ff1 = nn.Linear(embed_dim, ff_hidden_dim * 2)  # первый слой для SwiGLU

        # RMSNorm вместо LayerNorm
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)

    def forward(self, x, t_emb, attn_mask=None, is_causal=False):
        # Извлекаем 6 параметров модуляции из временного эмбеддинга
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(t_emb)

        # --- Блок внимания (Attention) ---
        res = x
        x_norm = F.rms_norm(x, self.norm1.normalized_shape, self.norm1.weight.to(x.dtype), self.norm1.eps)

        # Адаптивная нормализация (вместо обычного PreNorm)
        # unsqueeze(1) нужен для корректного умножения на последовательность [B, L, C]
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        # Ваш стандартный расчет Q, K, V
        batch_size, seq_len, embed_dim = x.size()
        head_dim = embed_dim // self.num_heads
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        def reshape_heads(tensor):
            return tensor.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        q, k, v = map(reshape_heads, (q, k, v))
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                     dropout_p=self.dropout_value, is_causal=is_causal)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Применяем gate перед сложением с residual (характерно для AdaLN-Zero)
        x = res + gate_msa.unsqueeze(1) * self.out_proj(attn_output)

        # --- Блок FeedForward (FFN) ---
        res = x
        x_norm = F.rms_norm(x, self.norm2.normalized_shape, self.norm2.weight.to(x.dtype), self.norm2.eps)
        # Адаптивная нормализация для второго слоя
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        # Ваш SwiGLU
        x1, x2 = self.ff1(x_norm).chunk(2, dim=-1)
        x_ff = F.silu(x1) * x2
        x_ff = self.ff_linear(x_ff)

        # Gate для FFN
        x = res + gate_mlp.unsqueeze(1) * self.dropout(x_ff)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_value = dropout
        self.dropout = nn.Dropout(dropout)

        # Модуляция на основе временного шага (t_emb)
        self.adaln = AdaLN(embed_dim, cond_dim=128)

        # Проекции: Q берем из x (RGB), K и V из context (Gray)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # SwiGLU FFN
        self.ff1 = nn.Linear(embed_dim, ff_hidden_dim * 2)
        self.ff_linear = nn.Sequential(
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

        # Нормализация для обоих потоков данных
        self.norm_x = nn.RMSNorm(embed_dim)  # Для Query (цвет)
        self.norm_context = nn.RMSNorm(embed_dim)  # Для Key/Value (серое)
        self.norm2 = nn.RMSNorm(embed_dim)  # Для FFN

    def forward(self, x, context, t_emb):
        # Извлекаем параметры модуляции (как в DiT / Stable Diffusion)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(t_emb)

        # --- Кросс-Внимание ---
        res = x

        # Нормализуем и модулируем Query (цвет)
        # Для Query (цвет)
        x_norm = F.rms_norm(x, self.norm_x.normalized_shape, self.norm_x.weight.to(x.dtype), self.norm_x.eps)

        # Для Context (серое) — тоже через F для стабильности dtypes
        c_norm = F.rms_norm(context, self.norm_context.normalized_shape, self.norm_context.weight.to(x.dtype),
                            self.norm_context.eps)

        # Для FFN в конце блока
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)


        batch_size, seq_len, _ = x.shape
        ctx_len = context.shape[1]
        head_dim = self.embed_dim // self.num_heads

        # Формируем Q, K, V
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(c_norm).view(batch_size, ctx_len, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(c_norm).view(batch_size, ctx_len, self.num_heads, head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # Здесь пиксели X ищут соответствия в пикселях Context
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_value if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Residual connection с гейтом (AdaLN-Zero)
        x = res + gate_msa.unsqueeze(1) * self.out_proj(attn_output)

        # --- FeedForward (стандартный для ветки x) ---
        res = x
        x_norm = F.rms_norm(x, self.norm2.normalized_shape, self.norm2.weight.to(x.dtype), self.norm2.eps)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        x1, x2 = self.ff1(x_norm).chunk(2, dim=-1)
        x_ff = F.silu(x1) * x2
        x_ff = self.ff_linear(x_ff)

        x = res + gate_mlp.unsqueeze(1) * self.dropout(x_ff)

        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()

        self.time_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
        )

        self.grayscale_downsample1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

        self.grayscale_downsample2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

        self.grayscale_downsample3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

        self.grayscale_downsample4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )




        self.rgb_downsample1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

        self.rgb_downsample2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

        self.rgb_downsample3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

        self.rgb_downsample4 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

        self.GrayWindowAttentionTransformer = TransformerBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256*4, dropout=0.0)
        self.RGBWindowAttentionTransformer = TransformerBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256*4, dropout=0.0)

        self.GrayFirstGlobalAttentionTransformer = TransformerBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256 * 4, dropout=0.05)
        self.RGBFirstGlobalAttentionTransformer1 = TransformerBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256 * 4, dropout=0.05)

        self.cross1 = CrossAttentionBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256*8)
        self.cross2 = CrossAttentionBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256*8)
        self.cross3 = CrossAttentionBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256*8)
        self.cross4 = CrossAttentionBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256*8)

        self.RGBFirstGlobalAttentionTransformer2 = TransformerBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256 * 4, dropout=0.05)
        self.RGBFirstGlobalAttentionTransformer3 = TransformerBlock(embed_dim=256, num_heads=4, ff_hidden_dim=256 * 4, dropout=0.05)

        self.up4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),  # rgb_downsample4 + skip4 = 256+256
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(256 + 64, 128, kernel_size=3, padding=1),  # x (256) + rgb_skip3 (64)
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # x (128) + rgb_skip2 (64)
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),  # x (64) + rgb_skip1 (32)
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)



    def forward(self, gray_img, noise_rgb_img, t):
        t_emb = self.sinusoidal_embedding(t)
        t_emb = self.time_projection(t_emb)

        noise_rgb_img = torch.cat((noise_rgb_img, gray_img), dim=1)

        gray_img = self.grayscale_downsample1(gray_img)
        gray_img = self.grayscale_downsample2(gray_img)
        gray_img = self.grayscale_downsample3(gray_img)
        gray_img = self.grayscale_downsample4(gray_img)

        noise_rgb_img = self.rgb_downsample1(noise_rgb_img)
        rgb_skip1 = noise_rgb_img
        noise_rgb_img = self.rgb_downsample2(noise_rgb_img)
        rgb_skip2 = noise_rgb_img
        noise_rgb_img = self.rgb_downsample3(noise_rgb_img)
        rgb_skip3 = noise_rgb_img
        noise_rgb_img = self.rgb_downsample4(noise_rgb_img)
        rgb_skip4 = noise_rgb_img

        gray_img = self.apply_window_attention(gray_img, self.GrayWindowAttentionTransformer, t_emb=t_emb)
        noise_rgb_img = self.apply_window_attention(noise_rgb_img, self.RGBWindowAttentionTransformer, t_emb=t_emb)

        B, C, H, W = gray_img.shape

        gray_img = gray_img.flatten(2).transpose(1, 2)  # [B, L, C]
        noise_rgb_img = noise_rgb_img.flatten(2).transpose(1, 2)  # [B, L, C]

        pos_embed = self.get_2d_sincos_pos_embed(H, W, embed_dim=256, device=gray_img.device)

        gray_img = gray_img + pos_embed
        noise_rgb_img = noise_rgb_img + pos_embed

        gray_img = self.GrayFirstGlobalAttentionTransformer(gray_img, t_emb=t_emb)
        noise_rgb_img = self.RGBFirstGlobalAttentionTransformer1(noise_rgb_img, t_emb=t_emb)

        noise_rgb_img = self.cross1(noise_rgb_img, gray_img, t_emb=t_emb)
        noise_rgb_img = self.cross2(noise_rgb_img, gray_img, t_emb=t_emb)
        noise_rgb_img = self.cross3(noise_rgb_img, gray_img, t_emb=t_emb)
        noise_rgb_img = self.cross4(noise_rgb_img, gray_img, t_emb=t_emb)

        noise_rgb_img = self.RGBFirstGlobalAttentionTransformer2(noise_rgb_img, t_emb=t_emb)
        noise_rgb_img = self.RGBFirstGlobalAttentionTransformer3(noise_rgb_img, t_emb=t_emb)

        # noise_rgb_img: [B, L, C] → [B, C, H, W]
        noise_rgb_img = noise_rgb_img.transpose(1, 2).contiguous()
        noise_rgb_img = noise_rgb_img.view(B, C, H, W)  # [B, 256, H, W]

        # Конкатенация с skip-связями на одинаковых разрешениях
        x = torch.cat([noise_rgb_img, rgb_skip4], dim=1)  # [B, 256+256, H, W] → same, up4 не трогаем
        x = self.up4(x)

        x = torch.cat([x, rgb_skip3], dim=1)  # [B, 256+64, H*2, W*2]
        x = self.up3(x)

        x = torch.cat([x, rgb_skip2], dim=1)  # [B, 128+64, H*4, W*4]
        x = self.up2(x)

        x = torch.cat([x, rgb_skip1], dim=1)  # [B, 64+32, H*8, W*8]
        x = self.up1(x)

        # Финальная свертка
        out_rgb = self.final_conv(x)  # [B, 3, H*8, W*8] — соответствует исходному разрешению входа
        return out_rgb

    def sinusoidal_embedding(self, t, dim=128):
        assert dim % 2 == 0, "Размерность должна быть четной"

        device = t.device if isinstance(t, torch.Tensor) else "cpu"

        half_dim = dim // 2
        # логарифмическое распределение частот
        emb = torch.exp(torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1))
        # умножаем t на частоты
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        # синусы и косинусы
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]
        return emb

    def apply_window_attention(self, x, transformer, t_emb, window_size=2):
        B, C, H, W = x.shape

        # Разбивка на окна
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

        # Схлопывание батча и окон для входа в трансформер
        x = x.view(-1, window_size * window_size, C)

        num_windows = x.shape[0] // B
        t_emb = t_emb.repeat_interleave(num_windows, dim=0)

        x = transformer(x, t_emb)

        # Расклейка батча и окон обратно
        x = x.view(B, H // window_size, W // window_size, window_size, window_size, C)

        # Сборка в исходную картинку
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)

        return x

    def get_2d_sincos_pos_embed(self, h, w, embed_dim, device):
        """
        Генерирует 2D синусоидальные эмбеддинги на лету.
        Output: [1, h*w, embed_dim]
        """
        assert embed_dim % 2 == 0, "Embed dim must be divisible by 2"

        # Сетка координат
        grid_h = torch.arange(h, device=device, dtype=torch.float32)
        grid_w = torch.arange(w, device=device, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

        # Flatten grids: [H*W]
        grid_h = grid_h.flatten()
        grid_w = grid_w.flatten()

        # Половина каналов для H, половина для W
        dim_h = embed_dim // 2
        dim_w = embed_dim - dim_h

        def get_1d_sincos(pos, dim):
            # Стандартная формула: pos / 10000^(2i/dim)
            omega = torch.arange(dim // 2, device=device, dtype=torch.float32)
            omega /= (dim / 2)
            omega = 1.0 / (10000 ** omega)  # [dim/2]

            out = torch.outer(pos, omega)  # [L, dim/2]
            return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [L, dim]

        emb_h = get_1d_sincos(grid_h, dim_h)
        emb_w = get_1d_sincos(grid_w, dim_w)

        # Конкатенация и добавление размерности батча для бродкастинга
        pos_embed = torch.cat([emb_h, emb_w], dim=1)  # [H*W, C]
        return pos_embed.unsqueeze(0)  # [1, H*W, C]

def main():
    model = ColorNet().cuda()
    model.eval()

    batch_size = 8

    gray_input = torch.randn(batch_size, 1, 512, 512).cuda()
    rgb_input = torch.randn(batch_size, 3, 512, 512).cuda()
    t = torch.randint(0, 1000, (batch_size,)).cuda()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        summary(
            model,
            input_data=(gray_input, rgb_input, t),
            depth=1,
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
        )




if __name__ == "__main__":
    main()