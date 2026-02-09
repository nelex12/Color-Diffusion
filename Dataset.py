import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from collections import defaultdict
import random
import json


# --- 1. ОБНОВЛЕННЫЙ КЛАСС ДАТАСЕТА ---
class NativeResolutionDataset(Dataset):
    def __init__(self, json_path, train=True):
        """
        json_path: путь к созданному файлу metadata.json
        """
        print(f"Загрузка метаданных из {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.buckets = defaultdict(list)
        self.train = train

        # Группируем индексы по бакетам сразу при загрузке
        # В json у нас структура: [{"p": path, "r": [w, h]}, ...]
        for idx, item in enumerate(self.data):
            res = tuple(item['r'])
            self.buckets[res].append(idx)

        print(f"Датасет готов. Найдено {len(self.data)} изображений в {len(self.buckets)} бакетах.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = item['p']
        target_res = tuple(item['r'])  # (W, H)

        # 1. Загрузка (настоящая)
        img = Image.open(path).convert('RGB')

        # 2. Умный ресайз (сохраняем пропорции, кропаем минимум)
        tw, th = target_res
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)

        img = img.resize((nw, nh), Image.BICUBIC)

        # 3. Центрированный кроп
        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        if self.train and random.random()> 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 4. Перевод в тензоры
        # RGB в диапазоне [-1, 1]
        rgb_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        # Gray в диапазоне [-1, 1]
        gray_img = img.convert('L')
        gray_tensor = torch.from_numpy(np.array(gray_img)).unsqueeze(0).float() / 127.5 - 1.0

        return {"gray": gray_tensor, "rgb": rgb_tensor}


# --- 2. КЛАСС СЭМПЛЕРА (остается без изменений) ---
class BucketManagerSampler(Sampler):
    def __init__(self, buckets, batch_size, shuffle=True):
        self.buckets = buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = []
        self._create_batches()

    def _create_batches(self):
        self.batches = []
        # Копируем ключи, чтобы можно было перемешать бакеты
        keys = list(self.buckets.keys())

        for res in keys:
            indices = list(self.buckets[res])
            if self.shuffle:
                np.random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i: i + self.batch_size]
                # Условие на пропуск неполных батчей убрано по твоему запросу
                if len(batch) > 0:
                    self.batches.append(batch)

    def __iter__(self):
        self._create_batches()
        if self.shuffle:
            np.random.shuffle(self.batches)
        yield from self.batches

    def __len__(self):
        return len(self.batches)


# --- 3. ПРИМЕР ИСПОЛЬЗОВАНИЯ (train / val раздельно) ---
if __name__ == "__main__":
    # Укажи пути к созданным ранее файлам
    JSON_TRAIN = "metadata_train.json"
    JSON_VAL = "metadata_val.json"
    BATCH_SIZE = 4

    # Инициализация датасетов
    train_dataset = NativeResolutionDataset(JSON_TRAIN)
    val_dataset = NativeResolutionDataset(JSON_VAL)

    # Самплеры: для train shuffle=True, для val shuffle=False
    train_sampler = BucketManagerSampler(train_dataset.buckets, batch_size=BATCH_SIZE, shuffle=True)
    val_sampler = BucketManagerSampler(val_dataset.buckets, batch_size=BATCH_SIZE, shuffle=False)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Проверка нескольких батчей из train и val
    print("\n--- Первые 10 батчей (train) ---")
    for i, batch in enumerate(train_loader):
        if i >= 10: break
        gray = batch['gray']
        rgb = batch['rgb']
        print(f"Train Batch {i:02d} | Gray: {gray.shape} | RGB: {rgb.shape}")

    print("\n--- Первые 5 батчей (val) ---")
    for i, batch in enumerate(val_loader):
        if i >= 5: break
        gray = batch['gray']
        rgb = batch['rgb']
        print(f"Val Batch {i:02d} | Gray: {gray.shape} | RGB: {rgb.shape}")
