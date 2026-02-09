import os
import imagesize
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def process_single_image(path):
    """Функция для обработки одного изображения (выполняется в потоке)."""
    try:
        w_raw, h_raw = imagesize.get(path)
        # Округляем до кратного 32
        w_bucket = (w_raw // 32) * 32
        h_bucket = (h_raw // 32) * 32

        if w_bucket > 0 and h_bucket > 0:
            return {"p": path, "r": [w_bucket, h_bucket]}
    except:
        pass
    return None


def create_native_buckets(root_dir, save_path="metadata.json", max_workers=16):
    all_paths = []
    print("Составление списка файлов...")
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                all_paths.append(os.path.join(root, f))

    metadata = []
    buckets_stats = defaultdict(int)

    print(f"Запуск обработки в {max_workers} потоках...")
    # Используем ThreadPoolExecutor для параллельного чтения заголовков
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # tqdm оборачивает map для индикации прогресса
        results = list(tqdm(executor.map(process_single_image, all_paths), total=len(all_paths)))

    # Собираем результаты
    for res in results:
        if res:
            metadata.append(res)
            buckets_stats[tuple(res['r'])] += 1

    print(f"Сохранение метаданных в {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(metadata, f)

    print(f"Индексация завершена. Найдено {len(metadata)} валидных изображений.")
    print(f"Создано {len(buckets_stats)} уникальных бакетов.")

    top_buckets = sorted(buckets_stats.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Топ-5 бакетов: {top_buckets}")


# Запуск: создаём отдельно файлы для train и val
if __name__ == "__main__":
    # Путь к папке с датасетом, где есть подпапки train и val
    # ВАЖНО: укажи правильный путь к своему датасету при необходимости
    create_native_buckets("dataset/train",
                          save_path="metadata_train.json",
                          max_workers=6)
    create_native_buckets("dataset/val",
                          save_path="metadata_val.json",
                          max_workers=6)
