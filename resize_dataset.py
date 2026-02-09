# resize_dataset_simple.py
# Требует: pip install pillow tqdm
import os, time, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

# ========== КОНСТАНТЫ ==========
DATASET = Path("dataset")
SUBFOLDERS = ("train", "val")
MAX_SIDE = 1600
JPEG_QUALITY = 95
WORKERS = max(1, (os.cpu_count() or 2))
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
# ===============================

def human(n):
    for u in ("B","KB","MB","GB"):
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

def all_images(root):
    out = []
    for s in SUBFOLDERS:
        p = root / s
        if p.exists():
            out += [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in EXTS]
    return out

def process(p: Path):
    res = {"path": str(p), "status": None, "orig": 0, "new": 0, "err": None}
    try:
        res["orig"] = p.stat().st_size
        with Image.open(p) as im:
            if getattr(im, "is_animated", False):
                res["status"] = "skipped_animated"; return res
            im = ImageOps.exif_transpose(im)
            w,h = im.size
            if max(w,h) <= MAX_SIDE:
                res["status"] = "skipped_small"; return res
            if w >= h:
                nw, nh = MAX_SIDE, round(h * MAX_SIDE / w)
            else:
                nh, nw = MAX_SIDE, round(w * MAX_SIDE / h)
            im = im.resize((nw, nh), Image.LANCZOS)

            fd, tmp = tempfile.mkstemp(prefix=p.name+".", suffix=".tmp", dir=p.parent)
            os.close(fd)
            fmt = ("JPEG" if p.suffix.lower() in (".jpg",".jpeg") else
                   "PNG" if p.suffix.lower()==".png" else None)
            save_kwargs = {}
            if fmt == "JPEG":
                save_kwargs.update(format="JPEG", quality=JPEG_QUALITY, optimize=True)
                exif = im.info.get("exif")
                if exif: save_kwargs["exif"] = exif
            try:
                if fmt:
                    im.save(tmp, **save_kwargs)
                else:
                    im.save(tmp)  # let PIL infer
            except Exception:
                try:
                    im.save(tmp, format="PNG")
                except Exception as e:
                    os.remove(tmp)
                    raise e
            res["new"] = Path(tmp).stat().st_size
            os.replace(tmp, p)  # atomic on same FS (Windows)
            res["status"] = "resized"
            return res
    except UnidentifiedImageError:
        res["status"]="error"; res["err"]="UnidentifiedImageError"; return res
    except Exception as e:
        res["status"]="error"; res["err"]=repr(e); return res

def main():
    files = all_images(DATASET)
    if not files:
        print("Не найдено изображений в dataset/train и dataset/val")
        return
    stats = Counter(); before = after = 0; errors = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process,f): f for f in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="images", unit="img"):
            r = fut.result()
            stats[r["status"]] += 1
            before += r.get("orig",0) or 0
            after += r.get("new",0) or 0
            if r["status"] == "error":
                errors.append((r["path"], r.get("err")))
    elapsed = time.time()-start
    print(f"\nВсего: {len(files)} | уменьшено: {stats['resized']} | пропущено: {stats['skipped_small'] + stats['skipped_animated']} | ошибок: {stats['error']}")
    print(f"Объём до: {human(before)} -> после: {human(after)} | сэкономлено: {human(max(0, before-after))}")
    print(f"Время: {elapsed:.1f}s")
    if errors:
        print("\nПервые ошибки:")
        for p,e in errors[:10]:
            print(f"- {p} -> {e}")

if __name__ == "__main__":
    main()
