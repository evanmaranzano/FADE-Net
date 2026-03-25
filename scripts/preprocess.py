import os
import sys
from PIL import Image

# Add project root to path to allow imports from src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.config import Config


def resize_and_center_crop(image, img_size=224):
    """
    Resize by the short side and center-crop to img_size x img_size.
    """
    w, h = image.size
    scale = img_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    return image.crop((left, top, left + img_size, top + img_size))


def process_afad(source_root, target_root, img_size=224):
    """
    Process AFAD nested structure:
    source_root/age/gender/*.jpg -> target_root/age/gender/*.jpg
    """
    if not os.path.exists(source_root):
        print(f"[WARN] AFAD source folder not found: {source_root}")
        return

    print(f"[INFO] Processing AFAD: {source_root} -> {target_root}", flush=True)
    os.makedirs(target_root, exist_ok=True)

    failed_count = 0
    success_count = 0

    all_folders = os.listdir(source_root)
    print(f"[INFO] Found {len(all_folders)} age folders", flush=True)

    for i, age_folder in enumerate(all_folders):
        if i % 10 == 0:
            print(f"[INFO] Age folder {i}/{len(all_folders)}: {age_folder}", flush=True)

        age_src = os.path.join(source_root, age_folder)
        age_dst = os.path.join(target_root, age_folder)
        if not os.path.isdir(age_src):
            continue

        for gender_folder in os.listdir(age_src):
            gender_src = os.path.join(age_src, gender_folder)
            gender_dst = os.path.join(age_dst, gender_folder)
            if not os.path.isdir(gender_src):
                continue

            os.makedirs(gender_dst, exist_ok=True)
            images = [
                x for x in os.listdir(gender_src)
                if x.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            for file_name in images:
                src_path = os.path.join(gender_src, file_name)
                dst_path = os.path.join(gender_dst, file_name)
                try:
                    image = Image.open(src_path).convert("RGB")
                    out_img = resize_and_center_crop(image, img_size=img_size)
                    out_img.save(dst_path, quality=95)
                    success_count += 1
                except Exception as e:
                    print(f"[ERROR] {file_name}: {e}", flush=True)
                    failed_count += 1

    print(
        f"[DONE] AFAD finished. Success: {success_count}, Failed: {failed_count}",
        flush=True
    )


def main():
    print("[INFO] Start preprocessing (no face alignment).", flush=True)
    cfg = Config()
    base_output_dir = os.path.join(PROJECT_ROOT, "datasets")

    # Adjust this path to your AFAD raw dataset location.
    raw_afad_dir = r"F:\QQFiles\Study\shit\tarball\tarball-master\AFAD-Full.tar\AFAD-Full~\AFAD-Full"

    if os.path.exists(raw_afad_dir):
        process_afad(
            source_root=raw_afad_dir,
            target_root=os.path.join(base_output_dir, "AFAD"),
            img_size=cfg.img_size,
        )
    else:
        print(f"[WARN] AFAD source folder not found: {raw_afad_dir}", flush=True)

    print(f"[DONE] Preprocessing completed. Output: {base_output_dir}", flush=True)


if __name__ == "__main__":
    main()
