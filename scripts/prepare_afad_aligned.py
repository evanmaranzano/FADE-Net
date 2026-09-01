"""Generate 100%-replica aligned AFAD images from originals + official aligned_bbox.

Replicates facebase `normalize_img` (prepare_data.py + lib/utils.py):
  out_size = input_size * (1 + 2*input_extension)   # 256*1.1 = 281
  margin   = input_extension + bbox_extension + 2*input_extension*bbox_extension
  crop_image(orig, aligned_bbox, out_size, margin, one_based_bbox=True)

Output keeps original directory layout (age/id/file.jpg) so the training
pipeline only needs to switch its data root; images are 281x281 aligned patches
(val center-crops / trn jitter-crops to 256 at training time).
"""
import json
import os
import cv2
import numpy as np
from tqdm import tqdm


def crop_image(img, bbox, out_size, margin=(0, 0), one_based_bbox=True):
    A = np.float32([bbox[0], bbox[1]])
    B = np.float32([bbox[2], bbox[3]])
    C = np.float32([bbox[4], bbox[5]])
    D = np.float32([bbox[6], bbox[7]])
    if one_based_bbox:
        A = A - 1
        B = B - 1
        C = C - 1
        D = D - 1
    ext_A = A + (A - B) * margin[0] + (A - D) * margin[1]
    ext_B = B + (B - A) * margin[0] + (B - C) * margin[1]
    ext_C = C + (C - D) * margin[0] + (C - B) * margin[1]
    pts1 = np.float32([ext_A, ext_B, ext_C])
    pts2 = np.float32([[0, 0], [out_size[0] - 1, 0],
                       [out_size[0] - 1, out_size[1] - 1]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (out_size[0], out_size[1]))


def main():
    db_path = '/opt/data/instance_gpu_3/fade-net-runtime/FADE-Net/data/official/AFAD-Full_aligned.json'
    orig_root = '/opt/data/instance_gpu_3/AFAD'
    out_root = '/opt/data/instance_gpu_3/AFAD_aligned_281'

    # AFAD_256x256.yaml: input_size=[256,256] input_extension=[0.05,0.05] bbox_extension=[0,0]
    input_size = (256, 256)
    input_extension = (0.05, 0.05)
    bbox_extension = (0, 0)
    out_size = (int(input_size[0] * (1 + 2 * input_extension[0])),
                int(input_size[1] * (1 + 2 * input_extension[1])))
    margin = (input_extension[0] + bbox_extension[0] + 2 * input_extension[0] * bbox_extension[0],
              input_extension[1] + bbox_extension[1] + 2 * input_extension[1] * bbox_extension[1])
    print(f'out_size={out_size} margin={margin}', flush=True)

    db = json.load(open(db_path))
    os.makedirs(out_root, exist_ok=True)
    ok = nobbox = miss = readfail = 0
    for face in tqdm(db):
        bbox = face.get('aligned_bbox', [])
        if not bbox or len(bbox) != 8:
            nobbox += 1
            continue
        ip = face['img_path']  # AFAD/AFAD-Full/15/111/638660-1.jpg
        rel = ip.split('AFAD-Full/')[-1]  # 15/111/638660-1.jpg
        src = os.path.join(orig_root, rel)
        dst = os.path.join(out_root, rel)
        if os.path.exists(dst):
            ok += 1
            continue
        if not os.path.exists(src):
            miss += 1
            continue
        img = cv2.imread(src)
        if img is None:
            readfail += 1
            continue
        out_img = crop_image(img, bbox, out_size, margin=margin, one_based_bbox=True)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, out_img)
        ok += 1
    print(f'DONE ok={ok} nobbox={nobbox} miss={miss} readfail={readfail} total={len(db)}', flush=True)


if __name__ == '__main__':
    main()
