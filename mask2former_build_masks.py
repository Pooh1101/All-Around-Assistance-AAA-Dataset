import os, sys, json, argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_img(p: Path) -> bool: return p.suffix.lower() in IMG_EXTS

def find_sequences(frames_root: Path) -> Dict[Path, List[Path]]:
    seq2imgs = {}
    for school in sorted([d for d in frames_root.iterdir() if d.is_dir()]):
        for aov in sorted([d for d in school.iterdir() if d.is_dir()]):
            for seq in sorted([d for d in aov.iterdir() if d.is_dir()]):
                imgs = []
                for fdir in sorted([d for d in seq.iterdir() if d.is_dir() and d.name.startswith("frame")]):
                    img = fdir / "image.jpg"
                    if img.exists() and is_img(img): imgs.append(img)
                if imgs: seq2imgs[seq] = imgs
    return seq2imgs

def pack_and_save(mask: np.ndarray, out_path: Path, H: int, W: int):
    packed = np.packbits(mask.astype(np.uint8).ravel())
    np.savez_compressed(out_path, packed=packed, shape=np.array([H, W], dtype=np.int32))

def main():
    ap = argparse.ArgumentParser(description="Run Mask2Former and write obj/reg npz in-place")
    ap.add_argument("frames_root", type=Path, help=".../frames")
    ap.add_argument("--model", default="facebook/mask2former-swin-large-coco-panoptic")
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH_SIZE_FRAMES", 4)))
    args = ap.parse_args()

    frames_root = args.frames_root.expanduser().resolve()
    if not frames_root.exists(): sys.exit(f"[Error] FRAMES_ROOT not found: {frames_root}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision("medium")
    except: pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device).eval()
    if device == "cuda": model = model.to(memory_format=torch.channels_last)

    seq_map = find_sequences(frames_root)
    if not seq_map: sys.exit(f"No sequences under: {frames_root}")

    for seq_dir, img_paths in seq_map.items():
        meta_path = seq_dir / "meta.jsonl"
        if meta_path.exists(): meta_path.unlink()
        with meta_path.open("w", encoding="utf-8") as meta_f:
            pbar = tqdm(total=len(img_paths), desc=f"[{seq_dir.relative_to(frames_root)}]", unit="frame")
            for i in range(0, len(img_paths), args.batch):
                batch_paths = img_paths[i:i+args.batch]
                imgs, sizes, fpaths = [], [], []
                for p in batch_paths:
                    try:
                        im = Image.open(p).convert("RGB")
                    except Exception: 
                        continue
                    W, H = im.size
                    imgs.append(im); sizes.append((H, W)); fpaths.append(p)
                if not imgs:
                    pbar.update(len(batch_paths)); continue

                inputs = processor(images=imgs, return_tensors="pt")
                if device == "cuda" and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(memory_format=torch.channels_last)
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device=="cuda")):
                    outputs = model(**inputs)

                panoptics = processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=sizes, label_ids_to_fuse=[]
                )

                for bi, panoptic in enumerate(panoptics):
                    seg_map = panoptic["segmentation"].cpu().numpy()
                    seg_infos = panoptic["segments_info"]
                    H, W = sizes[bi]
                    frame_dir = fpaths[bi].parent  # .../frameXXXX
                    frame_name = frame_dir.name

                    meta_f.write(json.dumps({"frame": frame_name, "file_name": "image.jpg", "w": W, "h": H}) + "\n")

                    obj_idx, reg_idx = 1, 1
                    for seg in seg_infos:
                        mask = (seg_map == seg["id"])
                        is_thing = bool(seg.get("isthing", seg.get("is_thing", True)))
                        out_npz = frame_dir / (f"obj_{obj_idx:02d}.npz" if is_thing else f"reg_{reg_idx:02d}.npz")
                        if is_thing: obj_idx += 1
                        else: reg_idx += 1
                        pack_and_save(mask, out_npz, H, W)
                pbar.update(len(batch_paths))
            pbar.close()
    print("[OK] done ->", frames_root)

if __name__ == "__main__":
    main()
