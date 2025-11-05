import sys, math, argparse
from pathlib import Path
from typing import List
import numpy as np, cv2
from tqdm.auto import tqdm
import py360convert as p360

AOVS = [360, 70, 110, 180]
E2P_MODE = "bilinear"
JPEG_QUALITY = 95
YAW_DEG = 0.0
PITCH_DEG = 0.0

def iter_school_seqs(root: Path):
    for s in sorted([p for p in root.iterdir() if p.is_dir()]):
        for q in sorted([p for p in s.iterdir() if p.is_dir()]):
            yield s.name, q

def list_frames(seq_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in seq_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def e2p(img, fov_deg, out_hw):
    return p360.e2p(img, fov_deg=float(fov_deg), u_deg=YAW_DEG, v_deg=PITCH_DEG,
                    out_hw=out_hw, mode=E2P_MODE)

def equirect_to_fisheye_180(e_img, out_size=640, yaw_deg=0.0, pitch_deg=0.0,
                            y_axis_down=True, return_mask=False):
    H_e, W_e = e_img.shape[:2]; H = W = int(out_size)
    yy, xx = np.meshgrid(np.linspace(-1,1,H), np.linspace(-1,1,W), indexing='ij')
    rr = np.sqrt(xx**2+yy**2); valid = rr<=1.0
    phi = np.arctan2(-yy if y_axis_down else yy, xx)
    theta = rr*(math.pi/2)
    Xc = np.sin(theta)*np.cos(phi); Yc = np.sin(theta)*np.sin(phi); Zc = np.cos(theta)
    yaw = math.radians(yaw_deg); pitch = math.radians(pitch_deg)
    cy,sy = math.cos(yaw), math.sin(yaw); cp,sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], dtype=np.float32)
    dirs = np.stack([Xc,Yc,Zc],-1).astype(np.float32) @ (Rx@Ry).T
    Xw,Yw,Zw = dirs[...,0],dirs[...,1],dirs[...,2]
    lon = np.arctan2(Xw,Zw); lat = np.arctan2(Yw, np.sqrt(Xw**2+Zw**2))
    map_x = ((lon/(2*math.pi))+0.5)*W_e; map_y = (0.5 - lat/math.pi)*H_e
    map_x[~valid] = -1; map_y[~valid] = -1
    out = cv2.remap(e_img, map_x.astype(np.float32), map_y.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return out

def save_jpg(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])

def process_seq(out_root: Path, school: str, seq_dir: Path, size: int):
    frames = list_frames(seq_dir)
    if not frames: 
        tqdm.write(f"[Warn] no images: {seq_dir}"); return
    seq_name = seq_dir.name
    for aov in AOVS:
        seq_out = out_root / "frames" / school / f"{aov}" / seq_name
        for idx, src in enumerate(tqdm(frames, desc=f"{school}-{seq_name}-AOV{aov}", unit="frame")):
            img = cv2.imread(str(src))
            if img is None: 
                tqdm.write(f"[Skip] unreadable: {src}"); continue
            if aov == 360:
                out = img
            elif aov in (70,110):
                out = e2p(img, fov_deg=aov, out_hw=(size, size))
            elif aov == 180:
                out = equirect_to_fisheye_180(img, out_size=size, yaw_deg=0.0, pitch_deg=0.0,
                                              y_axis_down=True, return_mask=False)
            fdir = seq_out / f"frame{idx:04d}"
            save_jpg(fdir / "image.jpg", out)

def main():
    ap = argparse.ArgumentParser(description="Build 360/70/110/180 views")
    ap.add_argument("input_root", type=Path)
    ap.add_argument("output_root", type=Path)
    ap.add_argument("--size", type=int, default=768, help="output size for 70/110/180 (HxW)")
    args = ap.parse_args()
    in_root = args.input_root.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()
    if not in_root.exists(): sys.exit(f"[Error] INPUT_ROOT not found: {in_root}")
    for school, seq_dir in iter_school_seqs(in_root):
        process_seq(out_root, school, seq_dir, size=args.size)
    print("[OK] done ->", out_root / "frames")

if __name__ == "__main__":
    main()
