import os
import re
import cv2
import json
import glob
import time
import psutil
import hashlib
from tqdm import tqdm
from collections import defaultdict

# ========== Prompt pieces ==========
QUESTION_PREFIX_PROMPT = (
    "You are given a short video clip (ending at a specific frame) and Jsonl. "
    "This is a MULTIPLE-CHOICE task. "
    "Answer the question BASED THE Jsonl First. "
    "Assume the video has an Angle of View (AoV) of 180 degrees. "
    "When choices (A-D) are provided, answer one letter among A, B, C, or D. "
    "Reply with ONE CHARACTER ONLY: A/B/C/D. No explanation."
)

def format_mcq_prompt(question: str, choices: dict) -> str:
    lines = [
        "Choices:",
        f"A) {choices.get('A','')}",
        f"B) {choices.get('B','')}",
        f"C) {choices.get('C','')}",
        f"D) {choices.get('D','')}"
    ]
    lines.append("\nAnswer ONLY ONE CHARACTER among A, B, C, D.")
    return (question or "").strip() + "\n" + "\n".join(lines)

def parse_mcq_letter(raw_answer: str, choices: dict):
    s = (raw_answer or "").strip()
    m = re.search(r"\b([ABCD])\b", s, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        return letter, choices.get(letter, "")
    return "x", ""


# ========== Resource monitoring ==========
try:
    from pynvml import *
    nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

def get_resource_usage():
    cpu_percent = float(psutil.cpu_percent())
    mem_mb = float(psutil.virtual_memory().used) / (1024**2)
    util = mem_used_mb = power_w = 0.0
    if _NVML:
        try:
            handle = nvmlDeviceGetHandleByIndex(1)
            util = float(nvmlDeviceGetUtilizationRates(handle).gpu)
            mem_used_mb = float(nvmlDeviceGetMemoryInfo(handle).used) / (1024**2)
            power_w = float(nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception:
            util = mem_used_mb = power_w = 0.0
    return {
        "cpu_percent": round(cpu_percent, 2),
        "cpu_mem_MB": round(mem_mb, 2),
        "gpu_util": round(util, 2),
        "gpu_mem_MB": round(mem_used_mb, 2),
        "gpu_power_W": round(power_w, 2),
    }

def ask_with_timing(adapter, msgs, **kwargs):
    t0 = time.time()
    ans = adapter.chat(msgs, **kwargs)
    return (ans or "").strip(), (time.time() - t0) * 1000.0


# ========== Logging helpers ==========
import logging
LOGGER = logging.getLogger("eva")
LOGGER.propagate = False
LOGGER.setLevel(logging.INFO)

def set_debug_log_file(log_path: str, debug: bool):
    for h in list(LOGGER.handlers):
        LOGGER.removeHandler(h)
    if debug:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[DEBUG] %(message)s")
        fh.setFormatter(fmt)
        LOGGER.addHandler(fh)

def dlog(msg: str, debug: bool):
    if debug:
        LOGGER.debug(msg)


# ========== Small utils ==========
def sha1_of_file(path: str, nbytes: int = 256 * 1024) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(nbytes)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return "NA"

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# ========== Filename parsing ==========
def parse_head_num(stem: str):
    head, tail = (stem.split("_", 1) + [""])[:2]
    m = re.match(r"^(.*?)(\d+)$", head)
    if not m: return None
    prefix, num_str = m.group(1), m.group(2)
    suffix = ("_" + tail) if tail else ""
    return prefix, int(num_str), len(num_str), suffix

def parse_frame_filename(fname: str):
    base = os.path.basename(fname)
    stem, ext = os.path.splitext(base)
    parsed = parse_head_num(stem)
    if not parsed:
        raise ValueError(f"Cannot parse frame index from filename: {fname}")
    prefix, idx, width, suffix = parsed
    return prefix, idx, width, suffix, ext

def canonical_name(path_or_name: str):
    base = os.path.basename(path_or_name)
    stem, ext = os.path.splitext(base)
    parsed = parse_head_num(stem)
    if not parsed: return None
    prefix, idx, width, _ = parsed
    return f"{prefix}{str(idx).zfill(width)}{ext}"


# ========== Folder mapping ==========
def qa_file_to_tokens(qa_filename: str):
    base = os.path.splitext(os.path.basename(qa_filename))[0]
    toks = base.split("_")
    cut = len(toks)
    for i, t in enumerate(toks):
        if t.lower().startswith("questions"):
            cut = i
            break
    return toks[:cut]

def derive_fov_tokens_from_path(path: str):
    parts = os.path.normpath(path).split(os.sep)
    hints = []
    for p in parts:
        lp = p.lower()
        if lp in {"70", "110", "360"} or lp.startswith("fov") or "fov70" in lp or "fov110" in lp or "fov360" in lp:
            hints.append(p)
    return hints

def find_dir_contains_tokens(root: str, tokens):
    if not tokens: return None
    norm_tokens = [t.strip(os.sep) for t in tokens if t]
    candidates = []
    for dirpath, _, _ in os.walk(root):
        parts = os.path.normpath(dirpath).split(os.sep)
        for i in range(len(parts) - len(norm_tokens) + 1):
            if parts[i:i + len(norm_tokens)] == norm_tokens:
                candidates.append(dirpath); break
    if not candidates: return None
    candidates.sort(key=lambda p: len(os.path.normpath(p).split(os.sep)), reverse=True)
    return candidates[0]

def find_det_json(det_root: str, tokens, frame_dir_hint: str = None):
    if frame_dir_hint:
        fov_tokens = derive_fov_tokens_from_path(frame_dir_hint)
    else:
        fov_tokens = []
    if fov_tokens:
        ext_tokens = fov_tokens + tokens
        d = find_dir_contains_tokens(det_root, ext_tokens)
        if d:
            target = os.path.join(d, "panoptic.jsonl")
            if os.path.exists(target): return target
    d = find_dir_contains_tokens(det_root, tokens)
    if d:
        target = os.path.join(d, "panoptic.jsonl")
        if os.path.exists(target): return target
        for dirpath, _, files in os.walk(d):
            for fn in files:
                if fn == "panoptic.jsonl":
                    return os.path.join(dirpath, fn)
    for dirpath, _, files in os.walk(det_root):
        for fn in files:
            if fn == "panoptic.jsonl":
                return os.path.join(dirpath, fn)
    return None


# ========== Detection helpers ==========
def round_floats(x, n=4):
    if isinstance(x, float): return round(x, n)
    if isinstance(x, list):  return [round_floats(v, n) for v in x]
    if isinstance(x, dict):  return {k: round_floats(v, n) for k, v in x.items()}
    return x

def load_segments_map(det_jsonl_path: str):
    mapping = {}
    if not det_jsonl_path: return mapping
    with open(det_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                fname = obj.get("file_name") or obj.get("image")
                segs = obj.get("segments", [])
                if fname is None: continue
                keys = {fname, os.path.basename(fname)}
                cano = canonical_name(fname)
                if cano: keys.add(cano)
                for k in keys:
                    mapping[k] = segs
            except Exception:
                continue
    return mapping

def summarize_det_index(det_jsonl_path: str, debug: bool):
    if not debug: return
    if not det_jsonl_path or not os.path.exists(det_jsonl_path):
        dlog("DET_SUMMARY: (none)", debug); return
    by_prefix = {}
    total = 0
    with open(det_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                fname = obj.get("file_name") or obj.get("image")
                if not fname: continue
                stem = os.path.splitext(os.path.basename(fname))[0]
                parsed = parse_head_num(stem)
                if not parsed: continue
                prefix, idx, _, _ = parsed
                d = by_prefix.setdefault(prefix, {"min": idx, "max": idx, "cnt": 0})
                d["min"] = min(d["min"], idx)
                d["max"] = max(d["max"], idx)
                d["cnt"] += 1
                total += 1
            except Exception:
                continue
    dlog(f"DET_SUMMARY: total_records={total}", debug)
    for p, d in sorted(by_prefix.items()):
        dlog(f"DET_SUMMARY: prefix={p} cnt={d['cnt']} range=[{d['min']}, {d['max']}]", debug)

def pack_det_json_for_frame(segments_map, frame_name, topk=20, exclude_labels=None, round_n=4, debug=False):
    targets = [frame_name, os.path.basename(frame_name)]
    cano = canonical_name(frame_name)
    if cano: targets.append(cano)

    segs = None
    hit_key = None
    for k in targets:
        if k in segments_map:
            segs = segments_map[k]; hit_key = k; break

    if segs is None:
        base = os.path.basename(frame_name)
        for k in segments_map.keys():
            if os.path.basename(k) == base:
                segs = segments_map[k]; hit_key = k; break

    if segs is None:
        segs = []; hit_key = "(MISS)"

    dlog(f"DET_HIT_KEY = {hit_key} -> segs={len(segs)} for frame={frame_name}", debug)

    if exclude_labels:
        segs = [s for s in segs if s.get("label_name") not in exclude_labels]
    segs.sort(key=lambda s: s.get("score", 0.0), reverse=True)
    if topk and topk > 0:
        segs = segs[:topk]

    slim = []
    for s in segs:
        slim.append({
            "label": s.get("label_name"),
            "score": s.get("score"),
            "bbox": s.get("bbox_norm") or s.get("bbox") or [],
            "center": s.get("center_norm") or [],
            "area": s.get("area_ratio", None),
            "direction": s.get("direction", None),
            "average_depth": s.get("average_depth", None),
        })
    payload = {"frame": os.path.basename(frame_name), "segments": round_floats(slim, n=round_n)}
    return json.dumps(payload, ensure_ascii=False), len(slim), hit_key


# ========== Clip building ==========
def resolve_end_frame(frame_dir: str, frame_name: str):
    prefix, idx, width, suffix_q, ext = parse_frame_filename(frame_name)
    exact = os.path.join(frame_dir, f"{prefix}{str(idx).zfill(width)}{suffix_q}{ext}")
    if os.path.exists(exact):
        return exact, (prefix, idx, width, suffix_q, ext)

    pat = os.path.join(frame_dir, f"{prefix}{str(idx).zfill(width)}*{ext}")
    hits = glob.glob(pat)
    if hits:
        hit = sorted(hits)[0]
        stem = os.path.splitext(os.path.basename(hit))[0]
        _, _, _, suffix_hit = parse_head_num(stem)
        return hit, (prefix, idx, width, suffix_hit, ext)

    pat2 = os.path.join(frame_dir, "**", f"{prefix}{str(idx).zfill(width)}*{ext}")
    hits2 = glob.glob(pat2, recursive=True)
    if hits2:
        hit = sorted(hits2)[0]
        stem = os.path.splitext(os.path.basename(hit))[0]
        _, _, _, suffix_hit = parse_head_num(stem)
        return hit, (prefix, idx, width, suffix_hit, ext)

    return None, (prefix, idx, width, suffix_q, ext)

def derive_clip_out_path(end_frame_path: str, frame_root: str, out_clip_root: str, prefix: str, idx: int, width: int, suffix: str):
    rel_dir = os.path.relpath(os.path.dirname(end_frame_path), start=frame_root)
    out_dir = os.path.join(out_clip_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"clip_{prefix}{str(idx).zfill(width)}{suffix}.mp4"
    return os.path.join(out_dir, fname)

def extract_clip_and_save(frame_dir: str, frame_root: str, out_clip_root: str, frame_name: str,
                          clip_frames=15, fps=15, debug=False, min_size_bytes=10*1024, validate_frames=True):
    end_path, (prefix, idx, width, suffix, ext) = resolve_end_frame(frame_dir, frame_name)
    if not end_path:
        raise FileNotFoundError(f"Ending frame not found for: {frame_name} under {frame_dir}")

    out_path = derive_clip_out_path(end_path, frame_root, out_clip_root, prefix, idx, width, suffix)

    start = max(0, idx - (clip_frames - 1))
    end = idx
    def mk(i):
        return os.path.join(os.path.dirname(end_path), f"{prefix}{str(i).zfill(width)}{suffix}{ext}")
    paths = []
    for i in range(start, end + 1):
        p = mk(i)
        if os.path.exists(p):
            paths.append(p)
        else:
            pat_i = os.path.join(frame_dir, "**", f"{prefix}{str(i).zfill(width)}*{ext}")
            hits_i = glob.glob(pat_i, recursive=True)
            if hits_i:
                paths.append(sorted(hits_i)[0])
    if not paths:
        raise FileNotFoundError(f"No frames found in [{start},{end}] near {frame_name}")
    expected_frames = len(paths)

    def _is_clip_valid(p: str, expect_frames: int) -> bool:
        if not os.path.isfile(p): return False
        if os.path.getsize(p) < min_size_bytes: return False
        if not validate_frames: return True
        cap = cv2.VideoCapture(p)
        ok = cap.isOpened()
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
        cap.release()
        return ok and (cnt + 1 >= expect_frames)  # allow off-by-one

    if _is_clip_valid(out_path, expected_frames):
        dlog(f"[CLIP] reuse: {out_path} (expect>={expected_frames})", debug)
        return out_path, (prefix, idx, width, suffix, ext)

    sample = cv2.imread(paths[0])
    if sample is None:
        raise ValueError(f"Cannot read image: {paths[0]}")
    H, W = sample.shape[:2]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        vw.write(im)
    vw.release()
    dlog(f"[CLIP] saved: {out_path} | frames={len(paths)} (target={end-start+1})", debug)
    return out_path, (prefix, idx, width, suffix, ext)


# ========== Counters & finalize ==========
def _finalize_stats_dict(counter_dict):
    out = {}
    for k, v in sorted(counter_dict.items(), key=lambda kv: str(kv[0])):
        tot = v.get("total", 0)
        cor = v.get("correct", 0)
        acc = (cor / tot) if tot > 0 else 0.0
        out[str(k)] = {
            "correct": cor,
            "total": tot,
            "accuracy": round(acc, 4),
            "accuracy_percent": f"{acc:.2%}",
        }
    return out
