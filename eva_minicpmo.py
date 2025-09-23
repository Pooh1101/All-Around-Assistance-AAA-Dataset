import os
import re
import cv2
import json
import time
import glob
import torch
import hashlib
import logging
import random
from tqdm import tqdm
from collections import defaultdict
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# ====================== Global Switches =========================
DEBUG = False
USE_CLIP = True
MAX_FALLBACK_GAP = 0
FILTER_TO_DET_COVERAGE = False
USE_DET_JSON = True  # True: 有 jsonl 就一起丟；False: 一律 video-only
# ===============================================================

# ====================== Config (EDIT THESE) =====================
FPS = 15
CLIP_FRAMES = 15
MODEL_NAME = "openbmb/MiniCPM-o-2_6-int4"

QA_ROOT = "/mnt/data1/zheyu/workshop/inf_360_bitmap_MCQ"
FRAME_ROOT = "/mnt/data1/zheyu/workshop/video_frames/70_110_video/70"
DET_JSON_ROOT = "/mnt/data1/zheyu/workshop/video_imformation/70_110_fisheye_video_imformation/70_jsonl_folder"
OUTPUT_DIR = "/mnt/data1/zheyu/workshop/eva_with_map_result/70_minicpmo"

# 影片輸出根目錄（鏡射 FRAME_ROOT 結構）
OUTPUT_CLIP_ROOT = "/mnt/data1/zheyu/workshop/70_saved_clips"

# category 篩選（None = 不限制）
INCLUDE_CATEGORIES = None
EXCLUDE_CATEGORIES = set()

# ---- 抽題設定（同一張 image 只抽 K 題；None = 不抽樣、全跑）----
PER_IMAGE_SAMPLE_K = 2
PER_IMAGE_SAMPLE_SEED = 42  # 設 None 每次不同
QID_FILTER = {8}

# ---- clip 重用與驗證 ----
REUSE_EXISTING_CLIP = True           # 先嘗試重用既有 clip
CLIP_VALIDATE_REQUIRE_FRAMES = True  # 驗證幀數達標才視為有效
CLIP_MIN_SIZE_BYTES = 10 * 1024      # 小於此視為壞檔（保險用）
# ===============================================================

# ====================== Prompt (MCQ) ===========================
QUESTION_PREFIX_PROMPT = (
    "You are given a short video clip (ending at a specific frame) and Jsonl. This is a MULTIPLE-CHOICE task."
    "Answer the question BASED THE Jsonl First."
    "Assume the video has an Angle of View (AoV) of 70 degrees."
    "When choices (A-D) are provided, answer one letter among A, B, C, or D."
    "Reply with ONE CHARACTER ONLY: A/B/C/D. No explanation."
)

# ====================== Detection packing ======================
DET_TOPK = 20
ROUND_N = 4
EXCLUDE_LABELS = []
# ===============================================================

# ====================== Model Init =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoGPTQForCausalLM.from_quantized(
    MODEL_NAME,
    device="cuda:0" if DEVICE == "cuda" else "cpu",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
    disable_exllama=True,
    disable_exllamav2=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
try:
    model.init_tts()
except Exception:
    pass
# ===============================================================

# ====================== Logging (file only) ====================
LOGGER = logging.getLogger("eva")
LOGGER.propagate = False
LOGGER.setLevel(logging.DEBUG if DEBUG else logging.INFO)

def set_debug_log_file(log_path: str):
    for h in list(LOGGER.handlers):
        LOGGER.removeHandler(h)
    if DEBUG:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[DEBUG] %(message)s")
        fh.setFormatter(fmt)
        LOGGER.addHandler(fh)

def dlog(msg: str):
    if DEBUG:
        LOGGER.debug(msg)
# ===============================================================

# ====================== Small utils ============================
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
# ===============================================================

# ====================== Filename parsing =======================
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
# ===============================================================

# ====================== Folder mapping =========================
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
# ===============================================================

# ====================== Detection JSON helpers =================
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

def summarize_det_index(det_jsonl_path: str):
    if not DEBUG: return
    if not det_jsonl_path or not os.path.exists(det_jsonl_path):
        dlog("DET_SUMMARY: (none)"); return
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
    dlog(f"DET_SUMMARY: total_records={total}")
    for p, d in sorted(by_prefix.items()):
        dlog(f"DET_SUMMARY: prefix={p} cnt={d['cnt']} range=[{d['min']}, {d['max']}]")

def pack_det_json_for_frame(segments_map, frame_name, topk=20, exclude_labels=None, round_n=4):
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

    if segs is None and MAX_FALLBACK_GAP > 0:
        tstem = os.path.splitext(os.path.basename(frame_name))[0]
        tparsed = parse_head_num(tstem)
        if tparsed:
            tpref, tidx, _, _ = tparsed
            best, best_gap = None, 10**9
            for k in segments_map.keys():
                kstem = os.path.splitext(os.path.basename(k))[0]
                kparsed = parse_head_num(kstem)
                if not kparsed: continue
                kpref, kidx, _, _ = kparsed
                if kpref != tpref: continue
                gap = abs(kidx - tidx)
                if gap < best_gap and gap <= MAX_FALLBACK_GAP:
                    best_gap = gap; best = k
            if best is not None:
                segs = segments_map[best]; hit_key = f"{best} (NEAREST Δ={best_gap})"

    if segs is None:
        segs = []; hit_key = "(MISS)"

    dlog(f"DET_HIT_KEY = {hit_key} -> segs={len(segs)} for frame={frame_name}")

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
# ===============================================================

# ====================== Frame & Clip ===========================
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

def derive_clip_out_path(end_frame_path: str, frame_root: str, prefix: str, idx: int, width: int, suffix: str):
    """鏡射 FRAME_ROOT 目錄，把 clip 存到 OUTPUT_CLIP_ROOT 對應路徑。"""
    rel_dir = os.path.relpath(os.path.dirname(end_frame_path), start=frame_root)
    out_dir = os.path.join(OUTPUT_CLIP_ROOT, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"clip_{prefix}{str(idx).zfill(width)}{suffix}.mp4"
    return os.path.join(out_dir, fname)

def extract_clip_and_save(frame_dir: str, frame_root: str, frame_name: str, clip_frames=15, fps=15):
    """
    產生固定 clip_frames 幀（含結尾幀），鏡射存檔；
    若已有有效 clip 則重用。回傳 (clip_path, (prefix, idx, width, suffix, ext))
    """
    end_path, (prefix, idx, width, suffix, ext) = resolve_end_frame(frame_dir, frame_name)
    if not end_path:
        raise FileNotFoundError(f"Ending frame not found for: {frame_name} under {frame_dir}")

    out_path = derive_clip_out_path(end_path, frame_root, prefix, idx, width, suffix)

    # 計算應包含的來源影格清單（也用於期望幀數）
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

    # ===== 先嘗試重用既有 clip =====
    def _is_clip_valid(p: str, expect_frames: int) -> bool:
        if not os.path.isfile(p):
            return False
        if os.path.getsize(p) < CLIP_MIN_SIZE_BYTES:
            return False
        if CLIP_VALIDATE_REQUIRE_FRAMES:
            cap = cv2.VideoCapture(p)
            ok = cap.isOpened()
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
            cap.release()
            # 容許編碼器差 1 幀
            return ok and (cnt + 1 >= expect_frames)
        return True

    if REUSE_EXISTING_CLIP and _is_clip_valid(out_path, expected_frames):
        dlog(f"[CLIP] reuse: {out_path} (expect>={expected_frames})")
        return out_path, (prefix, idx, width, suffix, ext)

    # ===== 重建 clip =====
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
    dlog(f"[CLIP] saved: {out_path} | frames={len(paths)} (target={end-start+1})")
    return out_path, (prefix, idx, width, suffix, ext)
# ===============================================================

# ====================== MCQ prompt & parsing ===================
def format_mcq_prompt(question: str, choices: dict) -> str:
    lines = [
        "Choices:",
        f"A) {choices.get('A','')}",
        f"B) {choices.get('B','')}",
        f"C) {choices.get('C','')}",
        f"D) {choices.get('D','')}"
    ]
    lines.append("\nAnswer ONLY ONE CHARACTER among A, B, C, D.")
    return question.strip() + "\n" + "\n".join(lines)

def parse_mcq_letter(raw_answer: str, choices: dict):
    s = (raw_answer or "").strip()
    if re.search(r"\bx\b", s, flags=re.IGNORECASE):
        return "x", ""
    m = re.search(r"\b([ABCD])\b", s, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        return letter, choices.get(letter, "")
    s_norm = norm_space(s)
    for k, v in choices.items():
        if v and norm_space(v) in s_norm:
            return k, v
    return "x", ""
# ===============================================================

# ====================== LLM call ===============================
def ask_vqa_with_video(video_path: str, det_json_text, question_block: str):
    if det_json_text:
        full_q = (
            f"{QUESTION_PREFIX_PROMPT}"
            f"Here is the detection JSON for the ending frame. Use it alongside the video; "
            f"prioritize the video to decide presence/count/direction.\n"
            f"{det_json_text}\n\n"
            f"{question_block}"
        )
    else:
        full_q = (
            f"{QUESTION_PREFIX_PROMPT}"
            f"(No detection JSON is provided. Use only the video.)\n\n"
            f"{question_block}"
        )

    if USE_CLIP:
        msgs = [{"role": "user", "content": [video_path, full_q]}]
    else:
        msgs = [{"role": "user", "content": [full_q]}]

    t0 = time.time()
    ans = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        use_image_id=False,
        max_slice_nums=2,
        do_sample=False
    )
    llm_ms = (time.time() - t0) * 1000.0
    return (ans or "").strip(), llm_ms
# ===============================================================

# ====================== Counters helpers =======================
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

def _merge_counter(dst: dict, src: dict):
    for k, v in (src or {}).items():
        d = dst.setdefault(k, {"correct": 0, "total": 0})
        d["correct"] += v.get("correct", 0)
        d["total"] += v.get("total", 0)
# ===============================================================

# ====================== Run one QA file ========================
def run_one_qa(qa_path: str, frame_dir: str, det_jsonl_path: str, out_path: str, log_path: str):
    """
    每行需包含: image, question, choices{A..D}, answer(letter)。
    檔末附 summary（per_id[=qid_number] / per_category / per_trigger）。
    """
    set_debug_log_file(log_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(qa_path, "r") as f:
        qa_lines = [json.loads(x) for x in f if x.strip()]

    # category 篩選
    def keep_row(row):
        # 類別過濾（沿用你原本的邏輯）
        cat = row.get("category") or row.get("question_type")
        if EXCLUDE_CATEGORIES and cat in EXCLUDE_CATEGORIES:
            return False
        if INCLUDE_CATEGORIES is not None:
            if not ((cat in INCLUDE_CATEGORIES) or (cat is None)):
                return False

        # ★ qid_number 過濾
        if QID_FILTER is not None:
            qid = row.get("qid_number")
            try:
                qid_int = int(qid) if qid is not None else None
            except Exception:
                qid_int = None
            if qid_int not in QID_FILTER:
                return False

        return True

    before_cnt = len(qa_lines)
    qa_lines = [r for r in qa_lines if keep_row(r)]
    dlog(f"FILTER_CATEGORY: kept {len(qa_lines)}/{before_cnt} rows | include={INCLUDE_CATEGORIES} exclude={EXCLUDE_CATEGORIES}")
    if not qa_lines:
        print(f"[Info] After filtering, no rows left for {os.path.basename(qa_path)}.")

    # ---- 同 image 隨機抽題（可選）----
    if PER_IMAGE_SAMPLE_K is not None:
        rng = random.Random(PER_IMAGE_SAMPLE_SEED) if PER_IMAGE_SAMPLE_SEED is not None else random
        indexed = list(enumerate(qa_lines))  # 保留原始順序索引
        by_image = defaultdict(list)
        for idx0, row in indexed:
            img = row.get("image") or row.get("file_name") or "(unknown)"
            by_image[img].append((idx0, row))

        sampled = []
        kept_cnt = 0
        for img, group in by_image.items():
            take = group if len(group) <= PER_IMAGE_SAMPLE_K else rng.sample(group, PER_IMAGE_SAMPLE_K)
            take.sort(key=lambda x: x[0])  # 維持檔內出現順序
            sampled.extend([row for _, row in take])
            kept_cnt += len(take)

        qa_lines = sampled
        dlog(f"PER_IMAGE_SAMPLE: kept={kept_cnt} after sampling K={PER_IMAGE_SAMPLE_K} per image "
             f"(before={before_cnt}, after_category_filter={len(indexed)})")
        if not qa_lines:
            print(f"[Info] Sampling removed all rows for {os.path.basename(qa_path)}.")

    # 決定是否使用 detection
    if USE_DET_JSON and det_jsonl_path and os.path.exists(det_jsonl_path):
        seg_map = load_segments_map(det_jsonl_path)
        use_det = True
        summarize_det_index(det_jsonl_path)
    else:
        seg_map = {}
        use_det = False
        dlog("Detection disabled or not found. Running in VIDEO-ONLY mode.")

    dlog(f"EVAL_OUT = {out_path}")
    dlog(f"QA_FILE = {qa_path}")
    dlog(f"FRAME_DIR = {frame_dir}")
    dlog(f"DET_JSONL = {det_jsonl_path if use_det else '(none / video-only)'}")
    dlog(f"USE_CLIP = {USE_CLIP}")

    correct = total = 0
    x_count = 0
    det_miss = 0
    results = []
    first_logged = False

    # 統計容器（★ per_id = qid_number）
    per_id_counts       = defaultdict(lambda: {"correct": 0, "total": 0})
    per_trigger_counts  = defaultdict(lambda: {"correct": 0, "total": 0})
    per_category_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in tqdm(qa_lines, desc=os.path.basename(qa_path)):
        frame_name = item.get("image") or item.get("file_name")
        question = (item.get("question") or "").strip()
        choices = item.get("choices") or {}
        gt_letter = (item.get("answer") or "").strip().upper()

        # schema 檢查
        if not (isinstance(choices, dict) and all(k in choices for k in ["A","B","C","D"]) and gt_letter in ["A","B","C","D"]):
            item["predicted_answer_raw"] = "Error: invalid MCQ row"
            results.append(item)
            continue

        # ★ 聚合鍵：qid_number
        per_id_key = item.get("qid_number") or item.get("question_number") or "(unknown-qid_number)"
        trigger_key  = item.get("trigger_type") or "(unknown)"
        category_key = (item.get("category") or item.get("question_type") or "(unknown)")

        # 偵測 JSON（可為 None）
        det_json_text = None
        seg_count = 0
        hit_key = "(NA)"
        det_sha = None

        if use_det:
            det_json_text, seg_count, hit_key = pack_det_json_for_frame(
                seg_map, frame_name, topk=DET_TOPK, exclude_labels=EXCLUDE_LABELS, round_n=ROUND_N
            )
            det_sha = hashlib.sha1(det_json_text.encode()).hexdigest()[:12]
            if hit_key == "(MISS)":
                det_miss += 1
                if FILTER_TO_DET_COVERAGE:
                    item["predicted_answer_raw"] = "Skipped: no detection for frame"
                    results.append(item)
                    continue

        # 產生/重用 clip（鏡射存檔）
        clip_path = None
        try:
            end_p, _ = resolve_end_frame(frame_dir, frame_name)
            if not first_logged:
                dlog(f"END_FRAME_PATH = {end_p if end_p else '(NOT FOUND)'} for image={frame_name}")

            clip_path, (prefix, idx, width, suffix, ext) = extract_clip_and_save(
                frame_dir=frame_dir, frame_root=FRAME_ROOT,
                frame_name=frame_name, clip_frames=CLIP_FRAMES, fps=FPS
            )
            clip_sha = sha1_of_file(clip_path)
            if not first_logged:
                dlog(f"CLIP_FILE = {clip_path}")
                dlog(f"CLIP_SHA = {clip_sha}")
                if det_sha: dlog(f"DET_JSON_SHA = {det_sha}")
                first_logged = True

            q_block = format_mcq_prompt(question, choices)
            ans, llm_ms = ask_vqa_with_video(clip_path, det_json_text, q_block)
            pred_letter, pred_text = parse_mcq_letter(ans, choices)

            # 記錄
            item["predicted_answer_raw"] = ans
            item["predicted_answer_letter"] = pred_letter
            item["predicted_answer_text"] = pred_text
            item["llm_ms"] = round(llm_ms, 2)

            item["det_hit_key"] = hit_key
            item["det_seg_count"] = seg_count
            item["det_used"] = bool(det_json_text)
            item["det_sha"] = det_sha
            item["visual_mode"] = "video" if USE_CLIP else "none"
            item["clip_path"] = clip_path

            # 寫回方便追蹤
            item["qid_number"]  = per_id_key
            item["category"]    = category_key
            item["trigger_type"]= trigger_key

            # 統計
            total += 1
            is_correct = (pred_letter == gt_letter)
            if is_correct: correct += 1
            if pred_letter == "x": x_count += 1

            per_id_counts[per_id_key]["total"] += 1
            if is_correct: per_id_counts[per_id_key]["correct"] += 1

            per_trigger_counts[trigger_key]["total"] += 1
            if is_correct: per_trigger_counts[trigger_key]["correct"] += 1

            per_category_counts[category_key]["total"] += 1
            if is_correct: per_category_counts[category_key]["correct"] += 1

        except Exception as e:
            item["predicted_answer_raw"] = f"Error: {e}"
            item["clip_path"] = clip_path

        results.append(item)

    # Summary & write
    if total > 0:
        acc = correct / total
        per_id      = _finalize_stats_dict(per_id_counts)         # ★ per_id = qid_number
        per_trigger = _finalize_stats_dict(per_trigger_counts)
        per_category= _finalize_stats_dict(per_category_counts)
        det_mode = "jsonl+bitmap" if use_det else "video-only"

        summary = {
            "summary": {
                "file": os.path.basename(qa_path),
                "video_id": "_".join(qa_file_to_tokens(qa_path)),
                "correct": correct, "total": total,
                "accuracy": round(acc, 4), "accuracy_percent": f"{acc:.2%}",
                "x_count": x_count, "x_rate": round(x_count / total, 4),
                "det_miss": det_miss, "det_miss_rate": round(det_miss / max(1, len(qa_lines)), 4),
                "clip_mode": "video" if USE_CLIP else "none",
                "det_mode": det_mode,

                "per_id": per_id,
                "per_category": per_category,
                "per_trigger": per_trigger
            }
        }
        print(f"Accuracy: {correct}/{total} = {acc:.2%} | det_miss={det_miss} | mode={det_mode}")
        print(f"[Info] unique qid_number in this video (per_id keys): {len(per_id_counts)}")
    else:
        summary = {"summary": "No valid MCQ rows to evaluate."}
        print("No valid MCQ rows to evaluate.")

    with open(out_path, "w") as fw:
        for it in results:
            fw.write(json.dumps(it, ensure_ascii=False) + "\n")
        fw.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return correct, total, summary.get("summary")
# ===============================================================

# ====================== Run all & overall & sequence ===========
def run_all(qa_root: str, frame_root: str, det_root: str, out_root: str):
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(OUTPUT_CLIP_ROOT, exist_ok=True)

    # 掃描 QA
    qa_files = []
    for d, _, fs in os.walk(qa_root):
        for fn in fs:
            if fn.endswith(".jsonl"):
                qa_files.append(os.path.join(d, fn))
    qa_files.sort()

    total_c = total_n = 0
    overall_rows = []

    # ALL 聚合容器（只三個維度）
    all_per_id = defaultdict(lambda: {"correct": 0, "total": 0})
    all_per_trigger = defaultdict(lambda: {"correct": 0, "total": 0})
    all_per_category = defaultdict(lambda: {"correct": 0, "total": 0})

    # sequence 聚合（每個 sequence 一份統計）
    seq_aggr = {}  # seq_id -> dict(counters...)

    def _merge_inplace(agg, sub):
        for k, v in (sub or {}).items():
            d = agg.setdefault(k, {"correct": 0, "total": 0})
            d["correct"] += v.get("correct", 0)
            d["total"] += v.get("total", 0)

    for qa_path in qa_files:
        tokens = qa_file_to_tokens(qa_path)
        seq_id = "_".join(tokens) if tokens else os.path.splitext(os.path.basename(qa_path))[0]
        frame_dir = find_dir_contains_tokens(frame_root, tokens)
        det_jsonl = find_det_json(det_root, tokens, frame_dir_hint=frame_dir)

        if not frame_dir:
            print(f"[Skip] No frame dir for {os.path.basename(qa_path)} -> tokens={tokens}")
            continue
        if not det_jsonl:
            print(f"[Warn] No panoptic.jsonl for {os.path.basename(qa_path)} -> tokens={tokens} (video-only)")

        out_subdir = os.path.join(out_root, *tokens)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f"pred_{os.path.basename(qa_path)}")
        log_path = os.path.join(out_subdir, f"debug_{os.path.basename(qa_path).replace('.jsonl','.log')}")

        print(f"\nQA: {os.path.basename(qa_path)}")
        print(f"   Frames: {frame_dir}")
        print(f"   Detect: {det_jsonl if det_jsonl else '(none)'}")
        if INCLUDE_CATEGORIES is not None:
            print(f"   Filter include categories: {INCLUDE_CATEGORIES}")
        if EXCLUDE_CATEGORIES:
            print(f"   Filter exclude categories: {EXCLUDE_CATEGORIES}")

        c, n, summary = run_one_qa(qa_path, frame_dir, det_jsonl, out_path, log_path)
        total_c += c; total_n += n

        if isinstance(summary, dict):
            overall_rows.append(summary)

            # 併入 ALL
            _merge_inplace(all_per_id, summary.get("per_id"))
            _merge_inplace(all_per_trigger, summary.get("per_trigger"))
            _merge_inplace(all_per_category, summary.get("per_category"))

            # 併入該 sequence
            s = seq_aggr.setdefault(seq_id, {
                "correct": 0, "total": 0,
                "per_id": defaultdict(lambda: {"correct": 0, "total": 0}),
                "per_trigger": defaultdict(lambda: {"correct": 0, "total": 0}),
                "per_category": defaultdict(lambda: {"correct": 0, "total": 0}),
            })
            s["correct"] += summary.get("correct", 0)
            s["total"] += summary.get("total", 0)
            _merge_inplace(s["per_id"], summary.get("per_id"))
            _merge_inplace(s["per_trigger"], summary.get("per_trigger"))
            _merge_inplace(s["per_category"], summary.get("per_category"))

    # overall_accuracy.jsonl（逐影片；最後 ALL）
    overall_path = os.path.join(out_root, "overall_accuracy.jsonl")
    with open(overall_path, "w") as sf:
        for s in overall_rows:
            sf.write(json.dumps(s, ensure_ascii=False) + "\n")
        if total_n > 0:
            acc = total_c / total_n
            all_sum = {
                "file": "ALL",
                "correct": total_c,
                "total": total_n,
                "accuracy": round(acc, 4),
                "accuracy_percent": f"{acc:.2%}",
                "clip_mode": "video" if USE_CLIP else "none",
                "det_mode_hint": "mix (some files may be video-only)",
                "per_id": _finalize_stats_dict(all_per_id),
                "per_trigger": _finalize_stats_dict(all_per_trigger),
                "per_category": _finalize_stats_dict(all_per_category)
            }
            sf.write(json.dumps(all_sum, ensure_ascii=False) + "\n")
            print("\n====== Overall Accuracy ======")
            print(f"Total: {total_c}/{total_n} = {acc:.2%} | clip={'video' if USE_CLIP else 'none'}")
        else:
            print("No valid MCQ results to compute overall accuracy.")

    # 每個 sequence 的統計各自輸出
    seq_dir = os.path.join(out_root, "sequence_stats")
    os.makedirs(seq_dir, exist_ok=True)
    for seq_id, ag in seq_aggr.items():
        acc = (ag["correct"] / ag["total"]) if ag["total"] > 0 else 0.0
        payload = {
            "sequence_id": seq_id,
            "correct": ag["correct"],
            "total": ag["total"],
            "accuracy": round(acc, 4),
            "accuracy_percent": f"{acc:.2%}",
            "per_id": _finalize_stats_dict(ag["per_id"]),
            "per_trigger": _finalize_stats_dict(ag["per_trigger"]),
            "per_category": _finalize_stats_dict(ag["per_category"]),
        }
        with open(os.path.join(seq_dir, f"{seq_id}.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
# ===============================================================

# ====================== Entry ==============================
if __name__ == "__main__":
    run_all(QA_ROOT, FRAME_ROOT, DET_JSON_ROOT, OUTPUT_DIR)
