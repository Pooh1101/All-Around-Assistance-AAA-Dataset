import os
import json
import math
import random
from collections import defaultdict

from common import (
    # prompt & parsing
    QUESTION_PREFIX_PROMPT, format_mcq_prompt, parse_mcq_letter,
    # resource + timing
    get_resource_usage, ask_with_timing,
    # logging
    set_debug_log_file, dlog,
    # path & filename helpers
    qa_file_to_tokens, find_dir_contains_tokens, find_det_json,
    # detection
    load_segments_map, summarize_det_index, pack_det_json_for_frame,
    # clip
    extract_clip_and_save, resolve_end_frame,
    # stats
    _finalize_stats_dict,
)

# ---------- Internal: call model with/without clip ----------
def _ask_vqa_with_video(adapter, use_clip: bool, video_path: str, det_json_text: str, question_block: str):
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
    msgs = [{"role": "user", "content": [video_path, full_q]}] if use_clip else [{"role": "user", "content": [full_q]}]
    return ask_with_timing(adapter, msgs, use_image_id=False, max_slice_nums=2, do_sample=False)


# ---------- Run one QA file ----------
def run_one_qa(adapter, cfg: dict):
    """
    cfg keys (required):
      qa_path, frame_dir, det_jsonl_path, out_path, log_path,
      DEBUG, USE_CLIP, FRAME_ROOT, OUTPUT_CLIP_ROOT,
      DET_TOPK, ROUND_N, EXCLUDE_LABELS,
      PER_IMAGE_SAMPLE_K, PER_IMAGE_SAMPLE_SEED,
      INCLUDE_CATEGORIES, EXCLUDE_CATEGORIES, QID_FILTER,
      FILTER_TO_DET_COVERAGE, CLIP_FRAMES, FPS, USE_DET_JSON
    """
    set_debug_log_file(cfg["log_path"], cfg["DEBUG"])
    os.makedirs(os.path.dirname(cfg["out_path"]), exist_ok=True)

    # Load & filter QA rows
    with open(cfg["qa_path"], "r") as f:
        qa_lines = [json.loads(x) for x in f if x.strip()]

    def keep_row(row):
        cat = row.get("category") or row.get("question_type")
        if cfg["EXCLUDE_CATEGORIES"] and cat in cfg["EXCLUDE_CATEGORIES"]:
            return False
        if cfg["INCLUDE_CATEGORIES"] is not None:
            if not ((cat in cfg["INCLUDE_CATEGORIES"]) or (cat is None)):
                return False
        if cfg["QID_FILTER"] is not None:
            qid = row.get("qid_number")
            try:
                qid_int = int(qid) if qid is not None else None
            except Exception:
                qid_int = None
            if qid_int not in cfg["QID_FILTER"]:
                return False
        return True

    qa_lines = [r for r in qa_lines if keep_row(r)]

    # Optional: sample per image
    if cfg["PER_IMAGE_SAMPLE_K"] is not None and qa_lines:
        rng = random.Random(cfg["PER_IMAGE_SAMPLE_SEED"]) if cfg["PER_IMAGE_SAMPLE_SEED"] is not None else random
        by_image = defaultdict(list)
        for idx0, row in enumerate(qa_lines):
            img = row.get("image") or row.get("file_name") or "(unknown)"
            by_image[img].append((idx0, row))
        sampled = []
        for img, group in by_image.items():
            if len(group) <= cfg["PER_IMAGE_SAMPLE_K"]:
                chosen = group
            else:
                chosen = rng.sample(group, cfg["PER_IMAGE_SAMPLE_K"])
            chosen.sort(key=lambda x: x[0])
            sampled.extend([row for _, row in chosen])
        qa_lines = sampled

    # Detection index (optional)
    if cfg["USE_DET_JSON"] and cfg["det_jsonl_path"] and os.path.exists(cfg["det_jsonl_path"]):
        seg_map = load_segments_map(cfg["det_jsonl_path"])
        use_det = True
        summarize_det_index(cfg["det_jsonl_path"], cfg["DEBUG"])
    else:
        seg_map = {}
        use_det = False
        dlog("Detection disabled or not found. Running in VIDEO-ONLY mode.", cfg["DEBUG"])

    # Aggregates
    correct = total = 0
    x_count = det_miss = 0
    per_id_counts       = defaultdict(lambda: {"correct": 0, "total": 0})
    per_trigger_counts  = defaultdict(lambda: {"correct": 0, "total": 0})
    per_category_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    # Resource stats over questions in this file
    res_keys = ["cpu_percent","cpu_mem_MB","gpu_util","gpu_mem_MB","gpu_power_W"]
    res_sum   = {k:0.0 for k in res_keys}
    res_sumsq = {k:0.0 for k in res_keys}
    res_min   = {k:float("inf") for k in res_keys}
    res_max   = {k:float("-inf") for k in res_keys}

    # LLM latency stats
    llm_sum = 0.0
    llm_sumsq = 0.0
    llm_min = float("inf")
    llm_max = float("-inf")

    results = []

    # Loop questions
    for item in qa_lines:
        frame_name = item.get("image") or item.get("file_name")
        question = (item.get("question") or "").strip()
        choices = item.get("choices") or {}
        gt_letter = (item.get("answer") or "").strip().upper()

        # schema check
        if not (isinstance(choices, dict) and all(k in choices for k in ["A","B","C","D"]) and gt_letter in ["A","B","C","D"]):
            results.append({"question_id": item.get("question_id"), "error": "invalid MCQ row"})
            continue

        per_id_key = item.get("qid_number") or item.get("question_number") or "(unknown-qid_number)"
        trigger_key  = item.get("trigger_type") or "(unknown)"
        category_key = (item.get("category") or item.get("question_type") or "(unknown)")

        # det json
        det_json_text = None
        if use_det:
            det_json_text, seg_count, hit_key = pack_det_json_for_frame(
                seg_map, frame_name, topk=cfg["DET_TOPK"], exclude_labels=cfg["EXCLUDE_LABELS"], round_n=cfg["ROUND_N"], debug=cfg["DEBUG"]
            )
            if hit_key == "(MISS)":
                det_miss += 1
                if cfg["FILTER_TO_DET_COVERAGE"]:
                    results.append({"question_id": item.get("question_id"), "skipped": "no detection for frame"})
                    continue

        # clip
        try:
            end_p, _ = resolve_end_frame(cfg["frame_dir"], frame_name)
            dlog(f"END_FRAME_PATH = {end_p if end_p else '(NOT FOUND)'} for image={frame_name}", cfg["DEBUG"])
            clip_path, _ = extract_clip_and_save(
                frame_dir=cfg["frame_dir"], frame_root=cfg["FRAME_ROOT"], out_clip_root=cfg["OUTPUT_CLIP_ROOT"],
                frame_name=frame_name, clip_frames=cfg["CLIP_FRAMES"], fps=cfg["FPS"], debug=cfg["DEBUG"]
            )

            # ask
            q_block = format_mcq_prompt(question, choices)
            ans, llm_ms = _ask_vqa_with_video(adapter, cfg["USE_CLIP"], clip_path, det_json_text, q_block)
            pred_letter, _ = parse_mcq_letter(ans, choices)

            # stats
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

            # resources snapshot (per-question)
            ru = get_resource_usage()
            for k in res_keys:
                v = float(ru.get(k, 0.0))
                res_sum[k]   += v
                res_sumsq[k] += v*v
                res_min[k]    = min(res_min[k], v)
                res_max[k]    = max(res_max[k], v)

            # latency stats
            llm_sum   += llm_ms
            llm_sumsq += llm_ms * llm_ms
            llm_min    = min(llm_min, llm_ms)
            llm_max    = max(llm_max, llm_ms)

            # minimal per-question output (+ resources)
            results.append({
                "question_id": item.get("question_id"),
                "qid_number": item.get("qid_number"),
                "image": item.get("image"),
                "category": item.get("category"),
                "trigger_type": item.get("trigger_type"),
                "direction": item.get("direction"),
                "question": item.get("question"),
                "choices": item.get("choices"),
                "answer": item.get("answer"),
                "mask": item.get("mask"),
                "is_correct": is_correct,
                "predicted_answer_raw": ans,
                "llm_ms": round(llm_ms, 2),
                **ru,
            })

        except Exception as e:
            results.append({"question_id": item.get("question_id"), "error": str(e)})

    # summary
    if total > 0:
        acc = correct / total
        det_mode = "jsonl+bitmap" if use_det else "video-only"

        # avg + std/min/max（以「題」為單位）
        avg_resources = {k: round(res_sum[k] / total, 4) for k in res_sum.keys()}
        std_resources = {}
        for k in res_sum.keys():
            mean = res_sum[k] / total
            var = max((res_sumsq[k] / total) - (mean * mean), 0.0)
            std_resources[k] = round(math.sqrt(var), 4)
        min_resources = {k: (None if res_min[k] == float("inf") else round(res_min[k], 4)) for k in res_min}
        max_resources = {k: (None if res_max[k] == float("-inf") else round(res_max[k], 4)) for k in res_max}

        avg_llm = llm_sum / total if total > 0 else 0.0
        var_llm = max((llm_sumsq / total) - (avg_llm * avg_llm), 0.0)
        std_llm = math.sqrt(var_llm)

        summary = {
            "summary": {
                "file": os.path.basename(cfg["qa_path"]),
                "video_id": "_".join(qa_file_to_tokens(cfg["qa_path"])),
                "correct": correct, "total": total,
                "accuracy": round(acc, 4), "accuracy_percent": f"{acc:.2%}",
                "x_count": x_count, "x_rate": round(x_count / total, 4),
                "det_miss": det_miss, "det_miss_rate": round(det_miss / max(1, len(qa_lines)), 4),
                "clip_mode": "video" if cfg["USE_CLIP"] else "none",
                "det_mode": det_mode,

                # 資源統計（題為單位）
                "avg_resources": avg_resources,
                "resources_stats": {
                    "count": total,
                    "sum": {k: round(res_sum[k], 4) for k in res_sum},
                    "sumsq": {k: round(res_sumsq[k], 4) for k in res_sumsq},
                    "std": std_resources,
                    "min": min_resources,
                    "max": max_resources,
                    "llm_ms": {
                        "avg": round(avg_llm, 4),
                        "std": round(std_llm, 4),
                        "sum": round(llm_sum, 4),
                        "sumsq": round(llm_sumsq, 4),
                        "min": (None if llm_min == float("inf") else round(llm_min, 4)),
                        "max": (None if llm_max == float("-inf") else round(llm_max, 4)),
                    }
                },

                "per_id": _finalize_stats_dict(per_id_counts),
                "per_category": _finalize_stats_dict(per_category_counts),
                "per_trigger": _finalize_stats_dict(per_trigger_counts)
            }
        }
    else:
        summary = {"summary": "No valid MCQ rows to evaluate."}

    # write file
    with open(cfg["out_path"], "w") as fw:
        for it in results:
            fw.write(json.dumps(it, ensure_ascii=False) + "\n")
        fw.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if isinstance(summary.get("summary"), dict):
        s = summary["summary"]
        print(f"Accuracy: {s['correct']}/{s['total']} = {s['accuracy_percent']} | "
              f"det_miss={s['det_miss']} | mode={s['det_mode']}")
    else:
        print("No valid MCQ rows to evaluate.")

    return correct, total, summary.get("summary")


# ---------- Run all ----------
def run_all(adapter, cfg: dict):
    """
    Produce:
      - per-question JSONL for each QA file (with summary at tail)
      - overall_accuracy.jsonl (each file summary + ALL)
      - sequence_stats/<seq_id>.json
      - school_stats/<SCHOOL>.summary.jsonl (含 avg/std/min/max 的 consumption 與 llm_ms 統計)
    """
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(cfg["OUTPUT_CLIP_ROOT"], exist_ok=True)

    # collect QA files
    qa_files = []
    for d, _, fs in os.walk(cfg["QA_ROOT"]):
        for fn in fs:
            if fn.endswith(".jsonl"):
                qa_files.append(os.path.join(d, fn))
    qa_files.sort()

    total_c = total_n = 0
    overall_rows = []
    all_per_id = defaultdict(lambda: {"correct": 0, "total": 0})
    all_per_trigger = defaultdict(lambda: {"correct": 0, "total": 0})
    all_per_category = defaultdict(lambda: {"correct": 0, "total": 0})

    # ALL consumption（聚合整體，單位：題）
    res_keys = ["cpu_percent","cpu_mem_MB","gpu_util","gpu_mem_MB","gpu_power_W"]
    all_res_sum   = {k:0.0 for k in res_keys}
    all_res_sumsq = {k:0.0 for k in res_keys}
    all_res_min   = {k:float("inf") for k in res_keys}
    all_res_max   = {k:float("-inf") for k in res_keys}
    all_llm_sum = 0.0
    all_llm_sumsq = 0.0
    all_llm_min = float("inf")
    all_llm_max = float("-inf")

    seq_aggr = {}

    # per-school aggregates（以題為單位聚合，支援 avg/std/min/max）
    school_aggr = {}

    def _merge_all(agg, sub):
        for k, v in (sub or {}).items():
            d = agg.setdefault(k, {"correct": 0, "total": 0})
            d["correct"] += v.get("correct", 0)
            d["total"] += v.get("total", 0)

    # iterate files
    for qa_path in qa_files:
        tokens = qa_file_to_tokens(qa_path)
        seq_id = "_".join(tokens) if tokens else os.path.splitext(os.path.basename(qa_path))[0]
        frame_dir = find_dir_contains_tokens(cfg["FRAME_ROOT"], tokens)
        det_jsonl = find_det_json(cfg["DET_JSON_ROOT"], tokens, frame_dir_hint=frame_dir)

        if not frame_dir:
            print(f"[Skip] No frame dir for {os.path.basename(qa_path)} -> tokens={tokens}")
            continue

        out_subdir = os.path.join(cfg["OUTPUT_DIR"], *tokens)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f"pred_{os.path.basename(qa_path)}")
        log_path = os.path.join(out_subdir, f"debug_{os.path.basename(qa_path).replace('.jsonl','.log')}")

        file_cfg = dict(cfg)
        file_cfg.update({
            "qa_path": qa_path,
            "frame_dir": frame_dir,
            "det_jsonl_path": det_jsonl,
            "out_path": out_path,
            "log_path": log_path,
        })

        c, n, summary = run_one_qa(adapter, file_cfg)
        total_c += c; total_n += n

        if isinstance(summary, dict):
            overall_rows.append(summary)
            _merge_all(all_per_id, summary.get("per_id"))
            _merge_all(all_per_trigger, summary.get("per_trigger"))
            _merge_all(all_per_category, summary.get("per_category"))

            # ==== 聚合 ALL（以題為單位）====
            rs = summary.get("resources_stats") or {}
            cnt = int(rs.get("count", 0))
            if cnt > 0:
                # sum/sumsq
                s_sum   = rs.get("sum", {})
                s_sumsq = rs.get("sumsq", {})
                s_min   = rs.get("min", {})
                s_max   = rs.get("max", {})
                for k in res_keys:
                    all_res_sum[k]   += float(s_sum.get(k, 0.0))
                    all_res_sumsq[k] += float(s_sumsq.get(k, 0.0))
                    if k in s_min and s_min[k] is not None:
                        all_res_min[k] = min(all_res_min[k], float(s_min[k]))
                    if k in s_max and s_max[k] is not None:
                        all_res_max[k] = max(all_res_max[k], float(s_max[k]))
                # llm
                llm = rs.get("llm_ms", {}) or {}
                all_llm_sum   += float(llm.get("sum", 0.0))
                all_llm_sumsq += float(llm.get("sumsq", 0.0))
                if llm.get("min") is not None:
                    all_llm_min = min(all_llm_min, float(llm.get("min")))
                if llm.get("max") is not None:
                    all_llm_max = max(all_llm_max, float(llm.get("max")))

            # ==== per-sequence ====
            s = seq_aggr.setdefault(seq_id, {
                "correct": 0, "total": 0,
                "per_id": defaultdict(lambda: {"correct": 0, "total": 0}),
                "per_trigger": defaultdict(lambda: {"correct": 0, "total": 0}),
                "per_category": defaultdict(lambda: {"correct": 0, "total": 0}),
            })
            s["correct"] += summary.get("correct", 0)
            s["total"]   += summary.get("total", 0)
            _merge_all(s["per_id"], summary.get("per_id"))
            _merge_all(s["per_trigger"], summary.get("per_trigger"))
            _merge_all(s["per_category"], summary.get("per_category"))

            # ==== per-school（以題為單位）====
            school = (tokens[0] if tokens else "UNKNOWN").upper()
            g = school_aggr.setdefault(school, {
                "files": 0,
                "correct": 0, "total": 0,
                "det_miss": 0, "x_count": 0,
                "res_sum":   {k:0.0 for k in res_keys},
                "res_sumsq": {k:0.0 for k in res_keys},
                "res_min":   {k:float("inf") for k in res_keys},
                "res_max":   {k:float("-inf") for k in res_keys},
                "llm_sum": 0.0, "llm_sumsq": 0.0, "llm_min": float("inf"), "llm_max": float("-inf")
            })
            g["files"]   += 1
            g["correct"] += summary.get("correct", 0)
            g["total"]   += summary.get("total", 0)
            g["det_miss"]+= summary.get("det_miss", 0)
            g["x_count"] += summary.get("x_count", 0)

            if cnt > 0:
                for k in res_keys:
                    g["res_sum"][k]   += float(rs.get("sum", {}).get(k, 0.0))
                    g["res_sumsq"][k] += float(rs.get("sumsq", {}).get(k, 0.0))
                    miv = rs.get("min", {}).get(k, None)
                    mav = rs.get("max", {}).get(k, None)
                    if miv is not None:
                        g["res_min"][k] = min(g["res_min"][k], float(miv))
                    if mav is not None:
                        g["res_max"][k] = max(g["res_max"][k], float(mav))
                g["llm_sum"]   += float(rs.get("llm_ms", {}).get("sum", 0.0))
                g["llm_sumsq"] += float(rs.get("llm_ms", {}).get("sumsq", 0.0))
                if rs.get("llm_ms", {}).get("min") is not None:
                    g["llm_min"] = min(g["llm_min"], float(rs["llm_ms"]["min"]))
                if rs.get("llm_ms", {}).get("max") is not None:
                    g["llm_max"] = max(g["llm_max"], float(rs["llm_ms"]["max"]))

    # overall_accuracy.jsonl
    overall_path = os.path.join(cfg["OUTPUT_DIR"], "overall_accuracy.jsonl")
    with open(overall_path, "w") as sf:
        for s in overall_rows:
            sf.write(json.dumps(s, ensure_ascii=False) + "\n")
        if total_n > 0:
            acc = total_c / total_n
            # ALL avg/std/min/max（以題為單位）
            all_avg_res = {k: round(all_res_sum[k] / total_n, 4) for k in res_keys}
            all_std_res = {}
            for k in res_keys:
                mean = all_res_sum[k] / total_n
                var = max((all_res_sumsq[k] / total_n) - (mean * mean), 0.0)
                all_std_res[k] = round(math.sqrt(var), 4)
            all_min_res = {k: (None if all_res_min[k] == float("inf") else round(all_res_min[k], 4)) for k in res_keys}
            all_max_res = {k: (None if all_res_max[k] == float("-inf") else round(all_res_max[k], 4)) for k in res_keys}

            all_avg_llm = all_llm_sum / total_n if total_n > 0 else 0.0
            all_std_llm = math.sqrt(max((all_llm_sumsq / total_n) - (all_avg_llm * all_avg_llm), 0.0))
            all_min_llm = None if all_llm_min == float("inf") else round(all_llm_min, 4)
            all_max_llm = None if all_llm_max == float("-inf") else round(all_llm_max, 4)

            all_sum = {
                "file": "ALL",
                "correct": total_c,
                "total": total_n,
                "accuracy": round(acc, 4),
                "accuracy_percent": f"{acc:.2%}",
                "clip_mode": "video" if cfg["USE_CLIP"] else "none",
                "det_mode_hint": "mix (some files may be video-only)",

                "resources": {
                    "avg": all_avg_res,
                    "std": all_std_res,
                    "min": all_min_res,
                    "max": all_max_res,
                },
                "llm_ms": {
                    "avg": round(all_avg_llm, 4),
                    "std": round(all_std_llm, 4),
                    "min": all_min_llm,
                    "max": all_max_llm,
                },

                "per_id": _finalize_stats_dict(all_per_id),
                "per_trigger": _finalize_stats_dict(all_per_trigger),
                "per_category": _finalize_stats_dict(all_per_category)
            }
            sf.write(json.dumps(all_sum, ensure_ascii=False) + "\n")
            print("\n====== Overall Accuracy ======")
            print(f"Total: {total_c}/{total_n} = {acc:.2%} | clip={'video' if cfg['USE_CLIP'] else 'none'}")
        else:
            print("No valid MCQ results to compute overall accuracy.")

    # per-sequence stats（同前版）
    seq_dir = os.path.join(cfg["OUTPUT_DIR"], "sequence_stats")
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

    # per-school summaries（含 consumption 全統計）
    school_dir = os.path.join(cfg["OUTPUT_DIR"], "school_stats")
    os.makedirs(school_dir, exist_ok=True)
    for school, ag in sorted(school_aggr.items()):
        c, n = ag["correct"], ag["total"]
        acc = (c / n) if n > 0 else 0.0

        # 平均/標準差/極值（以題為單位）
        avg_res = {k: round(ag["res_sum"][k] / n, 4) if n > 0 else 0.0 for k in res_keys}
        std_res = {}
        for k in res_keys:
            mean = (ag["res_sum"][k] / n) if n > 0 else 0.0
            var = max(((ag["res_sumsq"][k] / n) - (mean * mean)) if n > 0 else 0.0, 0.0)
            std_res[k] = round(math.sqrt(var), 4)
        min_res = {k: (None if ag["res_min"][k] == float("inf") else round(ag["res_min"][k], 4)) for k in res_keys}
        max_res = {k: (None if ag["res_max"][k] == float("-inf") else round(ag["res_max"][k], 4)) for k in res_keys}

        avg_llm = (ag["llm_sum"] / n) if n > 0 else 0.0
        std_llm = math.sqrt(max(((ag["llm_sumsq"] / n) - (avg_llm * avg_llm)) if n > 0 else 0.0, 0.0))
        min_llm = None if ag["llm_min"] == float("inf") else round(ag["llm_min"], 4)
        max_llm = None if ag["llm_max"] == float("-inf") else round(ag["llm_max"], 4)

        payload = {
            "school": school,
            "files": ag["files"],
            "correct": c,
            "total": n,
            "accuracy": round(acc, 4),
            "accuracy_percent": f"{acc:.2%}",
            "x_count": ag["x_count"],
            "det_miss": ag["det_miss"],

            "resources": {
                "avg": avg_res,
                "std": std_res,
                "min": min_res,
                "max": max_res,
            },
            "llm_ms": {
                "avg": round(avg_llm, 4),
                "std": round(std_llm, 4),
                "min": min_llm,
                "max": max_llm,
            }
        }
        with open(os.path.join(school_dir, f"{school}.summary.jsonl"), "w") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
