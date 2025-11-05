#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import hashlib
from collections import defaultdict, deque
from tqdm import tqdm
import random
import numpy as np

# =================== Paths ===================
input_dir = "" 
output_dir =  ""
os.makedirs(output_dir, exist_ok=True)

# =================== Params ===================
PERSISTENCE_THRESHOLD = 15           
GENERAL_QUESTION_INTERVAL = 150     
BITMAP_OVERLAP_MIN = 0.03            
BG_DILATE_PX = 3                    
DEPTH_NEAR_THR = 0.25                
AREA_BIG_THR = 0.08                  
OVERLAP_MATCH_THR = 0.10             

# =================== Label sets ===================
object_labels = {
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',
    'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear','hair drier','toothbrush','banner','bridge',
    'cardboard','counter','curtain','stairs','tent','towel'
}

background_labels = {
    'sky-other-merged','wall-brick','wall-stone','wall-tile','wall-wood',
    'ceiling-merged','window-other','mirror-stuff','light','tree-merged',
    'grass-merged','mountain-merged','dirt-merged','sand','snow','river',
    'sea','rock-merged','building-other-merged','flower','fruit',
    'paper-merged','food-other-merged','platform','cabinet-merged',
    'shelf','door-stuff','blanket','rug-merged','net','pillow','wall-other-merged',
    'water-other','playingfield','railroad','roof','window-blind'
}

walkable_backgrounds = {
    "pavement-merged", "road", "floor-wood",
    "floor-other-merged", "platform", "playingfield"
}

# =================== Templates ===================
question_templates = {
    # Object
    "count": ("How many {obj} are there?", "Object Information"),                      # 1
    "sidedness": ("Is the {obj} on my {direction}?", "Object Information"),            # 2

    # Safety
    "avoidance": ("Should I avoid the {obj} ahead?", "Safety"),                        # 3
    "blocking_obj_route": ("Is an {obj} blocking the {r}?", "Safety"),                 # 4
    "obstacle_bg": ("Are there obstacles on the {r}?", "Safety"),                      # 5
    "safe_to_walk": ("Is it safe to walk on the {r} ahead?", "Safety"),                # 6
    "general_blocking": ("What is blocking the {r} ahead?", "Safety"),                 # 7 

    # Navigation
    "direction": ("What is the relative position of the {obj}?", "Navigation"),        # 8
    "walking_surface": ("Am I still walking on the {r}?", "Navigation"),               # 9
    "route_confirmation": ("Which region should I follow?", "Navigation"),             # 10
    "presence": ("Is the {obj} visible from here?", "Navigation"),                     # 11
    "closest_object": ("Which of the following is closest in front of me?", "Navigation"),   # 12 
    "farthest_object": ("Which of the following is the farthest in front of me?", "Navigation"), # 13 

    # Surrounding
    "front_presence": ("Is there an {obj} in front of me?", "Vicinity Awareness"),     # 14
    "general_around": ("Which object is nearby?", "Vicinity Awareness"),               # 15 
    "general_direction": ("Which object is on my {direction}?", "Vicinity Awareness"), # 16 
}

# =================== Helpers ===================
def is_background(label):
    return (label in background_labels) or (label in walkable_backgrounds) or label.endswith("-other-merged")

def normalize_label(name):
    return name.replace("-other-merged","").replace("-merged","").replace("_"," ")

def seeded_random(*keys):
    s = "|".join(map(str, keys))
    h = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)
    return random.Random(h)

def yesno_options():
    return ["Yes","No","Not sure","Not visible"]

def direction_options():
    return ["front","left","right","back"]

def load_bitmap_abs(base_dir, mask_path):
    try_paths = []
    if os.path.isabs(mask_path):
        try_paths.append(mask_path)
    try_paths.append(os.path.join(base_dir, mask_path))
    try_paths.append(os.path.join(os.path.dirname(base_dir), mask_path))
    for p in try_paths:
        if os.path.isfile(p):
            try:
                npz = np.load(p)
                packed = npz["packed"]; H, W = map(int, npz["shape"])
                flat = np.unpackbits(packed)[: H*W]
                return flat.reshape((H, W)).astype(bool)
            except Exception:
                continue
    return None

# --- simple bitmap cache to speed up repeated loads ---
_BM_CACHE = {}
def load_bitmap_cached(base_dir, mask_path):
    key = (base_dir, mask_path)
    if key in _BM_CACHE:
        return _BM_CACHE[key]
    bm = load_bitmap_abs(base_dir, mask_path)
    _BM_CACHE[key] = bm
    return bm

def bm_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / float(union)

def union_walkable_bg_bitmap(segments, base_dir, dilate_px=0):
    union = None
    for seg in segments:
        ln = seg["label_name"]
        if ln in walkable_backgrounds:
            mpath = (seg.get("mask") or {}).get("path")
            if not mpath:
                continue
            bm = load_bitmap_abs(base_dir, mpath)
            if bm is None:
                continue
            union = bm.copy() if union is None else (union | bm)
    if union is None:
        return None
    if dilate_px > 0:
        k = dilate_px
        pad = np.pad(union, k, mode='constant', constant_values=False)
        out = np.zeros_like(union, dtype=bool)
        for dy in range(-k, k+1):
            for dx in range(-k, k+1):
                out |= pad[k+dy: k+dy+union.shape[0], k+dx: k+dx+union.shape[1]]
        union = out
    return union

def object_bitmap(seg, base_dir):
    mpath = (seg.get("mask") or {}).get("path")
    if not mpath:
        return None
    return load_bitmap_abs(base_dir, mpath)

def object_overlap_on_bg(object_seg, bg_union_bm, base_dir):
    if bg_union_bm is None:
        return 0.0, False, False
    obm = object_bitmap(object_seg, base_dir)
    if obm is None or obm.shape != bg_union_bm.shape:
        return 0.0, False, False
    inter = (obm & bg_union_bm).sum()
    area = obm.sum()
    if area <= 0:
        return 0.0, False, True
    ratio = inter / float(area)
    return ratio, (ratio >= BITMAP_OVERLAP_MIN), True

def pick_follow_region(segments):
    all_bg = []
    for s in segments:
        if is_background(s["label_name"]) or (s["label_name"] in walkable_backgrounds):
            all_bg.append((
                normalize_label(s["label_name"]),
                s.get("area_ratio", 0.0),
                s.get("direction", None),
            ))
    from collections import defaultdict as _dd
    best_area_per_name = _dd(float)
    for name, ar, _ in all_bg:
        if ar > best_area_per_name[name]:
            best_area_per_name[name] = ar
    options_sorted = sorted(best_area_per_name.items(), key=lambda x: x[1], reverse=True)
    options_names = [n for n,_ in options_sorted][:4] 

    front_walkables = [
        (normalize_label(s["label_name"]), s.get("area_ratio",0.0))
        for s in segments
        if (s["label_name"] in walkable_backgrounds and s.get("direction")=="front")
    ]
    if front_walkables:
        correct = max(front_walkables, key=lambda x: x[1])[0]
    else:
        any_walkables = [
            (normalize_label(s["label_name"]), s.get("area_ratio",0.0))
            for s in segments if (s["label_name"] in walkable_backgrounds)
        ]
        correct = max(any_walkables, key=lambda x: x[1])[0] if any_walkables else "none of the above"

    if correct not in options_names:
        options_names = [correct] + options_names
    options_names = options_names[:4]

    return correct, options_names

# ==== Category rename ====
CATEGORY_MAP = {
    "Object Information": "Object",
    "Safety": "Safety",
    "Navigation": "Navigation",
    "Vicinity Awareness": "Surrounding",
}
def remap_category(cat: str) -> str:
    return CATEGORY_MAP.get(cat, cat)

# ==== Direction helpers ====
def dirs_of_segments(segs):
    d = [s.get("direction") for s in segs if s.get("direction") in {"front","left","right","back"}]
    return sorted(set(d))

def singleton_dir_list(d):
    return [d] if d in {"front","left","right","back"} else []


def pack_mcq(qid_number, question_id, image, category, trigger_type, direction,
             q_text, correct, options, rnd: random.Random, mask_path=None):
    uniq, seen = [], set()
    for o in options:
        if o not in seen:
            uniq.append(o); seen.add(o)
    if correct not in seen:
        uniq = ([correct] + uniq) if len(uniq) < 4 else (uniq[:-1] + [correct])
        seen.add(correct)
    filler_pool = ["none of the above","not applicable","unknown","not visible"]
    i = 0
    while len(uniq) < 4:
        cand = filler_pool[i % len(filler_pool)]
        if cand not in seen and cand != correct:
            uniq.append(cand); seen.add(cand)
        i += 1
    if len(uniq) > 4:
        others = [o for o in uniq if o != correct]
        rnd.shuffle(others)
        uniq = [correct] + others[:3]
    rnd.shuffle(uniq)
    letters = ["A","B","C","D"]
    choices = {letters[i]: uniq[i] for i in range(4)}
    answer_letter = letters[uniq.index(correct)]

    out = {
        "question_id": f"{question_id:06d}",
        "qid_number": qid_number,
        "image": image,
        "category": remap_category(category),
        "trigger_type": trigger_type,
        "direction": direction,         
        "question": q_text,
        "choices": choices,
        "answer": answer_letter,
    }
    if mask_path:
        out["mask"] = mask_path
    return out

# =================== Main ===================
panoptic_files = []
for root, _, files in os.walk(input_dir):
    if "panoptic.jsonl" in files:
        panoptic_files.append(os.path.join(root, "panoptic.jsonl"))

for pan_path in sorted(panoptic_files):
    seq_dir = os.path.dirname(pan_path)                 
    rel_dir = os.path.relpath(seq_dir, input_dir)      
    base_dir_for_masks = seq_dir                        

    output_entries = []
    object_state = defaultdict(lambda: {"hist": deque(maxlen=PERSISTENCE_THRESHOLD), "armed": True})
    last_time_tick = 0
    question_id = 1
    prev_bms_by_label = defaultdict(list)

    with open(pan_path, "r") as infile:
        for idx, line in enumerate(tqdm(infile, desc=f"Processing {rel_dir}/panoptic.jsonl")):
            item = json.loads(line)
            image = item["file_name"]
            segments = item.get("segments", [])

            current_labels = defaultdict(int)
            label_to_segments = defaultdict(list)
            for seg in segments:
                label_to_segments[seg["label_name"]].append(seg)
                if not is_background(seg["label_name"]):
                    current_labels[seg["label_name"]] += 1

            curr_bms_by_label = defaultdict(list)
            for seg in segments:
                ln = seg["label_name"]
                if is_background(ln):
                    continue
                mpath = (seg.get("mask") or {}).get("path")
                if not mpath:
                    continue
                bm = load_bitmap_cached(base_dir_for_masks, mpath)
                if bm is not None:
                    curr_bms_by_label[ln].append(bm)

            continuity_by_label = {}
            all_labels = set(list(prev_bms_by_label.keys()) + list(curr_bms_by_label.keys()))
            for lbl in all_labels:
                has_cont = False
                if curr_bms_by_label.get(lbl) and prev_bms_by_label.get(lbl):
                    for bm_now in curr_bms_by_label[lbl]:
                        if bm_now is None or bm_now.sum() == 0:
                            continue
                        for bm_prev in prev_bms_by_label[lbl]:
                            if bm_prev is None or bm_prev.sum() == 0:
                                continue
                            if bm_iou(bm_now, bm_prev) >= OVERLAP_MATCH_THR:
                                has_cont = True
                                break
                        if has_cont:
                            break
                continuity_by_label[lbl] = has_cont

            for lbl in all_labels:
                cont = bool(continuity_by_label.get(lbl, False))
                st = object_state[lbl]
                st["hist"].append(cont)
                if not cont:
                    st["hist"].clear()
                    st["armed"] = True
            prev_bms_by_label = curr_bms_by_label

            bg_union = union_walkable_bg_bitmap(segments, base_dir_for_masks, dilate_px=BG_DILATE_PX)

            # ---------- Object  ----------
            triggered_labels = []
            for lbl, st in object_state.items():
                if is_background(lbl):
                    continue
                if st["armed"] and (len(st["hist"]) == PERSISTENCE_THRESHOLD) and all(st["hist"]):
                    triggered_labels.append(lbl)
                    st["armed"] = False         

            for tlabel in triggered_labels:
                obj_disp = normalize_label(tlabel)
                reps = label_to_segments.get(tlabel, [])
                rep_seg = max(reps, key=lambda s: s.get("area_ratio", 0.0)) if reps else None
                rep_mask = (rep_seg.get("mask", {}).get("path") if rep_seg else None)
                r_disp = next((normalize_label(s["label_name"]) for s in segments if s["label_name"] in walkable_backgrounds), "road")

                obj_dirs = dirs_of_segments(reps)

                # 1) How many {obj} are there? 
                q_text, q_cat = question_templates["count"]
                correct = str(current_labels.get(tlabel, 0))
                options = [correct, str(max(0, int(correct) - 1)), str(int(correct) + 1)]
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(1, question_id, image, q_cat, "object", obj_dirs,
                             q_text.format(obj=obj_disp), correct, options, rnd, mask_path=rep_mask)
                ); question_id += 1

                # 2) Is the {obj} on my {direction}?
                for direction in ["left", "right", "front", "back"]:
                    q_text, q_cat = question_templates["sidedness"]
                    has_dir = any(seg.get("direction") == direction for seg in reps)
                    correct = "Yes" if has_dir else "No"
                    rnd = seeded_random(rel_dir, image, question_id, q_text + direction)
                    output_entries.append(
                        pack_mcq(2, question_id, image, q_cat, "object", singleton_dir_list(direction),
                                 q_text.format(obj=obj_disp, direction=direction), correct, yesno_options(), rnd, mask_path=rep_mask)
                    ); question_id += 1

                # 3) Should I avoid the {obj} ahead?
                q_text, q_cat = question_templates["avoidance"]
                frontish = any(seg.get("direction") == "front" for seg in reps)
                risky = any(s.get("mean_depth", 1.0) <= DEPTH_NEAR_THR or s.get("area_ratio", 0.0) >= AREA_BIG_THR for s in reps)
                hit_by_bitmap = False
                if rep_seg is not None and bg_union is not None:
                    _, hit_by_bitmap, _ = object_overlap_on_bg(rep_seg, bg_union, base_dir_for_masks)
                correct = "Yes" if (frontish and (risky or hit_by_bitmap)) else "No"
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(3, question_id, image, q_cat, "object", ["front"],
                             q_text.format(obj=obj_disp), correct, yesno_options(), rnd, mask_path=rep_mask)
                ); question_id += 1

                # 4) Is an {obj} blocking the {r}?
                q_text, q_cat = question_templates["blocking_obj_route"]
                is_blocking, used_bitmap = False, False
                if rep_seg is not None:
                    _, is_blocking, used_bitmap = object_overlap_on_bg(rep_seg, bg_union, base_dir_for_masks)
                if not used_bitmap:
                    is_blocking = frontish and risky
                correct = "Yes" if is_blocking else "No"
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(4, question_id, image, q_cat, "object", ["front"],
                             q_text.format(obj=obj_disp, r=r_disp), correct, yesno_options(), rnd, mask_path=rep_mask)
                ); question_id += 1

                # 5) Are there obstacles on the {r}?
                q_text, q_cat = question_templates["obstacle_bg"]
                has_obstacle = False
                used_any_bitmap = False
                if bg_union is not None:
                    for lab, segs in label_to_segments.items():
                        if is_background(lab):
                            continue
                        rep = max(segs, key=lambda s: s.get("area_ratio", 0.0))
                        _, hit, used = object_overlap_on_bg(rep, bg_union, base_dir_for_masks)
                        used_any_bitmap |= used
                        if hit:
                            has_obstacle = True; break
                if not used_any_bitmap:
                    has_obstacle = any(not is_background(s["label_name"]) for s in segments)
                correct = "Yes" if has_obstacle else "No"
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(5, question_id, image, q_cat, "object", [],
                             q_text.format(r=r_disp), correct, yesno_options(), rnd)
                ); question_id += 1

                # 6) Is it safe to walk on the {r} ahead?
                q_text, q_cat = question_templates["safe_to_walk"]
                unsafe = False
                used_any_bitmap = False
                if bg_union is not None:
                    for lab, segs in label_to_segments.items():
                        if is_background(lab):
                            continue
                        rep = max(segs, key=lambda s: s.get("area_ratio", 0.0))
                        _, hit, used = object_overlap_on_bg(rep, bg_union, base_dir_for_masks)
                        used_any_bitmap |= used
                        if hit:
                            unsafe = True; break
                if not used_any_bitmap:
                    fronts = [s for s in segments if (not is_background(s["label_name"]) and s.get("direction") == "front")]
                    closest_front = min([s.get("mean_depth", 1.0) for s in fronts], default=1.0)
                    unsafe = closest_front <= DEPTH_NEAR_THR
                correct = "No" if unsafe else "Yes"
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(6, question_id, image, q_cat, "object", ["front"],
                             q_text.format(r=r_disp), correct, yesno_options(), rnd)
                ); question_id += 1

                # 8) What is the relative position of the {obj}?
                q_text, q_cat = question_templates["direction"]
                dirs = [s.get("direction", "front") for s in reps]
                main_dir = max(set(dirs), key=dirs.count) if dirs else "front"
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(8, question_id, image, q_cat, "object", singleton_dir_list(main_dir),
                             q_text.format(obj=obj_disp), main_dir, direction_options(), rnd, mask_path=rep_mask)
                ); question_id += 1

                # 9) Am I still walking on the {r}?
                q_text, q_cat = question_templates["walking_surface"]
                has_bg = any(s["label_name"] in walkable_backgrounds for s in segments)
                correct = "Yes" if has_bg else "No"
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(9, question_id, image, q_cat, "object", [],
                             q_text.format(r=r_disp), correct, yesno_options(), rnd)
                ); question_id += 1

                # 10) Which region should I follow?
                q_text, q_cat = question_templates["route_confirmation"]
                correct_opt, options = pick_follow_region(segments)
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(10, question_id, image, q_cat, "object", ["front"],
                             q_text, correct_opt, options, rnd)
                ); question_id += 1

                # 11) Is the {obj} visible from here?
                q_text, q_cat = question_templates["presence"]
                present = (current_labels.get(tlabel, 0) > 0)
                correct = "Yes" if present else "No"
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(11, question_id, image, q_cat, "object", obj_dirs,
                             q_text.format(obj=obj_disp), correct, yesno_options(), rnd, mask_path=rep_mask)
                ); question_id += 1

                # 14) Is there an {obj} in front of me?
                q_text, q_cat = question_templates["front_presence"]
                in_front = any(s.get("direction") == "front" for s in reps)
                correct = "Yes" if in_front else "No"
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                output_entries.append(
                    pack_mcq(14, question_id, image, q_cat, "object", ["front"],
                             q_text.format(obj=obj_disp), correct, yesno_options(), rnd, mask_path=rep_mask)
                ); question_id += 1

            # ---------- Time ----------
            if (idx - last_time_tick) >= GENERAL_QUESTION_INTERVAL:
                objs = [s for s in segments if not is_background(s["label_name"])]

                # 7) What is blocking the {r} ahead?
                q_text, q_cat = question_templates["general_blocking"]
                obj_names = sorted({normalize_label(s["label_name"]) for s in objs})
                correct = obj_names[0] if obj_names else "none of the above"
                pool = [o for o in object_labels if normalize_label(o) not in obj_names]
                rnd = seeded_random(rel_dir, image, question_id, q_text)  
                rnd.shuffle(pool)                                          
                options = [correct] + pool[:3]
                r_disp = next((normalize_label(s["label_name"]) for s in segments
                               if s["label_name"] in walkable_backgrounds), "road")
                output_entries.append(
                    pack_mcq(7, question_id, image, q_cat, "time", ["front"],
                             q_text.format(r=r_disp), correct, options, rnd)
                ); question_id += 1

                # 12) Closest in front
                q_text, q_cat = question_templates["closest_object"]
                front_objs = [(normalize_label(s["label_name"]), s.get("mean_depth", None), s.get("area_ratio", 0.0))
                              for s in objs if s.get("direction") == "front"]
                if front_objs:
                    score = [(n, (d if d is not None else (1.0 - a))) for (n, d, a) in front_objs]
                    correct = min(score, key=lambda x: x[1])[0]
                else:
                    correct = "none of the above"
                pool = [o for o in object_labels if normalize_label(o) not in {n for n, _, _ in front_objs}]
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                rnd.shuffle(pool)
                options = [correct] + list(pool)[:3]
                output_entries.append(
                    pack_mcq(12, question_id, image, q_cat, "time", ["front"],
                             q_text, correct, options, rnd)
                ); question_id += 1

                # 13) Farthest in front
                q_text, q_cat = question_templates["farthest_object"]
                if front_objs:
                    score = [(n, (d if d is not None else (1.0 - a))) for (n, d, a) in front_objs]
                    correct = max(score, key=lambda x: x[1])[0]
                else:
                    correct = "none of the above"
                pool = [o for o in object_labels if normalize_label(o) not in {n for n, _, _ in front_objs}]
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                rnd.shuffle(pool)
                options = [correct] + list(pool)[:3]
                output_entries.append(
                    pack_mcq(13, question_id, image, q_cat, "time", ["front"],
                             q_text, correct, options, rnd)
                ); question_id += 1

                # 15) Which object is nearby?
                q_text, q_cat = question_templates["general_around"]
                if objs:
                    score_all = [(normalize_label(s["label_name"]), (s.get("mean_depth", None)), s.get("area_ratio", 0.0)) for s in objs]
                    best = min(score_all, key=lambda x: (x[1] if x[1] is not None else (1.0 - x[2])))
                    correct = best[0]
                else:
                    correct = "none of the above"
                pool = [o for o in object_labels if normalize_label(o) not in {normalize_label(s["label_name"]) for s in objs}]
                rnd = seeded_random(rel_dir, image, question_id, q_text)
                rnd.shuffle(pool)
                options = [correct] + list(pool)[:3]
                output_entries.append(
                    pack_mcq(15, question_id, image, q_cat, "time", dirs_of_segments(objs),
                             q_text, correct, options, rnd)
                ); question_id += 1

                # 16) Which object is on my {direction}?
                for direction in ["left", "right", "front", "back"]:
                    q_text, q_cat = question_templates["general_direction"]
                    dir_objs = [normalize_label(s["label_name"]) for s in objs if s.get("direction") == direction]
                    if dir_objs:
                        rnd_pick = seeded_random(rel_dir, image, question_id, "dirpick")
                        rnd_pick.shuffle(dir_objs)
                        correct = dir_objs[0]
                        pool = [o for o in object_labels if normalize_label(o) not in set(dir_objs)]
                    else:
                        correct = "none of the above"
                        pool = [o for o in object_labels]
                    rnd = seeded_random(rel_dir, image, question_id, q_text + direction)
                    rnd.shuffle(pool)
                    options = [correct] + list(pool)[:3]
                    output_entries.append(
                        pack_mcq(16, question_id, image, q_cat, "time", singleton_dir_list(direction),
                                 q_text.format(direction=direction), correct, options, rnd)
                    ); question_id += 1

                last_time_tick = idx

    rel_name = rel_dir.replace(os.sep, "_")
    os.makedirs(output_dir, exist_ok=True)
    outname = os.path.join(output_dir, f"{rel_name}_questions_mcq.jsonl")
    with open(outname, "w") as fout:
        for e in output_entries:
            fout.write(json.dumps(e) + "\n")
    print(f"[Saved] {rel_name} -> {len(output_entries)} questions -> {outname}")
