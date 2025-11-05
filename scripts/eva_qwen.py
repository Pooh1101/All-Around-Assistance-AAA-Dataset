#!/usr/bin/env python3
# eva_qwen.py
import os
from model_adapter import QwenVLAdapter
from core import run_all

# -------- Paths & switches (EDIT THESE to your env) --------
CFG = {
    "QA_ROOT": "/mnt/data1/zheyu/workshop/MCQ",
    "FRAME_ROOT": "/mnt/data1/zheyu/workshop/video_frames/raw_GND360frames",
    "DET_JSON_ROOT": "/mnt/data1/zheyu/workshop/video_imformation/360video_information/jsonl_folder",
    "OUTPUT_DIR": "/mnt/data1/zheyu/workshop/scripts_930/eva_result_930/360_qwen3",
    "OUTPUT_CLIP_ROOT": "/mnt/data1/zheyu/workshop/360_saved_clips",

    "DEBUG": False,
    "USE_CLIP": True,
    "CLIP_FRAMES": 15,
    "FPS": 15,

    "USE_DET_JSON": True,
    "FILTER_TO_DET_COVERAGE": False,
    "DET_TOPK": 20,
    "ROUND_N": 4,
    "EXCLUDE_LABELS": [],

    "INCLUDE_CATEGORIES": None,
    "EXCLUDE_CATEGORIES": set(),
    "QID_FILTER": None,            

    "PER_IMAGE_SAMPLE_K": 2,      
    "PER_IMAGE_SAMPLE_SEED": 42,
}

def main():
    adapter = QwenVLAdapter(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    run_all(adapter, CFG)

if __name__ == "__main__":
    main()
