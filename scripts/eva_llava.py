#!/usr/bin/env python3
# eva_llava.py
import os
from model_adapter import LlavaOneVisionAdapter
from core import run_all

# -------- Paths & switches (EDIT THESE to your env) --------
CFG = {
    "QA_ROOT": "/mnt/data1/zheyu/workshop/MCQ",
    "FRAME_ROOT": "/mnt/data1/zheyu/workshop/video_frames/fisheye_video",
    "DET_JSON_ROOT": "/mnt/data1/zheyu/workshop/video_imformation/70_110_fisheye_video_imformation/fisheye_jsonl_folder",
    "OUTPUT_DIR": "/mnt/data1/zheyu/workshop/scripts_930/eva_result_930/180_llava05",
    "OUTPUT_CLIP_ROOT": "/mnt/data1/zheyu/workshop/180_saved_clips",

    # core/common switches
    "DEBUG": False,
    "USE_CLIP": True,             
    "USE_DET_JSON": True,         
    "FILTER_TO_DET_COVERAGE": False,
    "DET_TOPK": 20,
    "ROUND_N": 4,
    "EXCLUDE_LABELS": [],

    # clip 
    "CLIP_FRAMES": 15,            
    "FPS": 15,

    "PER_IMAGE_SAMPLE_K": 2,      
    "PER_IMAGE_SAMPLE_SEED": 42,
    "INCLUDE_CATEGORIES": None,  
    "EXCLUDE_CATEGORIES": set(),
    "QID_FILTER": None,           
}

# -------- Model (EDIT THESE if needed) --------
MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
LOAD_IN_4BIT = False
TARGET_NUM_FRAMES = 24  

def main():
    adapter = LlavaOneVisionAdapter(
        model_name=MODEL_NAME,
        load_in_4bit=LOAD_IN_4BIT,
        target_num_frames=TARGET_NUM_FRAMES,
        force_pad_aspect=True,
        device_map="auto",
    )

    run_all(adapter, CFG)

if __name__ == "__main__":
    main()
