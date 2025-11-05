import os
import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

from model_adapter import AutoGPTQAdapter
from core import run_all

# -------- Paths & switches (EDIT THESE to your env) --------
QA_ROOT = "/mnt/data1/zheyu/workshop/MCQ"
FRAME_ROOT = "/mnt/data1/zheyu/workshop/video_frames/70_110_video/70"
DET_JSON_ROOT = "/mnt/data1/zheyu/workshop/video_imformation/70_110_fisheye_video_imformation/70_jsonl_folder"
OUTPUT_DIR = "/mnt/data1/zheyu/workshop/scripts_930/eva_result_930/70_minicpmo"
OUTPUT_CLIP_ROOT = "/mnt/data1/zheyu/workshop/70_saved_clips"

# Core switches
DEBUG = False
USE_CLIP = True
USE_DET_JSON = True
FILTER_TO_DET_COVERAGE = False

# Sampling / filtering
INCLUDE_CATEGORIES = None
EXCLUDE_CATEGORIES = set()
PER_IMAGE_SAMPLE_K = 2
PER_IMAGE_SAMPLE_SEED = 42
QID_FILTER = None  # set None to disable

# Clip building
FPS = 15
CLIP_FRAMES = 15

# Detection packing
DET_TOPK = 20
ROUND_N = 4
EXCLUDE_LABELS = []

# Model
MODEL_NAME = "openbmb/MiniCPM-o-2_6-int4"


def build_adapter():
    """Load MiniCPM via AutoGPTQ and wrap it into AutoGPTQAdapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoGPTQForCausalLM.from_quantized(
        MODEL_NAME,
        device="cuda:0" if device == "cuda" else "cpu",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        disable_exllama=True,
        disable_exllamav2=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    try:
        model.init_tts()
    except Exception:
        pass
    return AutoGPTQAdapter(model, tokenizer)


def main():
    adapter = build_adapter()

    # Build unified config dict for core
    cfg = {
        # global IO roots
        "QA_ROOT": QA_ROOT,
        "FRAME_ROOT": FRAME_ROOT,
        "DET_JSON_ROOT": DET_JSON_ROOT,
        "OUTPUT_DIR": OUTPUT_DIR,
        "OUTPUT_CLIP_ROOT": OUTPUT_CLIP_ROOT,

        # switches
        "DEBUG": DEBUG,
        "USE_CLIP": USE_CLIP,
        "USE_DET_JSON": USE_DET_JSON,
        "FILTER_TO_DET_COVERAGE": FILTER_TO_DET_COVERAGE,

        # filtering/sampling
        "INCLUDE_CATEGORIES": INCLUDE_CATEGORIES,
        "EXCLUDE_CATEGORIES": EXCLUDE_CATEGORIES,
        "PER_IMAGE_SAMPLE_K": PER_IMAGE_SAMPLE_K,
        "PER_IMAGE_SAMPLE_SEED": PER_IMAGE_SAMPLE_SEED,
        "QID_FILTER": QID_FILTER,

        # clip
        "FPS": FPS,
        "CLIP_FRAMES": CLIP_FRAMES,

        # detection packing
        "DET_TOPK": DET_TOPK,
        "ROUND_N": ROUND_N,
        "EXCLUDE_LABELS": EXCLUDE_LABELS,
    }

    run_all(adapter, cfg)


if __name__ == "__main__":
    main()
