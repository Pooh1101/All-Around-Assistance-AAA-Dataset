from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    """Unified interface. Different models implement chat(msgs, **kwargs)->str."""
    @abstractmethod
    def chat(self, msgs, **kwargs) -> str:
        ...


class AutoGPTQAdapter(ModelAdapter):
    """
    Adapter for models loaded via AutoGPTQForCausalLM with .chat(...).
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def chat(self, msgs, **kwargs) -> str:
        # kwargs may include: use_image_id, max_slice_nums, do_sample, etc.
        out = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            **kwargs
        )
        return (out or "").strip()

import os, re, torch
from typing import List, Dict, Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None
try:
    from torchvision.io import read_video
except Exception:
    read_video = None


class QwenVLAdapter(ModelAdapter):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", load_in_4bit: bool = False):
        self.model_name = model_name
        kw = dict(device_map="auto", trust_remote_code=True)
        if load_in_4bit and BitsAndBytesConfig and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            kw["quantization_config"] = bnb
        else:
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForVision2Seq.from_pretrained(model_name, **kw)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    def _extract_video_and_text(self, msgs: List[Dict[str, Any]]):
        assert isinstance(msgs, list) and len(msgs) == 1 and msgs[0].get("role") == "user", \
            "msgs must be a single-turn list with role=='user'"
        content = msgs[0].get("content", [])
        if not isinstance(content, list):
            content = [content]
        video_path, text = None, None
        if len(content) == 2 and isinstance(content[0], str):
            video_path, text = content[0], content[1]
        elif len(content) == 1:
            text = content[0]
        return (video_path if isinstance(video_path, str) else None), ("" if text is None else str(text))

    def _gen(self, inputs, max_new_tokens=2):
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        tail = out_ids[:, inputs["input_ids"].shape[-1]:]
        return self.processor.tokenizer.batch_decode(tail, skip_special_tokens=True)[0].strip()

    def chat(self, msgs, **kwargs) -> str:
        video_path, text = self._extract_video_and_text(msgs)

        qmsgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        has_video = bool(video_path and os.path.exists(video_path))
        if has_video:
            qmsgs = [{"role": "user", "content": [
                {"type": "video", "video": os.path.abspath(video_path)},
                {"type": "text",  "text": text}
            ]}]

        prompt = self.processor.tokenizer.apply_chat_template(
            qmsgs, tokenize=False, add_generation_prompt=True
        )

        try:
            if has_video:
                inputs = self.processor(
                    text=[prompt],
                    videos=[os.path.abspath(video_path)], 
                    padding=True, return_tensors="pt",
                )
            else:
                inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
            out = self._gen(inputs, max_new_tokens=kwargs.get("max_new_tokens", 2))
        except Exception as e1:
            if has_video and read_video is not None:
                try:
                    frames, _, _ = read_video(os.path.abspath(video_path), pts_unit="sec")
                    if frames.numel() > 0:
                        T = frames.shape[0]
                        if T > 48:
                            idx = torch.linspace(0, T - 1, steps=48).round().long()
                            frames = frames.index_select(0, idx)
                        pil_list = [Image.fromarray(f.cpu().numpy()) for f in frames]
                        inputs = self.processor(text=[prompt], videos=[pil_list], padding=True, return_tensors="pt")
                        out = self._gen(inputs, max_new_tokens=kwargs.get("max_new_tokens", 2))
                    else:
                        out = ""
                except Exception:
                    out = ""
            else:
                out = ""
        return (out or "").strip()

# --- append below your existing adapters in model_adapter.py ---
import os, re, time, cv2
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoProcessor
try:
    # LLaVA-OneVision model class
    from transformers import LlavaOnevisionForConditionalGeneration
except Exception as _e:
    LlavaOnevisionForConditionalGeneration = None
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

class LlavaOneVisionAdapter(ModelAdapter):
    def __init__(self,
                 model_name: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                 load_in_4bit: bool = True,
                 target_num_frames: int = 24,
                 force_pad_aspect: bool = True,
                 device_map: str = "auto"):
        assert LlavaOnevisionForConditionalGeneration is not None, 

        self.model_name = model_name
        self.target_num_frames = int(target_num_frames)

        kw = dict(device_map=device_map, trust_remote_code=True)
        if load_in_4bit and BitsAndBytesConfig and torch.cuda.is_available():
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(self.model_name, **kw)
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, use_fast=True)

        try:
            if force_pad_aspect and hasattr(self.processor, "image_processor"):
                self.processor.image_processor.image_aspect_ratio = "pad"
        except Exception:
            pass

    def _extract_video_and_text(self, msgs: List[Dict[str, Any]]):
        assert isinstance(msgs, list) and len(msgs) == 1 and msgs[0].get("role") == "user", \
            "msgs must be a single-turn list with role=='user'"
        content = msgs[0].get("content", [])
        if not isinstance(content, list):
            content = [content]
        video_path, text = None, None
        if len(content) == 2 and isinstance(content[0], str):
            video_path, text = content[0], content[1]
        elif len(content) == 1:
            text = content[0]
        return (video_path if isinstance(video_path, str) else None), ("" if text is None else str(text))

    def _frames_from_video(self, video_path: str, num_frames: int) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idxs = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
        frames = []
        for t in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        if not frames:
            raise RuntimeError("Video has no readable frames.")
        return frames

    def _generate(self, inputs, max_new_tokens=2) -> str:
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=(
                    self.processor.tokenizer.eos_token_id
                    if getattr(self.processor, "tokenizer", None) and self.processor.tokenizer.eos_token_id is not None
                    else getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
                ),
                pad_token_id=(
                    self.processor.tokenizer.eos_token_id
                    if getattr(self.processor, "tokenizer", None) and self.processor.tokenizer.eos_token_id is not None
                    else getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
                ),
            )
        tail = out_ids[:, inputs["input_ids"].shape[-1]:]
        return self.processor.tokenizer.batch_decode(tail, skip_special_tokens=True)[0].strip()

    def chat(self, msgs, **kwargs) -> str:
        max_new_tokens = int(kwargs.get("max_new_tokens", 2))

        video_path, text = self._extract_video_and_text(msgs)
        has_video = bool(video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 0)

        chat = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        if has_video:
            chat = [{"role": "user", "content": [
                {"type": "text",  "text": text}  
            ]}]

        prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        try:
            if has_video:
                inputs = self.processor(
                    text=prompt,
                    video=os.path.abspath(video_path),     
                    num_frames=self.target_num_frames,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(text=prompt, padding=True, return_tensors="pt")
            out = self._generate(inputs, max_new_tokens=max_new_tokens)
            return (out or "").strip()
        except Exception:
            pass

        try:
            if has_video:
                inputs = self.processor(
                    text=prompt,
                    videos=os.path.abspath(video_path),      
                    num_frames=self.target_num_frames,
                    padding=True,
                    return_tensors="pt",
                )
                out = self._generate(inputs, max_new_tokens=max_new_tokens)
                return (out or "").strip()
        except Exception:
            pass

        try:
            if has_video:
                frames = self._frames_from_video(os.path.abspath(video_path), self.target_num_frames)
                inputs = self.processor(
                    text=prompt,
                    videos=frames,                           
                    padding=True,
                    return_tensors="pt",
                )
                out = self._generate(inputs, max_new_tokens=max_new_tokens)
                return (out or "").strip()
        except Exception:
            pass

        return ""
