from pathlib import Path
from typing import Any

import cv2
import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


class QwenEngine:
    """Manages the model state and handles generation and logit extraction."""

    def __init__(self, model_id: str, is_qwen3: bool = True):
        model_class = (
            Qwen3VLForConditionalGeneration if is_qwen3 else Qwen2_5_VLForConditionalGeneration
        )
        self.model = model_class.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = self.model.device

    def _get_nframes(self, video_path: Path | None) -> int | None:
        if video_path is None or not video_path.exists():
            return None
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return (total_frames // 2) * 2

    def generate(
        self, prompt: str, system_prompt: str | None = None, video_path: Path | None = None
    ) -> str:
        nframes = self._get_nframes(video_path)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if video_path:
            content.append({"type": "video", "video": str(video_path), "nframes": nframes})

        messages = [{"role": "user", "content": content}]

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True
        )

        if video_inputs:
            video_inputs, video_metadatas = zip(*video_inputs, strict=True)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[input_text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=2048,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        trimmed_ids = [
            out[len(ins) :] for ins, out in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def get_scoring_logits(
        self, prompt1: str, reasoning: str, prompt2: str, video_path: Path | None = None
    ):
        nframes = self._get_nframes(video_path)
        msg1_content: list[dict[str, Any]] = [{"type": "text", "text": prompt1}]

        if video_path:
            msg1_content.append({"type": "video", "video": str(video_path), "nframes": nframes})

        messages = [
            {"role": "user", "content": msg1_content},
            {"role": "assistant", "content": [{"type": "text", "text": reasoning}]},
            {"role": "user", "content": [{"type": "text", "text": prompt2}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in, vid_kw = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True
        )

        if vid_in:
            vid_in, vid_metas = zip(*vid_in, strict=True)
            vid_in, vid_metas = list(vid_in), list(vid_metas)
        else:
            vid_metas = None

        inputs = self.processor(
            text=[text],
            images=img_in,
            videos=vid_in,
            video_metadata=vid_metas,
            **vid_kw,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0, -1, :]

    def calculate_soft_score(
        self, logits: torch.Tensor, target_tokens: list[str], target_weights: list[float]
    ) -> float:
        target_ids = [
            self.processor.tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens
        ]

        full_probs = torch.softmax(logits, dim=-1)
        target_probs = full_probs[torch.tensor(target_ids, device=self.device)]

        total_target_prob = target_probs.sum().item()
        if total_target_prob < 0.5:
            top_token_id = torch.argmax(logits).item()
            top_token_text = repr(self.processor.tokenizer.decode([top_token_id]))
            print(
                f"\n[WARNING] Soft score math breaking! Model wants {top_token_text}. Mass: {total_target_prob:.3f}"
            )

        normalized_probs = target_probs / (target_probs.sum() + 1e-9)
        score_values = torch.tensor(target_weights, device=self.device)
        return torch.sum(normalized_probs * score_values).item()
