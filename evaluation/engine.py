import logging
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

logger = logging.getLogger(__name__)


class QwenEngine:
    """Manages the model state and handles generation and logit extraction."""

    def __init__(self, model_id: str, is_qwen3: bool = True):
        model_class = (
            Qwen3VLForConditionalGeneration if is_qwen3 else Qwen2_5_VLForConditionalGeneration
        )
        self.model = model_class.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
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
        content: list[dict[str, Any]] = []

        if video_path:
            content.append({"type": "video", "video": str(video_path), "nframes": nframes})

        content.append({"type": "text", "text": prompt})

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": content})

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
            **video_kwargs,  # ty:ignore[invalid-argument-type]
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
            )  # ty:ignore[invalid-argument-type]

        trimmed_ids = [
            out[len(ins) :] for ins, out in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logger.debug("Raw Model Output [generate]:\n---\n%s\n---", output_text[0])

        return output_text[0]

    def get_scoring_logits(
        self, prompt1: str, reasoning: str, prompt2: str, video_path: Path | None = None
    ):
        nframes = self._get_nframes(video_path)
        msg1_content: list[dict[str, Any]] = []

        if video_path:
            msg1_content.append({"type": "video", "video": str(video_path), "nframes": nframes})

        msg1_content.append({"type": "text", "text": prompt1})

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
            **vid_kw,  # ty:ignore[invalid-argument-type]
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        seq_len = inputs.attention_mask[0].sum().item()

        logger.debug("Actual end of sequence index: %d", seq_len - 1)
        logger.debug("Raw Logits shape: %s", outputs.logits.shape)

        # Decode the highest probability token the model ACTUALLY wanted to predict next
        last_token_id = torch.argmax(outputs.logits[0, seq_len - 1, :]).item()
        last_token_text = repr(self.processor.tokenizer.decode([last_token_id]))
        logger.debug("Predicted top-1 token for sequence end: %s", last_token_text)

        raw_last_token_logits = outputs.logits[0, seq_len - 1, :]
        logger.debug(
            "Raw logits for last token (first 10): %s", raw_last_token_logits[:10].tolist()
        )

        return outputs.logits[0, seq_len - 1, :]

    def calculate_soft_score(
        self, logits: torch.Tensor, target_tokens: list[str], target_weights: list[float]
    ) -> float:
        logger.debug("Target tokens: %s", target_tokens)
        target_ids = []
        for t in target_tokens:
            encoded = self.processor.tokenizer.encode(t, add_special_tokens=False)

            if not encoded:
                logger.warning("Target token '%s' encoded to empty list.", t)
                continue

            token_id = encoded[0]
            target_ids.append(token_id)
            logger.debug(
                "Token: %r -> ID: %d -> Decoded: %r",
                t,
                token_id,
                self.processor.tokenizer.decode([token_id]),
            )

        target_ids_tensor = torch.tensor(target_ids, device=self.device)
        logger.debug("Collected target IDs: %s", target_ids_tensor.tolist())

        full_probs = torch.softmax(logits, dim=-1)
        target_probs = full_probs[target_ids_tensor]

        total_target_prob = target_probs.sum().item()
        if total_target_prob < 0.5:
            top_token_id = torch.argmax(logits).item()
            top_token_text = repr(self.processor.tokenizer.decode([top_token_id]))

            logger.warning(
                "Soft score math breaking! Model wants %s. Mass: %.3f",
                top_token_text,
                total_target_prob,
            )

        normalized_probs = target_probs / (target_probs.sum() + 1e-9)
        score_values = torch.tensor(target_weights, device=self.device)
        return torch.sum(normalized_probs * score_values).item()
