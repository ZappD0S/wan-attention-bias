import gc

import cv2
import torch
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


class QwenVLEngine:
    def __init__(self, model, processor):
        print(f"--- Loading Model: {MODEL_ID} ---")
        self.model = model
        self.processor = processor
        self.model.eval()

    def generate(self, system_prompt, user_text, video_path=None):
        messages = [{"role": "system", "content": system_prompt}]

        if video_path:
            # video_path = str(video_path)
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Qwen prefers an even number of frames for its temporal patches (stride 2)
            nframes_to_use = (total_frames // 2) * 2
            cap.release()

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_path),
                            # "fps": FPS,
                            "nframes": nframes_to_use,
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})

        # processing
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)  # ty:ignore[invalid-assignment]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Inference
        with torch.no_grad():
            # generated_ids = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)

        # Decoding
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()
