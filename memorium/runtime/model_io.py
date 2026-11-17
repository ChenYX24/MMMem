import json
from pathlib import Path
import re
import sys
import time
ROOT = Path(__file__).resolve().parents[2]


def import_sentence_transformer():
    original_path = sys.path[:]
    sys.modules.pop("datasets", None)
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    finally:
        sys.path = original_path


def json_object(raw):
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)[:-3].strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("expected one JSON object")
    return obj


class Generator:
    def __init__(self, model_path: str, backend: str, max_new_tokens: int):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.processor = None
        if backend in {"qwen25", "qwen35"}:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            if backend == "qwen35":
                from transformers import AutoModelForMultimodalLM

                model_cls = AutoModelForMultimodalLM
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration

                model_cls = Qwen2_5_VLForConditionalGeneration
            self.model = model_cls.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).eval()
            self.tokenizer = self.processor.tokenizer
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True, use_fast=False
        )
        kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if backend == "internvl":
            self.model = AutoModel.from_pretrained(model_path, **kwargs).eval().cuda()
        elif backend == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", **kwargs
            ).eval()
        else:
            raise ValueError(f"unsupported backend: {backend}")

    def generate(
        self, prompt: str, image_path: Path | None = None
    ) -> tuple[str, int, int, float]:
        import torch

        started = time.perf_counter()
        self.last_usage = {}
        vision_tokens = 0
        if self.backend in {"qwen25", "qwen35"}:
            content = []
            if image_path is not None:
                content.append({"type": "image", "image": str(image_path)})
            content.append({"type": "text", "text": prompt})
            messages = [{
                "role": "user",
                "content": content,
            }]
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
            except TypeError:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
            input_tokens = int(inputs["input_ids"].shape[-1])
            image_token_id = getattr(self.model.config, "image_token_id", None)
            vision_tokens = int(inputs["input_ids"].eq(image_token_id).sum().item()) if image_token_id is not None else 0
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            text = self.processor.decode(
                output_ids[0][input_tokens:], skip_special_tokens=True
            )
        elif self.backend == "internvl":
            output = self.model.chat(
                self.tokenizer,
                None,
                prompt,
                {"max_new_tokens": self.max_new_tokens, "do_sample": False},
            )
            text = output[0] if isinstance(output, tuple) else output
            input_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=True))
        else:
            messages = [
                {"role": "system", "content": "Answer with a short span. Use only the supplied evidence."},
                {"role": "user", "content": prompt},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted, return_tensors="pt")
            inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
            input_tokens = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(
                output_ids[0][input_tokens:], skip_special_tokens=True
            )
        self.last_usage = {
            "output_tokens": (int(output_ids.shape[-1]) - input_tokens) if self.backend != "internvl" else len(self.tokenizer.encode(str(text), add_special_tokens=False)),
            "output_token_accounting": "generated_ids" if self.backend != "internvl" else "retokenized_text_estimate",
            "vision_token_accounting": "local_processor",
        }
        return (
            str(text).strip(),
            input_tokens,
            vision_tokens,
            (time.perf_counter() - started) * 1000,
        )
