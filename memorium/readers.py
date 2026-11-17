"""Inference backends for Qwen and MiniCPM-V."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

_BOX = re.compile(r"<\|(?:begin|end)_of_box\|>")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_SPAN_TAG = re.compile(r"<(?=/?(?:image|slice)>)")


def strip_reader_markup(text: str) -> str:
    """Remove answer-box markers and empty or complete think blocks; keep everything else."""
    return _THINK.sub("", _BOX.sub("", str(text))).strip()


def minicpm_plain_text(prompt: str) -> str:
    """MiniCPM's processor takes <image>, </image>, <slice> and </slice> anywhere in the text as image-span bounds,
    so a document that only quotes one (six LooGLE test documents quote "<image>") opens a span that never closes
    and the processor fails. A zero-width space after the "<" keeps the text as written; a prompt without these
    tags is returned unchanged."""
    return _SPAN_TAG.sub("<​", prompt)


def configure_qwen_image_processor(processor):
    """Match the inference ModelCalls image grid when preparing SFT inputs."""
    image = processor.image_processor
    factor = int(image.patch_size) * int(image.merge_size)
    image.min_pixels, image.max_pixels = factor ** 2, factor ** 2 * 16384
    if hasattr(image, "size"):
        image.size = {"shortest_edge": image.min_pixels, "longest_edge": image.max_pixels}


def make_calls(model_path, backend, directory, max_input_tokens=32768):
    if backend == "minicpmv":
        return MiniCPMCalls(model_path, directory, max_input_tokens=max_input_tokens)
    from memorium.runtime.inference import ModelCalls
    if backend == "glm4v":
        return _StripOutput(ModelCalls(model_path, "qwen35", directory, max_input_tokens=max_input_tokens), backend)
    calls = ModelCalls(model_path, backend, directory, max_input_tokens=max_input_tokens)
    configure_qwen_image_processor(calls.processor)
    return calls


class _StripOutput:
    """Delegates to ModelCalls; cleans the reader's output text and records the real backend name."""

    def __init__(self, inner, backend):
        self.inner, self.backend = inner, backend

    def count(self, prompt, images=()):
        return self.inner.count(prompt, images)

    def call(self, stage, prompt, images=(), max_new_tokens=512, json_schema=None):
        row = self.inner.call(stage, prompt, images=images, max_new_tokens=max_new_tokens, json_schema=json_schema)
        if "output" in row:
            row = {**row, "raw_output": row["output"], "output": strip_reader_markup(row["output"]),
                   "reader_backend": self.backend}
        return row

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_minicpmv(model_path):
    """MiniCPM-V 4.5 in bf16 on the current card (sdpa attention), not yet in eval or train mode."""
    import torch
    from transformers import AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # The remote MiniCPMV class predates transformers 5, whose loader reads all_tied_weights_keys (set by
    # post_init, which MiniCPMV never calls). Its Qwen3-8B head is untied, so an empty mapping is exact.
    model_class = get_class_from_dynamic_module("modeling_minicpmv.MiniCPMV", model_path, local_files_only=True)
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    return AutoModel.from_pretrained(model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
                                     trust_remote_code=True, local_files_only=True).cuda()


class MiniCPMInputs:
    """The processor side of MiniCPMCalls (no weights): the trainer builds exactly the inputs the reader sees."""

    def __init__(self, model_path, max_input_tokens=32768, max_slice_nums=None):
        from transformers import AutoProcessor, AutoTokenizer

        kwargs = {"trust_remote_code": True, "local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path, **kwargs)
        self.max_input_tokens = max_input_tokens
        self.max_slice_nums = max_slice_nums
        self.unk_id = self.tokenizer.convert_tokens_to_ids(self.processor.image_processor.unk_token)

    def inputs(self, prompt, images=(), temporal_ids=None):
        from PIL import Image

        pictures = [Image.open(path).convert("RGB") for path in images]
        content = "\n".join(["(<image>./</image>)"] * len(pictures) + [minicpm_plain_text(prompt)])
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        if temporal_ids is not None:
            if sum(len(group) for group in temporal_ids) != len(pictures):
                raise ValueError("temporal_ids must cover every frame exactly once")
            return self.processor([text], [pictures], max_slice_nums=1, use_image_id=False,
                                  temporal_ids=temporal_ids, return_tensors="pt", max_length=self.max_input_tokens)
        return self.processor([text], [pictures], max_slice_nums=self.max_slice_nums, return_tensors="pt",
                              max_length=self.max_input_tokens)

    def count(self, prompt, images=(), temporal_ids=None):
        return int(self.inputs(prompt, images, temporal_ids)["input_ids"].shape[-1])


class MiniCPMCalls(MiniCPMInputs):
    def __init__(self, model_path, directory, max_input_tokens=32768, max_slice_nums=None):
        super().__init__(model_path, max_input_tokens=max_input_tokens, max_slice_nums=max_slice_nums)
        self.model = load_minicpmv(model_path).eval()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.call_number = len(list(self.directory.glob("*.json")))

    def call(self, stage, prompt, images=(), max_new_tokens=512, json_schema=None, temporal_ids=None):
        if json_schema is not None:
            raise ValueError("minicpmv reader has no structured decoding")
        started = time.perf_counter()
        self.call_number += 1
        record = self.directory / f"{self.call_number:08d}.json"
        row = {"stage": stage, "call": self.call_number, "prompt": prompt, "reader_backend": "minicpmv",
               "images": [{"path": str(p), "sha256": _sha(p)} for p in images], "max_new_tokens": max_new_tokens,
               "temporal_packing": None if temporal_ids is None else [len(group) for group in temporal_ids],
               "weights": "pretrained_frozen_no_task_adapter", "decoding": "greedy", "status": "STARTED"}
        try:
            inputs = self.inputs(prompt, images, temporal_ids)
            ids = inputs["input_ids"]
            total = int(ids.shape[-1])
            row.update(input_tokens=total, input_vision_tokens=int(ids.eq(self.unk_id).sum()))
            if total + max_new_tokens > self.max_input_tokens:
                raise ValueError("model context cap exceeded; no silent truncation")
            record.write_text(json.dumps(row, ensure_ascii=False))
            inputs = inputs.to(self.model.device)
            inputs.pop("image_sizes", None)
            texts = self.model.generate(**inputs, tokenizer=self.tokenizer, max_new_tokens=max_new_tokens,
                                        decode_text=True, do_sample=False, num_beams=1)
            raw = texts[0] if isinstance(texts, (list, tuple)) else str(texts)
            row.update(raw_output=raw, output=strip_reader_markup(raw),
                       output_tokens=len(self.tokenizer.encode(raw, add_special_tokens=False)),
                       output_token_accounting="retokenized_text", status="OK")
        except Exception as exc:  # recorded and re-raised, like ModelCalls
            row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            row["elapsed_seconds"] = time.perf_counter() - started
            record.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return row
