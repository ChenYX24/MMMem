"""One backbone serving both the frozen view writer and the adapter-bearing answerer.

Loading the model twice (frozen for views, LoRA for answers) does not fit one 80 GB card next to the
encoders, reranker and Qwen-Image. PEFT lets a single backbone run with the adapter enabled for
"answerer" calls and disabled for every other stage, which is exactly the frozen model.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path


class SharedCalls:
    """Wraps a frozen vmc ModelCalls (or MiniCPMCalls); the adapter is on for "answerer" calls only."""

    def __init__(self, frozen, adapter_path: str) -> None:
        from peft import PeftModel

        from memorium.runtime.inference import file_sha

        self.frozen = frozen
        self.adapter_path = str(adapter_path)
        self.peft = PeftModel.from_pretrained(frozen.model, adapter_path, is_trainable=False).eval()
        self.adapter_hashes = {p.name: file_sha(p) for p in Path(adapter_path).glob("adapter*") if p.is_file()}
        if {"adapter_config.json", "adapter_model.safetensors"} - set(self.adapter_hashes):
            raise ValueError("incomplete answerer adapter")
        # Every stage now runs through the PEFT wrapper; non-answerer stages disable the adapter.
        if hasattr(frozen, "generator"):  # vmc ModelCalls generates through generator.model
            frozen.model = self.peft
            frozen.generator.model = self.peft
        # MiniCPMCalls keeps calling its own model's generate(): PEFT injected the LoRA layers into that model
        # in place, and disable_adapter() switches them off there just the same.
        original = frozen.call

        def call(stage, prompt, images=(), max_new_tokens=512, json_schema=None, **extra):
            context = nullcontext() if stage == "answerer" else self.peft.disable_adapter()
            with context:
                row = original(stage, prompt, images, max_new_tokens=max_new_tokens, json_schema=json_schema,
                               **extra)
            weights = ({"weights": "answerer_lora", "adapter_path": self.adapter_path,
                        "adapter_hashes": self.adapter_hashes} if stage == "answerer"
                       else {"weights": "pretrained_frozen_adapter_disabled"})
            row.update(weights)
            # The frozen call already wrote its record; rewrite it so the file states which weights ran.
            record = frozen.directory / f"{row['call']:08d}.json"
            if record.is_file():
                saved = json.loads(record.read_text(encoding="utf-8"))
                saved.update(weights)
                record.write_text(json.dumps(saved, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            return row

        frozen.call = call
