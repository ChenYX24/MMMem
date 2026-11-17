"""Incremental dense T/V indices over node views.

T embeds text views with Qwen3-Embedding-0.6B; V embeds image views with Qwen3-VL-Embedding-2B.
A node's score on a route is the best score over its views of that modality. The encoding code
follows scripts/run_memorium_route_probe.py, lifted into long-lived encoders that load once.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

QUERY_INSTRUCTION = "Retrieve memory nodes that contain evidence needed to answer the question."


class Encoder(Protocol):
    def encode_items(self, items: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class RouteIndex:
    """One route: node id -> several view vectors, scored by the max over views."""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self._owners: list[str] = []
        self._scopes: list[str] = []
        self._keys: set[tuple[str, str]] = set()
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._owners)

    def add(self, entries: Iterable[tuple[str, str, str]]) -> int:
        """Add (node_id, scope, item) triples; an identical (node_id, item) is never embedded twice."""
        fresh = [(node, scope, item) for node, scope, item in entries if (node, item) not in self._keys]
        if not fresh:
            return 0
        vectors = _unit(self.encoder.encode_items([item for _, _, item in fresh]))
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        for node, scope, item in fresh:
            self._owners.append(node)
            self._scopes.append(scope)
            self._keys.add((node, item))
        return len(fresh)

    def scores(self, query: str, scope: str, allowed: Callable[[str], bool] | None = None) -> dict[str, float]:
        """Best cosine per node over its views, within one unit."""
        if self._vectors is None:
            return {}
        scores = self._vectors @ _unit(self.encoder.encode_query(query))[0]
        best: dict[str, float] = {}
        for index, (owner, owner_scope) in enumerate(zip(self._owners, self._scopes)):
            if owner_scope != scope or (allowed is not None and not allowed(owner)):
                continue
            value = float(scores[index])
            if value > best.get(owner, -np.inf):
                best[owner] = value
        return best

    def rank(self, query: str, scope: str, *, top_k: int, allowed: Callable[[str], bool] | None = None) -> list[str]:
        best = self.scores(query, scope, allowed)
        return [node for node, _ in sorted(best.items(), key=lambda item: (-item[1], item[0]))][:top_k]

    def rank_item(self, item: str, scope: str, *, top_k: int) -> list[str]:
        """Rank by an item embedded like the index entries (e.g. a query image against image views)."""
        if self._vectors is None:
            return []
        scores = self._vectors @ _unit(self.encoder.encode_items([item]))[0]
        best: dict[str, float] = {}
        for index, (owner, owner_scope) in enumerate(zip(self._owners, self._scopes)):
            if owner_scope == scope and float(scores[index]) > best.get(owner, -np.inf):
                best[owner] = float(scores[index])
        return [node for node, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))][:top_k]


def _load_qwen_module(model_path: Path):
    path = model_path / "scripts" / "qwen3_vl_embedding.py"
    if not path.is_file():
        raise ValueError(f"missing official Qwen3-VL embedding module: {path}")
    spec = importlib.util.spec_from_file_location("memorium_qwen3_vl_embedding", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QwenTextEncoder:
    def __init__(self, model_path: str, device: str = "cuda", batch_size: int = 64) -> None:
        from memorium.runtime.model_io import import_sentence_transformer

        self.model = import_sentence_transformer()(model_path, device=device, local_files_only=True)
        self.model.max_seq_length = 512
        self.batch_size = batch_size

    def encode_items(self, items: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(items), batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False, prompt_name="query")


class QwenVisualEncoder:
    """Images in, vectors out; queries are embedded as text with the retrieval instruction."""

    def __init__(self, model_path: str, device: str = "cuda", max_pixels: int = 1843200) -> None:
        import torch
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        qwen = _load_qwen_module(Path(model_path))
        self.formatter = qwen.Qwen3VLEmbedder.__new__(qwen.Qwen3VLEmbedder)
        self.formatter.min_pixels, self.formatter.max_pixels = 4096, max_pixels
        self.formatter.default_instruction = "Represent the user's input."
        self.formatter.processor = Qwen3VLProcessor.from_pretrained(model_path, padding_side="right",
                                                                    local_files_only=True)
        self.formatter.model, loading = qwen.Qwen3VLForEmbedding.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            output_loading_info=True)
        errors = {key: loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys",
                                                     "error_msgs") if loading.get(key)}
        if errors:
            raise ValueError(f"Qwen3-VL embedding load mismatch: {errors}")
        self.formatter.model.eval().to(device)
        self.device = device
        self.vision_tokens = 0

    def _encode(self, item: dict) -> np.ndarray:
        import torch
        from qwen_vl_utils.vision_process import process_vision_info

        conversation = [self.formatter.format_model_input(**item)]
        text = self.formatter.processor.apply_chat_template(conversation, add_generation_prompt=True,
                                                            tokenize=False)
        images, videos, kwargs = process_vision_info(conversation, image_patch_size=16,
                                                     return_video_metadata=True, return_video_kwargs=True)
        values = self.formatter.processor(text=text, images=images, truncation=False, padding=True,
                                          do_resize=False, return_tensors="pt", **kwargs)
        if ("image_grid_thw" in values) != ("image" in item):
            raise ValueError("visual input was silently dropped")
        values = values.to(self.device)
        with torch.inference_mode():
            output = self.formatter.model(**values, use_cache=False)
            feature = self.formatter._pooling_last(output.last_hidden_state, values["attention_mask"])
            vector = torch.nn.functional.normalize(feature, p=2, dim=-1).float().cpu().numpy()[0]
        self.vision_tokens += int(values["input_ids"].eq(self.formatter.model.config.image_token_id).sum())
        return vector

    def encode_items(self, items: Sequence[str]) -> np.ndarray:
        return np.stack([self._encode({"image": path}) for path in items])

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode({"text": query, "instruction": QUERY_INSTRUCTION})


class HashEncoder:
    """Deterministic bag-of-words encoder for CPU tests and dry runs; never used for reported numbers."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        import re
        import zlib

        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        return vector

    def encode_items(self, items: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(item) for item in items])

    def encode_query(self, query: str) -> np.ndarray:
        return self._vector(query)
