"""V2 integrated images: a writer spec, Qwen-Image generation, and a faithfulness check.

The generated picture is a visual entry point to a node, not a transcript of it; it may carry
little text. What must hold is that it depicts nothing the spec does not support.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Sequence

SPEC_SYSTEM = (
    "You design one illustration that helps someone recall a memory. Read the records and return JSON with "
    "keys: facts (list of {text, source_id}; only facts stated in the records), entities (list of names), "
    "scene (one sentence describing a single picture of the most memorable content: who, doing what, where; "
    "no dates, no numbers, no invented details), people (integer number of people in the picture), "
    "genders (list with one entry per person, each 'man', 'woman' or 'unknown', inferred only from the "
    "records, e.g. pronouns or names used), setting (one or two words, e.g. 'dance studio', 'beach'). "
    "The scene must describe people by gender (e.g. 'a man'), never by name alone. Return JSON only."
)

CHECK_SYSTEM = (
    "Look at the image and report only what is visible, as JSON: "
    "{\"people\": <integer count of people>, \"genders\": [\"man\" | \"woman\" | \"unknown\", one per person], "
    "\"setting\": \"<one or two words for the place>\", "
    "\"objects\": [\"<up to 6 main objects>\"]}. Return JSON only."
)


@dataclass(frozen=True)
class IntegrationSpec:
    facts: tuple[tuple[str, str], ...]
    entities: tuple[str, ...]
    scene: str
    people: int | None = None
    setting: str = ""
    genders: tuple[str, ...] = ()

    def digest(self) -> str:
        return hashlib.sha256(json.dumps([self.facts, self.entities, self.scene]).encode()).hexdigest()


def parse_spec(raw: str, allowed_sources: Sequence[str]) -> IntegrationSpec:
    """Keep only facts citing the node's own sources; reject a spec without a usable scene."""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("integration spec is not JSON")
    payload = json.loads(match.group(0))
    allowed = set(allowed_sources)
    facts = tuple((str(item["text"]).strip(), str(item["source_id"]).strip())
                  for item in payload.get("facts", ())
                  if isinstance(item, dict) and str(item.get("source_id", "")).strip() in allowed
                  and str(item.get("text", "")).strip())
    scene = str(payload.get("scene", "")).strip()
    if not facts or not scene:
        raise ValueError("integration spec has no cited facts or no scene")
    entities = tuple(str(name).strip() for name in payload.get("entities", ()) if str(name).strip())
    people = payload.get("people")
    people = int(people) if isinstance(people, (int, float)) or str(people).isdigit() else None
    genders = tuple(g for g in (str(x).strip().lower() for x in payload.get("genders", ()) or ())
                    if g in {"man", "woman", "unknown"})
    return IntegrationSpec(facts, entities, scene, people, str(payload.get("setting", "")).strip().lower(), genders)


def spec_prompt(lines: Sequence[str]) -> str:
    return f"{SPEC_SYSTEM}\n\nRecords:\n" + "\n".join(lines)


def check_prompt(scene: str = "") -> str:
    return CHECK_SYSTEM


_WORD = re.compile(r"[a-z]+")


def _related(expected: str, observed: str) -> bool:
    """Loose setting match: shared content word (studio~studio, stage~stage)."""
    stop = {"a", "an", "the", "of", "in", "room", "area", "place", "indoor", "outdoor"}
    e = set(_WORD.findall(expected.lower())) - stop
    o = set(_WORD.findall(observed.lower())) - stop
    return not e or not o or bool(e & o)


def parse_check(raw: str, spec: "IntegrationSpec | None" = None) -> tuple[bool, dict[str, Any]]:
    """Code, not the VLM, decides: the observed people count and setting must match the spec."""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return False, {"reason": "check output is not JSON", "raw": raw[:400]}
    try:
        seen = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, {"reason": "check output is not JSON", "raw": raw[:400]}
    problems = []
    if spec is not None and spec.people is not None:
        try:
            count = int(seen.get("people"))
        except (TypeError, ValueError):
            count = None
        if count is None or (spec.people <= 2 and count != spec.people) or (spec.people > 2 and count < 3):
            problems.append(f"people: expected {spec.people}, saw {seen.get('people')}")
    if spec is not None and spec.genders and spec.people is not None and spec.people <= 2:
        expected = sorted(g for g in spec.genders if g != "unknown")
        observed = [str(g).strip().lower() for g in (seen.get("genders") or [])]
        for gender in set(expected):
            if observed.count(gender) < expected.count(gender):
                problems.append(f"genders: expected {sorted(spec.genders)}, saw {observed}")
                break
    if spec is not None and spec.setting and not _related(spec.setting, str(seen.get("setting", ""))):
        problems.append(f"setting: expected {spec.setting!r}, saw {seen.get('setting')!r}")
    return not problems, {"observed": seen, "contradictions": problems}


class QwenImageGenerator:
    """Loads Qwen-Image once; generates square images at a fixed step count and seed."""

    def __init__(self, model_path: str, directory: Path, device: str = "cuda", size: int = 768,
                 steps: int = 30, cfg: float = 4.0, offload: bool = True) -> None:
        import torch
        from diffusers import QwenImagePipeline

        self.pipe = QwenImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        # Offload keeps one component on the GPU at a time, so it can share a card with reader and encoders.
        self.pipe.enable_model_cpu_offload() if offload else self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device, self.size, self.steps, self.cfg = device, size, steps, cfg
        self.model_path = model_path

    def __call__(self, scene: str, seed: int) -> str:
        import torch

        key = hashlib.sha256(f"{scene}|{seed}|{self.size}|{self.steps}|{self.cfg}".encode()).hexdigest()[:24]
        path = self.directory / f"v2-{key}.png"
        if not path.exists():
            generator = torch.Generator(device=self.device).manual_seed(seed)
            image = self.pipe(prompt=scene, negative_prompt=" ", width=self.size, height=self.size,
                              num_inference_steps=self.steps, true_cfg_scale=self.cfg,
                              generator=generator).images[0]
            image.save(path)
        return str(path)


@dataclass
class V2Result:
    path: str | None
    spec: IntegrationSpec | None
    verified: bool
    attempts: list[dict]


def integrate(lines: Sequence[str], sources: Sequence[str], *, write: Callable[[str], str],
              draw: Callable[[str, int], str], look: Callable[[str, str], str], seed: int) -> V2Result:
    """Spec -> image -> check; one retry on a failed check, then give up and record why."""
    attempts: list[dict] = []
    spec = None
    for attempt in range(2):
        try:
            spec = parse_spec(write(spec_prompt(lines) + ("\nReturn strictly valid JSON." if attempt else "")), sources)
            break
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            attempts.append({"stage": "spec", "attempt": attempt, "error": str(exc)})
    if spec is None:
        return V2Result(None, None, False, attempts)
    for attempt in range(2):
        path = draw(spec.scene, seed + attempt)
        ok, detail = parse_check(look(path, check_prompt(spec.scene)), spec)
        attempts.append({"stage": "check", "attempt": attempt, "path": path, "scene": spec.scene, "passed": ok,
                         **detail})
        if ok:
            return V2Result(path, spec, True, attempts)
    return V2Result(None, spec, False, attempts)
