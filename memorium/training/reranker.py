"""Qwen3-Reranker as a yes/no cross-encoder: score = logit(yes) - logit(no) at the final position.

Training pairs come from RSI items on train units: positives are candidates holding the item's evidence,
hard negatives are other candidates the untrained T route ranked highly for the same question.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

INSTRUCTION = "Given a question about a long conversation, judge whether the memory record helps answer it."
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def pair_text(question: str, document: str) -> str:
    return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: {document}{SUFFIX}"


class Reranker:
    def __init__(self, model_path: str, adapter: str | None = None, device: str = "cuda",
                 max_length: int = 1024, batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        self.device, self.max_length, self.batch_size = device, max_length, batch_size

    def logits(self, pairs: Sequence[str]):
        batch = self.tokenizer(list(pairs), padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
        last = self.model(**batch).logits[:, -1, :]
        return last[:, self.yes] - last[:, self.no]

    def __call__(self, question: str, documents: Sequence[str]) -> list[float]:
        import torch

        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(documents), self.batch_size):
                chunk = [pair_text(question, d) for d in documents[start:start + self.batch_size]]
                scores += self.logits(chunk).float().cpu().tolist()
        return scores


def load_reranker(spec: str) -> Callable[[str, Sequence[str]], Sequence[float]]:
    """spec = '<base model dir>' or '<base model dir>::<adapter dir>'."""
    base, _, adapter = spec.partition("::")
    return Reranker(base, adapter or None)


def train(pairs: Sequence[tuple[str, str, int]], base: str, output: Path, *, epochs: int = 1, lr: float = 1e-4,
          batch_size: int = 16, rank: int = 16, device: str = "cuda", log: Callable[[dict], None] = print) -> Path:
    """Pointwise BCE on the yes-no logit with LoRA on attention and MLP projections."""
    import random

    import torch
    from peft import LoraConfig, get_peft_model

    scorer = Reranker(base, device=device, batch_size=batch_size)
    model = get_peft_model(scorer.model.train(), LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    scorer.model = model
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    order = list(pairs)
    step = 0
    for epoch in range(epochs):
        random.Random(epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            logits = scorer.logits([pair_text(q, d) for q, d, _ in chunk]).float()
            labels = torch.tensor([float(y) for _, _, y in chunk], device=logits.device)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 20 == 0:
                log({"epoch": epoch, "step": step, "loss": float(loss)})
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    return output
