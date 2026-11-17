"""Which optional views a node carries, decided by value against cost.

Build time applies cheap rules; answer time upgrades nodes that tasks keep using. Every
decision is recorded so an ablation can switch a view family off without changing the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from .core import MemoryNode

HUB_KINDS = {"session", "entity", "timeline"}


@dataclass(frozen=True)
class ViewConfig:
    """Pre-registered switches and thresholds for one experimental arm."""

    t1: bool = False
    v1: bool = True
    v2: bool = False
    v2_coverage: float = 1.0
    t1_min_tokens: int = 512
    v1_min_tokens: int = 160
    v2_unit_budget: int = 48
    upgrade_after_uses: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.v2_coverage <= 1.0 or self.upgrade_after_uses < 1 or self.v2_unit_budget < 0:
            raise ValueError("invalid view configuration")


@dataclass
class ViewLedger:
    """Mutable per-unit bookkeeping: node usage counts and the generated-image budget."""

    uses: dict[str, int] = field(default_factory=dict)
    v2_generated: int = 0
    decisions: list[dict] = field(default_factory=list)

    def use(self, node_id: str) -> int:
        self.uses[node_id] = self.uses.get(node_id, 0) + 1
        return self.uses[node_id]


def node_kind(node: MemoryNode) -> str:
    return str(node.canvas.get("type", ""))


def keeps_v2(node_id: str, coverage: float) -> bool:
    """Deterministic subsample for the image-density dose-response analysis."""
    if coverage >= 1.0:
        return True
    if coverage <= 0.0:
        return False
    bucket = int(hashlib.sha256(node_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < coverage


class MaterializationPolicy:
    def __init__(self, config: ViewConfig) -> None:
        self.config = config

    def wants_t1(self, node: MemoryNode, text_tokens: int, has_relative_time: bool) -> bool:
        if not self.config.t1 or node.view("T1") is not None:
            return False
        return (text_tokens > self.config.t1_min_tokens or has_relative_time
                or node_kind(node) in HUB_KINDS)

    def wants_v1(self, node: MemoryNode, text_tokens: int) -> bool:
        return self.config.v1 and node.view("V1") is None and text_tokens >= self.config.v1_min_tokens

    def wants_v2_at_build(self, node: MemoryNode, ledger: ViewLedger) -> bool:
        return (self.config.v2 and node_kind(node) in HUB_KINDS and node.view("V2") is None
                and ledger.v2_generated < self.config.v2_unit_budget
                and keeps_v2(node.id, self.config.v2_coverage))

    def upgrades(self, node: MemoryNode, ledger: ViewLedger) -> tuple[str, ...]:
        """Views to add now that a task has used this node; empty until the use threshold."""
        if ledger.uses.get(node.id, 0) < self.config.upgrade_after_uses or node_kind(node) == "note":
            return ()
        wanted = []
        if self.config.t1 and node.view("T1") is None:
            wanted.append("T1")
        if (self.config.v2 and node.view("V2") is None
                and ledger.v2_generated < self.config.v2_unit_budget
                and keeps_v2(node.id, self.config.v2_coverage)):
            wanted.append("V2")
        return tuple(wanted)
