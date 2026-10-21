"""Productions and conflict resolution.

A production is an IF-THEN rule: IF the buffers look like this THEN do that.
In every cycle we test all rules, and only ONE of the matching rules fires.
If several rules match, the one with the highest utility wins (plus a little
noise if noise > 0). After an action we can learn:  U <- U + alpha * (R - U).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence


@dataclass
class Production:
    name: str
    utility: float
    condition: Callable[[Any], bool]    # LHS: looks at the buffers, changes nothing
    action: Callable[[Any], None]       # RHS: changes buffers / asks a module
    group: str = ""                     # rules of one group compete with each other
    quiet: bool = False                 # do not print every firing (used for the monitor)
    doc: str = ""
    fired: int = 0
    initial_utility: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.initial_utility = self.utility


def update_utility(prod: Production, reward: float, alpha: float = 0.2) -> float:
    """U <- U + alpha * (R - U)"""
    prod.utility += alpha * (reward - prod.utility)
    return prod.utility


def logistic_noise(rng: random.Random, s: float) -> float:
    if s <= 0:
        return 0.0
    u = min(max(rng.random(), 1e-9), 1 - 1e-9)
    return s * math.log(u / (1 - u))


def choose_production(ctx: Any, productions: Sequence[Production], noise: float = 0.0,
                      rng: Optional[random.Random] = None) -> Optional[Production]:
    """Highest utility among the productions whose condition is true."""
    rng = rng or random.Random(0)
    best, best_score = None, None
    for p in productions:
        if p.condition(ctx):
            score = p.utility + logistic_noise(rng, noise)
            if best_score is None or score > best_score:
                best, best_score = p, score
    return best


def choose_softmax(matched: Sequence[Production], tau: float = 0.5,
                   rng: Optional[random.Random] = None) -> Optional[Production]:
    """P(p_i) = exp(U_i / tau) / sum_j exp(U_j / tau). Small tau is close to 'take the maximum'."""
    if not matched:
        return None
    rng = rng or random.Random(0)
    top = max(p.utility for p in matched)
    weights = [math.exp((p.utility - top) / tau) for p in matched]
    r = rng.random() * sum(weights)
    acc = 0.0
    for p, w in zip(matched, weights):
        acc += w
        if r <= acc:
            return p
    return matched[-1]


class ProductionSystem:
    """Holds the rules and runs one production cycle."""

    def __init__(self, productions: Iterable[Production], noise: float = 0.0,
                 rng: Optional[random.Random] = None):
        self.productions: List[Production] = list(productions)
        self.noise = noise
        self.rng = rng or random.Random(0)
        self.cycles = 0
        self.idle_cycles = 0

    def by_name(self, name: str) -> Production:
        for p in self.productions:
            if p.name == name:
                return p
        raise KeyError(name)

    def cycle(self, ctx: Any) -> Optional[Production]:
        """Test all rules, fire the winner. Returns it (or None if nothing matched)."""
        self.cycles += 1
        chosen = choose_production(ctx, self.productions, self.noise, self.rng)
        if chosen is None:
            self.idle_cycles += 1
            return None
        self.idle_cycles = 0
        chosen.fired += 1
        chosen.action(ctx)
        return chosen
