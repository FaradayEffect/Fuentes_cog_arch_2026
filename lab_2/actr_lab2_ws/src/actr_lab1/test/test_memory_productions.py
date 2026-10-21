import math
import random
import unittest

import common  # noqa: F401
from actr_lab1.core.chunks import Chunk
from actr_lab1.core.memory import DeclarativeMemory
from actr_lab1.core.productions import (Production, ProductionSystem, choose_production,
                                        choose_softmax, update_utility)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.m = DeclarativeMemory(self.clock)

    def test_retrieval_returns_matching_chunk(self):
        self.m.add(Chunk("episode", {"task": "pick", "result": "failure"}))
        hit = self.m.retrieve({"kind": "episode", "task": "pick"})
        self.assertIsNotNone(hit)
        self.assertEqual(hit.get("result"), "failure")
        self.assertIsNone(self.m.retrieve({"kind": "episode", "task": "navigate"}))
        self.assertEqual(self.m.failures, 1)

    def test_activation_falls_with_time(self):
        e = self.m.add(Chunk("a", {}))
        a0 = self.m.activation(e, now=1.0)
        a1 = self.m.activation(e, now=100.0)
        self.assertGreater(a0, a1)

    def test_activation_grows_with_practice(self):
        e1 = self.m.add(Chunk("a", {}))
        e2 = self.m.add(Chunk("b", {}))
        e2.times.extend([5.0, 6.0, 7.0])
        self.assertGreater(self.m.activation(e2, now=10.0), self.m.activation(e1, now=10.0))

    def test_formula(self):
        e = self.m.add(Chunk("a", {}))
        e.times = [0.0, 10.0]
        expected = math.log(20.0 ** -0.5 + 10.0 ** -0.5)
        self.assertAlmostEqual(self.m.activation(e, now=20.0), expected)

    def test_old_chunk_is_forgotten_but_permanent_is_not(self):
        self.m.add(Chunk("episode", {"id": 1}))
        self.m.add(Chunk("fact", {"id": 2}), permanent=True)
        self.clock.t = 100000.0
        self.assertIsNone(self.m.retrieve({"kind": "episode"}))
        self.assertIsNotNone(self.m.retrieve({"kind": "fact"}))

    def test_most_recent_wins_for_equal_use(self):
        self.m.add(Chunk("episode", {"n": 1}))
        self.clock.t = 50.0
        self.m.add(Chunk("episode", {"n": 2}))
        self.clock.t = 60.0
        self.assertEqual(self.m.retrieve({"kind": "episode"}).get("n"), 2)

    def test_retrieval_strengthens_the_chunk(self):
        e = self.m.add(Chunk("episode", {}))
        self.clock.t = 10.0
        n = len(e.times)
        self.m.retrieve({"kind": "episode"})
        self.assertEqual(len(e.times), n + 1)

    def test_retrieve_all_has_no_side_effect_and_is_sorted(self):
        e1 = self.m.add(Chunk("x", {"n": 1}))
        self.clock.t = 5.0
        e2 = self.m.add(Chunk("x", {"n": 2}))
        self.clock.t = 6.0
        found = self.m.retrieve_all({"kind": "x"})
        self.assertEqual([c.get("n") for c in found], [2, 1])
        self.assertEqual(len(e1.times), 1)
        self.assertEqual(len(e2.times), 1)

    def test_callable_pattern(self):
        self.m.add(Chunk("x", {"v": 3}))
        self.assertIsNotNone(self.m.retrieve(lambda c: c.get("v") > 2))
        self.assertIsNone(self.m.retrieve(lambda c: c.get("v") > 5))

    def test_noise_is_reproducible_with_a_seed(self):
        m1 = DeclarativeMemory(self.clock, noise=0.3, rng=random.Random(4))
        m2 = DeclarativeMemory(self.clock, noise=0.3, rng=random.Random(4))
        e1, e2 = m1.add(Chunk("a", {})), m2.add(Chunk("a", {}))
        self.assertEqual(m1.activation(e1, 10.0), m2.activation(e2, 10.0))


class ProductionTests(unittest.TestCase):
    def test_high_utility_safety_rule_wins(self):      # like the quick test of Seminar 2
        state = {"distance": 0.2, "action": None}
        prods = [
            Production("continue", 1.0, lambda s: True, lambda s: s.update(action="MOVE")),
            Production("stop", 100.0, lambda s: s["distance"] < 0.3, lambda s: s.update(action="STOP")),
        ]
        chosen = choose_production(state, prods)
        chosen.action(state)
        self.assertEqual(chosen.name, "stop")
        self.assertEqual(state["action"], "STOP")

    def test_low_utility_rule_runs_when_the_other_does_not_match(self):
        state = {"distance": 2.0}
        prods = [Production("continue", 1.0, lambda s: True, lambda s: None),
                 Production("stop", 100.0, lambda s: s["distance"] < 0.3, lambda s: None)]
        self.assertEqual(choose_production(state, prods).name, "continue")

    def test_no_match_gives_none(self):
        self.assertIsNone(choose_production({}, [Production("p", 1.0, lambda s: False, lambda s: None)]))

    def test_utility_learning_formula(self):             # TEST 6 of Seminar 2
        p = Production("demo", 0.0, lambda s: True, lambda s: None)
        update_utility(p, reward=1.0, alpha=0.2)
        self.assertTrue(math.isclose(p.utility, 0.2))
        update_utility(p, reward=-1.0, alpha=0.2)
        self.assertTrue(math.isclose(p.utility, 0.2 + 0.2 * (-1.0 - 0.2)))

    def test_utility_moves_towards_the_reward(self):
        p = Production("demo", 0.0, lambda s: True, lambda s: None)
        for _ in range(60):
            update_utility(p, 1.0, 0.2)
        self.assertAlmostEqual(p.utility, 1.0, places=3)

    def test_only_one_rule_fires_per_cycle(self):
        log = []
        prods = [Production("a", 2.0, lambda s: True, lambda s: log.append("a")),
                 Production("b", 1.0, lambda s: True, lambda s: log.append("b"))]
        ps = ProductionSystem(prods)
        ps.cycle(None)
        self.assertEqual(log, ["a"])
        self.assertEqual(prods[0].fired, 1)
        self.assertEqual(prods[1].fired, 0)

    def test_idle_cycles_are_counted(self):
        ps = ProductionSystem([Production("a", 1.0, lambda s: False, lambda s: None)])
        self.assertIsNone(ps.cycle(None))
        self.assertIsNone(ps.cycle(None))
        self.assertEqual(ps.idle_cycles, 2)

    def test_noise_changes_the_choice_sometimes(self):
        prods = [Production("a", 1.0, lambda s: True, lambda s: None),
                 Production("b", 0.9, lambda s: True, lambda s: None)]
        rng = random.Random(1)
        names = {choose_production(None, prods, noise=0.5, rng=rng).name for _ in range(200)}
        self.assertEqual(names, {"a", "b"})
        rng = random.Random(1)
        names = {choose_production(None, prods, noise=0.0, rng=rng).name for _ in range(50)}
        self.assertEqual(names, {"a"})

    def test_softmax_temperature(self):                  # Task C of Seminar 2
        prods = [Production("hi", 1.0, lambda s: True, lambda s: None),
                 Production("lo", 0.0, lambda s: True, lambda s: None)]
        def share_hi(tau):
            rng = random.Random(7)
            return sum(choose_softmax(prods, tau, rng).name == "hi" for _ in range(2000)) / 2000
        cold, warm, hot = share_hi(0.05), share_hi(0.5), share_hi(50.0)
        self.assertGreater(cold, 0.99)
        self.assertTrue(0.75 < warm < 0.95)              # exp(2)/(1+exp(2)) = 0.88
        self.assertTrue(0.45 < hot < 0.56)
        self.assertIsNone(choose_softmax([], 0.5))


if __name__ == "__main__":
    unittest.main()
