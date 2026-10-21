"""The math of the controller: the Riccati equation, the LQR gain, the Kalman filter, the reference path."""
import math
import random
import unittest

import common  # noqa: F401  (sets sys.path)
from lqg_nav.lqg import (KalmanFilter, LqrSchedule, add, dare, diag, eye, inv, lqr_gain, model, mul, sub,
                         transpose, wrap)
from lqg_nav.path import RefPath


def eig2(m):
    """Eigenvalues of a 2x2 matrix (complex allowed)."""
    tr, det = m[0][0] + m[1][1], m[0][0] * m[1][1] - m[0][1] * m[1][0]
    disc = complex(tr * tr / 4 - det) ** 0.5
    return tr / 2 + disc, tr / 2 - disc


class MatrixTests(unittest.TestCase):
    def test_helpers(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(mul(a, eye(2)), a)
        self.assertEqual(transpose(a), [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(add(a, a), [[2.0, 4.0], [6.0, 8.0]])
        self.assertEqual(sub(a, a), [[0.0, 0.0], [0.0, 0.0]])
        product = mul(a, inv(a))
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0)
        self.assertEqual(inv([[4.0]]), [[0.25]])
        with self.assertRaises(ValueError):
            inv([[1.0, 2.0], [2.0, 4.0]])

    def test_wrap(self):
        self.assertAlmostEqual(wrap(3 * math.pi / 2), -math.pi / 2)
        self.assertAlmostEqual(wrap(-3 * math.pi / 2), math.pi / 2)
        self.assertAlmostEqual(wrap(0.3), 0.3)

    def test_model_is_exact_for_a_constant_turn(self):
        """x[k+1] = A x + B u must match the integration of e_y' = v e_theta, e_theta' = u."""
        v, dt, u = 0.2, 0.1, 0.5
        A, B = model(v, dt)
        x = [[0.01], [0.05]]
        nxt = add(mul(A, x), mul(B, [[u]]))
        ey, et, h = 0.01, 0.05, dt / 10000
        for _ in range(10000):
            ey, et = ey + v * et * h, et + u * h
        self.assertAlmostEqual(nxt[0][0], ey, places=6)
        self.assertAlmostEqual(nxt[1][0], et, places=9)


class LqrTests(unittest.TestCase):
    Q, R = diag(100.0, 6.0), [[1.0]]

    def test_riccati_solution_satisfies_the_equation(self):
        for v in (0.05, 0.1, 0.2):
            A, B = model(v, 0.1)
            P = dare(A, B, self.Q, self.R)
            At, Bt = transpose(A), transpose(B)
            K = mul(inv(add(self.R, mul(mul(Bt, P), B))), mul(mul(Bt, P), A))
            right = add(self.Q, sub(mul(mul(At, P), A), mul(mul(mul(At, P), B), K)))
            for i in range(2):
                for j in range(2):
                    self.assertAlmostEqual(P[i][j], right[i][j], delta=1e-6 * (1 + abs(P[i][j])))

    def test_closed_loop_is_stable(self):
        for v in (0.05, 0.12, 0.22):
            A, B = model(v, 0.1)
            K = lqr_gain(A, B, self.Q, self.R)
            closed = sub(A, mul(B, K))
            for lam in eig2(closed):
                self.assertLess(abs(lam), 1.0, f"v={v}")

    def test_known_gain(self):
        """Checked once with scipy.linalg.solve_discrete_are (v = 0.2, Q = diag(100, 6.25), R = 1)."""
        A, B = model(0.2, 0.1)
        K = lqr_gain(A, B, diag(100.0, 6.25), [[1.0]])
        self.assertAlmostEqual(K[0][0], 8.524, places=3)
        self.assertAlmostEqual(K[0][1], 2.820, places=3)

    def test_against_scipy_when_it_is_there(self):
        try:
            import numpy as np
            import scipy.linalg as sl
        except ImportError:
            self.skipTest("scipy is not installed")
        for v in (0.05, 0.2):
            A, B = model(v, 0.1)
            P = sl.solve_discrete_are(np.array(A), np.array(B), np.array(self.Q), np.array(self.R))
            mine = dare(A, B, self.Q, self.R)
            self.assertTrue(np.allclose(P, np.array(mine), rtol=1e-6))

    def test_schedule_caches_and_uses_a_minimum_speed(self):
        s = LqrSchedule(100.0, 6.0, 1.0, 0.1, v_min=0.05)
        self.assertEqual(s.gain(0.0), s.gain(0.05))          # a robot that stands still uses v_min
        self.assertEqual(s.gain(0.2), s.gain(0.2001))
        self.assertGreater(s.gain(0.2)[0], 0)
        self.assertGreater(s.gain(0.2)[1], 0)
        self.assertEqual(len(s._cache), 2)

    def test_more_weight_on_the_error_gives_a_stronger_gain(self):
        A, B = model(0.2, 0.1)
        soft = lqr_gain(A, B, diag(10.0, 6.0), [[1.0]])
        hard = lqr_gain(A, B, diag(1000.0, 6.0), [[1.0]])
        self.assertGreater(hard[0][0], soft[0][0])


class KalmanTests(unittest.TestCase):
    def test_noisy_measurements_are_smoothed(self):
        rng = random.Random(1)
        kf = KalmanFilter(0.001, 0.002, 0.02, 0.03)
        A, B = model(0.2, 0.1)
        true = (0.05, 0.0)
        kf.reset((true[0] + rng.gauss(0, 0.02), true[1] + rng.gauss(0, 0.03)))
        errors_raw, errors_kf = [], []
        for _ in range(300):
            z = (true[0] + rng.gauss(0, 0.02), true[1] + rng.gauss(0, 0.03))
            kf.predict(A, B, 0.0)
            kf.update(z)
            errors_raw.append((z[0] - true[0]) ** 2)
            errors_kf.append((kf.estimate()[0] - true[0]) ** 2)
        self.assertLess(sum(errors_kf[50:]), 0.3 * sum(errors_raw[50:]))

    def test_gain_and_covariance(self):
        kf = KalmanFilter(0.003, 0.01, 0.02, 0.03)
        kf.reset((0.0, 0.0))
        self.assertTrue(kf.ready)
        A, B = model(0.2, 0.1)
        for _ in range(200):
            kf.predict(A, B, 0.0)
            kf.update((0.0, 0.0))
        K = kf.last_gain
        for i in range(2):
            self.assertTrue(0.0 < K[i][i] < 1.0)                    # partly the model, partly the measurement
            self.assertLess(kf.P[i][i], kf.V[i][i])                 # better than one measurement alone

    def test_the_prediction_follows_the_model(self):
        kf = KalmanFilter(0.003, 0.01, 0.02, 0.03)
        kf.reset((0.0, 0.1))
        A, B = model(0.2, 0.1)
        kf.predict(A, B, 0.5)
        ey, et = kf.estimate()
        self.assertAlmostEqual(ey, 0.2 * 0.1 * 0.1 + 0.5 * 0.2 * 0.01 * 0.5)
        self.assertAlmostEqual(et, 0.1 + 0.05)


class PathTests(unittest.TestCase):
    def test_straight_line(self):
        p = RefPath([(0, 0), (1, 0), (2, 0)], ds=0.05)
        self.assertAlmostEqual(p.length, 2.0)
        self.assertEqual(p.end, (2.0, 0.0))
        self.assertEqual((p.xs[0], p.ys[0]), (0.0, 0.0))
        self.assertTrue(all(abs(h) < 1e-9 for h in p.headings))
        self.assertTrue(all(abs(k) < 1e-9 for k in p.curvatures))
        steps = [math.hypot(p.xs[i + 1] - p.xs[i], p.ys[i + 1] - p.ys[i]) for i in range(len(p) - 1)]
        self.assertTrue(all(abs(s - 0.05) < 1e-9 for s in steps))

    def test_circle_curvature(self):
        r = 1.0
        pts = [(r * math.cos(a / 100), r * math.sin(a / 100)) for a in range(0, 158)]
        p = RefPath(pts, ds=0.05, smooth=0)
        middle = p.curvatures[5:-5]
        self.assertAlmostEqual(sum(middle) / len(middle), 1 / r, delta=0.03)
        self.assertGreater(p.max_curvature(0, 0.4), 0.9)

    def test_errors_sign(self):
        p = RefPath([(0, 0), (2, 0)])
        i = p.nearest(1.0, 0.1)
        e_y, e_th = p.errors(i, 1.0, 0.1, 0.2)
        self.assertAlmostEqual(e_y, 0.1)                 # left of the path: positive
        self.assertAlmostEqual(e_th, 0.2)
        e_y, e_th = p.errors(i, 1.0, -0.1, 2 * math.pi - 0.2)
        self.assertAlmostEqual(e_y, -0.1)
        self.assertAlmostEqual(e_th, -0.2)

    def test_nearest_looks_near_the_hint(self):
        """A path that comes back near itself: the robot must not jump to the other part."""
        pts = [(x / 10, 0.0) for x in range(21)] + [(2.0, 0.1)] + [(2 - x / 10, 0.2) for x in range(21)]
        p = RefPath(pts, ds=0.05, smooth=0)
        i = p.nearest(1.0, 0.05, start=20)
        self.assertLess(p.xs[i] - 1.0, 0.1)
        self.assertLess(abs(p.ys[i]), 0.06)

    def test_remaining_and_ahead(self):
        p = RefPath([(0, 0), (1, 0)])
        self.assertAlmostEqual(p.remaining(0), 1.0)
        self.assertEqual(p.remaining(len(p) - 1), 0.0)
        self.assertAlmostEqual(p.ahead(0, 0.3)[0], 0.3)
        self.assertEqual(p.ahead(0, 5.0), p.end)

    def test_repeated_points_and_a_single_point(self):
        p = RefPath([(0, 0), (0, 0), (1, 0), (1, 0)])
        self.assertAlmostEqual(p.length, 1.0)
        one = RefPath([(0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(len(one), 1)
        self.assertEqual(one.length, 0.0)
        with self.assertRaises(ValueError):
            RefPath([])

    def test_smoothing_keeps_the_ends(self):
        zigzag = [(0.05 * k, 0.02 * (k % 2)) for k in range(40)]
        p = RefPath(zigzag)
        self.assertEqual((p.xs[0], p.ys[0]), zigzag[0])
        self.assertEqual(p.end, zigzag[-1])
        self.assertLess(max(abs(y - 0.01) for y in p.ys[5:-5]), 0.006)    # the steps are smoothed away


if __name__ == "__main__":
    unittest.main()
