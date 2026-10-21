# Pure pursuit vs LQR vs LQG (simulated robot, not Gazebo)

6 routes on the map of the world x 10 seeds x 3 noise levels. Lateral error = distance of the TRUE robot position to the reference path. Turn jerk = how fast the turn command changes (smaller = smoother).

| pose noise | controller | success | lateral error RMS, cm | lateral error max, cm | time, s | final error, cm | turn jerk RMS, rad/s^2 |
|---|---|---|---|---|---|---|---|
| none (0 cm, 0.0 deg) | pure_pursuit | 60/60 | 0.6 | 1.9 | 46.6 | 2.3 | 0.44 |
| none (0 cm, 0.0 deg) | lqr | 60/60 | 0.2 | 0.8 | 46.7 | 2.3 | 0.76 |
| none (0 cm, 0.0 deg) | lqg | 60/60 | 0.3 | 1.0 | 46.7 | 2.3 | 0.48 |
| amcl (1 cm, 1.1 deg) | pure_pursuit | 60/60 | 0.7 | 2.0 | 46.5 | 2.8 | 0.93 |
| amcl (1 cm, 1.1 deg) | lqr | 60/60 | 0.3 | 1.0 | 46.6 | 2.7 | 1.53 |
| amcl (1 cm, 1.1 deg) | lqg | 60/60 | 0.3 | 1.1 | 46.6 | 2.7 | 0.55 |
| high (3 cm, 2.9 deg) | pure_pursuit | 60/60 | 0.9 | 2.4 | 46.1 | 4.5 | 1.78 |
| high (3 cm, 2.9 deg) | lqr | 60/60 | 0.8 | 2.2 | 46.2 | 4.6 | 2.28 |
| high (3 cm, 2.9 deg) | lqg | 60/60 | 0.7 | 1.7 | 46.1 | 4.5 | 0.88 |

## Whole missions with the agent of Lab 1

6 missions x 4 seeds, the ACT-R agent of Lab 1 unchanged (mock planner), the LQG driving the simulated robot, pose noise 2 cm and 0.03 rad: 24/24 missions succeeded, 0 unsafe, 0 failed navigate steps, 0 samples closer than 10 cm to a wall.
