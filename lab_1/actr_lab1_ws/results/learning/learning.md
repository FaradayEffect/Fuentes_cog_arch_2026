# A. The first waypoint of room A is blocked for ever

| episode | retries (learning on) | time, s (learning on) | retries (learning off) | time, s (learning off) | U(RETRY) after (learning on) | U(ALT) after (learning on) |
|---|---|---|---|---|---|---|
| 1 | 1 | 68 | 2 | 70 | 0.28 | 0.60 |
| 2 | 0 | 66 | 2 | 70 | 0.28 | 0.68 |
| 3 | 0 | 66 | 2 | 70 | 0.28 | 0.74 |
| 4 | 0 | 66 | 2 | 70 | 0.28 | 0.80 |
| 5 | 0 | 66 | 2 | 70 | 0.28 | 0.84 |
| 6 | 0 | 66 | 2 | 70 | 0.28 | 0.87 |
| 7 | 0 | 66 | 2 | 70 | 0.28 | 0.90 |
| 8 | 0 | 66 | 2 | 70 | 0.28 | 0.92 |
| 9 | 0 | 66 | 2 | 70 | 0.28 | 0.93 |
| 10 | 0 | 66 | 2 | 70 | 0.28 | 0.95 |
| 11 | 0 | 66 | 2 | 70 | 0.28 | 0.96 |
| 12 | 0 | 66 | 2 | 70 | 0.28 | 0.97 |
| 13 | 0 | 66 | 2 | 70 | 0.28 | 0.97 |
| 14 | 0 | 66 | 2 | 70 | 0.28 | 0.98 |
| 15 | 0 | 66 | 2 | 70 | 0.28 | 0.98 |
| **mean** | 0.07 | 66 | 2.00 | 70 | | |

# B. The mission asks for the restricted zone

| condition | episodes | robot entered the zone | mission done | mean refusals by the gate |
|---|---|---|---|---|
| gate OFF, learning on | 10 | 10 | 10 | 0.0 |
| gate ON, learning on | 10 | 0 | 10 | 1.0 |
