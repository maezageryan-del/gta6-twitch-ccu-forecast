[![make.fun gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/maezageryan-del/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/maezageryan-del/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Launch-Day Twitch Viewers — Two-Model Ensemble Forecaster

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), hour by hour, from everything
recorded up to the end of the day's first hour — as five quantiles per hour
— for make.fun's `gta6-twitch-ccu-forecast-01` ($250, top 20 paid).

V2 of the comparable-shape forecaster (V1 scored 0.2220 on the practice
board). A leave-one-out diagnosis of V1 showed its loss lives in the
median, not the spread: each launch's day shape deviates from any single
regression line idiosyncratically. V2 therefore ensembles two deliberately
different models and averages their quantiles in log space — the one
change that never hurt in testing.

## The models

Shared idea: **past big launches tell us how an audience grows after the
first hour, and how that growth depends on how big the first hour was.**
For each hour `h = 2..24`, both models regress the log ratio
`ln(viewers_h / viewers_1)` on `ln(viewers_1)` and a console-only flag
across the comparable launches in the case, then read the target launch
off that line. Both keep **same-kind launches only**: a comparable counts
only when its own day curve never rises far above its hour 1 (max ratio
≤ 10x), the rolling-midnight console family — a simultaneous global
unlock has almost nobody in hour 1 and explodes later, and mixing the two
families poisons any regression. Both extrapolate: a regression slope
keeps going past the biggest launch ever seen, where a nearest neighbour
would clamp.

- **Model A** (V1's model): weighted least squares, each hour's slope
  shrunk toward the pooled slope with empirical-Bayes weights, the
  console coefficient shrunk 20%, log-normal quantiles from the residual
  SD, widened by the regression's leverage term.
- **Model B** (new): the same regression fitted **Huber-robustly**, so one
  freak launch in the donor pool cannot drag an hour's line; a fixed 30%
  slope blend toward the pooled slope; and **empirical residual
  quantiles**, pooled over neighbouring hours, instead of a normal
  spread — asymmetric bands that follow how launches actually miss their
  line (they overshoot more violently than they undershoot).

Standard library only, deterministic, ~0.08 s per case.

## Validation (leave-one-out over the comparables)

The kit ships no practice answers (the scorer is server-side), so the
models were validated by treating each comparable launch as a pseudo-case:
its hour-1 print is the only input, its real hours 2–24 are the answer.

| Test | V1 loss | V2 loss | Typical miss | 90% coverage |
| --- | --- | --- | --- | --- |
| Console-only targets (n=8, the family GTA VI belongs to) | 0.165 | **0.163** | 31% | 0.97 |
| All rolling-midnight targets (n=27) | 0.212 | 0.212 | 42% | 0.86 |
| Extrapolation: fit on hour-1 < 200k, test on the 6 biggest | 0.139 | **0.135** | 22% | 1.00 |

For reference: the house baseline scores ~0.387 on the practice cases;
the practice leaderboard's top entries sit at ≈0.122–0.136 and the
top-20 cutoff at ≈0.15.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `*.out` per case:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}`, hours 2–24.
