[![make.fun gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/maezageryan-del/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/maezageryan-del/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Launch-Day Twitch Viewers — Two-Model Ensemble Forecaster

What will Grand Theft Auto VI's Twitch audience look like, hour by hour,
on release day (2026-11-19 UTC)? This entry answers with five quantiles
per hour, predicted from nothing beyond the first hour's viewer print,
for make.fun's `gta6-twitch-ccu-forecast-01` ($250, top 20 paid).

It is the second version of a forecaster built on comparable launches.
Version 1 — a single regression per hour — reached 0.2220 on the practice
board. Testing it leave-one-out style, with each comparable launch in
turn hidden and predicted from the others, showed the remaining error
was concentrated in the median rather than the band width: individual
launches simply trace idiosyncratic day shapes that no single fitted
line follows. Version 2 responds by forecasting with two differently
built models and averaging their quantiles in log space — in testing,
the one change that never made the score worse.

## How it works

The premise both models share: **earlier launches reveal how an audience
grows after hour 1, and that growth pattern shifts with the size of
hour 1.** Concretely, for every hour `h = 2..24` the models regress
`ln(viewers_h / viewers_1)` on `ln(viewers_1)` plus a console-only flag
over the case's comparable launches, then evaluate the fitted line at
the target launch. Two design choices matter:

- **Only like-for-like donors.** A comparable is used only if its own
  day curve stays within 10x of its hour 1 — the signature of a
  rolling-midnight console launch, whose audience is there from the
  start. A simultaneous global unlock looks completely different (near
  zero in hour 1, then an explosion) and would poison the regression if
  mixed in.
- **Lines, not neighbours.** Because the forecast comes from a
  regression slope, it extrapolates past the largest launch in the
  donor pool instead of clamping to it the way a nearest-neighbour
  method would.

The two models then diverge on purpose:

- **Model A** is version 1's fit, unchanged: weighted least squares,
  each hour's slope pulled toward the pooled slope with empirical-Bayes
  weights, the console coefficient shrunk by 20%, and log-normal
  quantile bands taken from the residual SD and widened by the
  regression's leverage when the target is far from the donors.
- **Model B** fits the same regression **Huber-robustly**, so one freak
  launch among the donors cannot drag an hour's line; blends every
  slope a fixed 30% toward the pooled slope; and builds its bands from
  the **empirical quantiles of its own residuals**, pooled across
  neighbouring hours. The resulting bands are asymmetric, matching how
  launches actually deviate — the overshoots are wilder than the
  undershoots.

The published quantiles are the geometric mean of the two models',
hour by hour and level by level. Pure standard library, fully
deterministic, roughly 0.08 s per case.

## Validation (leave-one-out over the comparables)

The kit includes no practice answers (scoring is server-side), so the
models were graded by replaying history: hide one comparable launch,
predict its day from its hour-1 print alone, and score against its real
hours 2–24.

| Test | V1 loss | V2 loss | Typical miss | 90% coverage |
| --- | --- | --- | --- | --- |
| Console-only targets (n=8, the family GTA VI belongs to) | 0.165 | **0.163** | 31% | 0.97 |
| All rolling-midnight targets (n=27) | 0.212 | 0.212 | 42% | 0.86 |
| Extrapolation: fit on hour-1 < 200k, test on the 6 biggest | 0.139 | **0.135** | 22% | 1.00 |

Context: the house baseline scores ≈0.387 on the practice cases, the
practice leaderboard's leaders sit at ≈0.122–0.136, and its top-20
cutoff is ≈0.15.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Every `*.in` in the input directory produces one `*.out`:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}` for hours 2–24.
