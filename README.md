[![make.fun gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/maezageryan-del/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/maezageryan-del/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Launch-Day Twitch Viewers — Comparable-Shape Forecaster

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), hour by hour, from everything
recorded up to the end of the day's first hour — as five quantiles per hour
— for make.fun's `gta6-twitch-ccu-forecast-01` ($250, top 20 paid).

## The model

One idea, carried all the way through: **past big launches tell us how an
audience grows after the first hour, and how that growth depends on how big
the first hour was.** For each hour `h = 2..24`, the program regresses the
log ratio `ln(viewers_h / viewers_1)` on `ln(viewers_1)` across the
comparable launches in the case, then reads the target launch off that line.

Three things make it work where a nearest-neighbour or a flat median fails:

1. **Same-kind launches only.** A launch's shape is set by how it unlocks.
   A rolling-midnight console launch already has a real audience in hour 1
   and stays within a few multiples of it all day; a simultaneous global
   unlock (often a PC launch) has almost nobody in hour 1 and explodes
   10x–1000x later. Mixing the two families poisons any regression, so a
   comparable is kept only when its own day curve never rises far above its
   hour 1 (max ratio ≤ 10x). This also drops the GTA VI category's own
   pre-release history, whose "day" is not a launch day.
2. **A console-only effect.** Console-only launches ramp much harder into
   the evening (about 2.4x hour 1 by hour 20, against about 1.5x for
   launches with a PC audience). The regression carries a console-only
   indicator, read from the case itself (its notes, or the target game's
   own comparable row), shrunk 20% toward zero because only a handful of
   comparables are console-only.
3. **It extrapolates.** GTA VI will very likely open far bigger than any
   comparable, so the forecast has to keep going past the edge of the data.
   A regression slope does that; a nearest neighbour would clamp the answer
   to the biggest launch ever seen. The slope is shrunk toward the pooled
   slope across hours, the shape is smoothed lightly across neighbouring
   hours, and the spread is the residual standard deviation, widened by
   the regression's own leverage term — so the further the target sits
   from the comparables' centre, the wider the honest band.

Quantiles are log-normal: `q_tau = p1 * exp(yhat_h + z_tau * sigma_h)`.
Standard library only, deterministic, ~0.07 s per case.

## Validation (leave-one-out over the comparables)

The kit ships no practice answers (the scorer is server-side), so the model
was validated by treating each comparable launch as a pseudo-case: its
hour-1 print is the only input, its real hours 2–24 are the answer.

| Test | Pinball loss | Typical miss | 90% coverage |
| --- | --- | --- | --- |
| Console-only targets (n=8, the family GTA VI belongs to) | **0.165** | 31% | 0.95 |
| All rolling-midnight targets (n=27) | 0.212 | 42% | 0.87 |
| Extrapolation: fit on hour-1 < 200k, test on the 6 biggest launches | **0.139** | 22% | 1.00 |

For reference: the house baseline scores ~0.387 on the practice cases, the
practice leaderboard's #1 sits at 0.169, and the top-20 cutoff is 0.240.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `*.out` per case:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}`, hours 2–24.
