"""GTA VI launch-day Twitch viewers, hour by hour: a comparable-shape forecaster.

The idea, in one sentence: past big launches tell us how an audience grows
after the first hour, and how that growth depends on how big the first hour
was -- so for each hour we regress the log ratio viewers_h / viewers_1 on
log viewers_1 across every comparable launch, then read the target launch
off that line.

Why this shape, and not the nearest neighbour: GTA VI will very likely open
far bigger than any comparable launch, so the forecast has to EXTRAPOLATE.
A nearest-neighbour or similarity-weighted average would clamp the answer
to the biggest launch ever seen; a regression slope keeps going. The slope
is the whole game: big launches grow less, proportionally, than small ones
(their first hour already holds the world), and the fitted line carries
that pattern past the edge of the data.

Method, per hour h = 2..24:
  0. Keep only SAME-KIND launches. A launch's shape is set by how it
     unlocks: a rolling-midnight console launch (New Zealand first, then
     each region at its own midnight) already has a real audience in hour 1
     and stays within a few multiples of it all day; a simultaneous global
     unlock (one worldwide hour, often a PC launch) has almost nobody in
     hour 1 and explodes 10x-1000x later. Mixing the two families poisons
     any regression, so a comparable is kept only when its own day curve
     never rises far above its hour 1 (max ratio over hours 2..24 <= 10).
     GTA VI -- and every practice case -- is a rolling-midnight console
     launch, so this is the family the forecast is read from.
  1. From each kept comparable, take its release-day hourly curve
     (hours 1..24 of its day_start) and form x = ln(viewers_1) and
     y_h = ln(viewers_h / viewers_1). A second regressor marks whether
     the launch was console-only (no PC): console-only launches ramp
     much harder into the evening (about 2.4x hour 1 by hour 20, against
     about 1.5x for launches with a PC audience), because their whole
     audience arrives through the same rolling midnight. The target's
     console-only status is read from the case itself (its notes, or the
     target game's own comparable row); GTA VI and every practice case
     are console-only. The console coefficient is shrunk 20% toward zero:
     only a handful of comparables are console-only, so the raw
     coefficient is noisier than the slope.
  2. Weighted least squares of y_h on (x - xbar) and the console flag.
     The slope beta_h is then shrunk toward the pooled slope across all
     hours (empirical-Bayes style: a single hour's slope is noisy; the
     hours agree with each other), and the intercept alpha_h is smoothed
     lightly across neighbouring hours, because a launch-day audience
     curve moves smoothly except where a regional unlock lands.
  3. The spread comes from the residuals: sigma_h is the residual standard
     deviation for that hour, smoothed across neighbouring hours, floored so
     no hour claims false precision, and widened by the regression's own
     leverage term -- the further the target's first hour sits from the
     comparables' centre, the wider the honest band. The whole spread is
     scaled by 0.9: the raw residual spread slightly over-covers.
  4. Quantiles are log-normal: q_tau = p1 * exp(yhat_h + z_tau * sigma_h).

What it uses: only the case's hour-1 print and the comparables' release-day
curves. The target's own pre-release history is not used: before launch the
category holds trailers and hype, not the launch audience, and it carries
no information about the day's shape that the comparables do not already
hold more cleanly.

Fallbacks keep every case valid: if a comparable lacks hour 1 or most of
hours 2..24 it is skipped (this also drops the GTA VI category's own
pre-release history, whose "day" is not a launch day); if too few
comparables survive for an hour, that hour falls back to the pooled
median shape; if nothing survives at all, the forecast is the house
persistence baseline. No randomness anywhere: two runs are byte-identical.

Standard library only. Reads every *.in in the input directory and writes
one *.out per case, exactly the format the rules ask for.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# --- tuning constants (chosen on leave-one-out over the comparables) ---
MIN_DAY_HOURS = 20      # a comparable must hold hour 1 and >= 20 of hours 2..24
MIN_POINTS_PER_HOUR = 5  # else fall back to the pooled shape for that hour
MAX_DAY_RATIO = 10.0    # rolling-midnight test: no day hour above 10x hour 1
BETA_PRIOR_SD = 0.15    # how far one hour's slope may stray from the pooled slope
SMOOTH_ALPHA = (0.2, 0.6, 0.2)   # neighbour smoothing of the shape, (prev, self, next)
SMOOTH_SIGMA = (0.25, 0.5, 0.25)
SIGMA_FLOOR = 0.20      # no hour claims a tighter band than this (log units)
SIGMA_CAP = 1.10        # and no hour is allowed to be wider than this
SIGMA_SCALE = 0.9       # residual spread slightly over-covers; scale it down
CC_SHRINK = 0.8         # shrink the console-only coefficient 20% toward zero
BASE_SIGMA = 0.15       # persistence fallback spread, per sqrt-hour (house baseline)
BASELINE_SIGMA = 0.15


def parse_ts(text):
    """ISO 8601 with a trailing Z into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, mirroring the house baseline's output."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Acklam's approximation to the standard normal quantile function."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def day_hours(comp):
    """A comparable's release-day hourly viewers, hour 1..24 -> viewers.

    Returns {} when the record is not a launch day at all (for example the
    GTA VI category's own pre-release history, whose day_start is the
    premiere date, not a launch day): it has no hour 1 and too few day
    hours, so the caller's completeness check drops it.
    """
    try:
        start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for row in comp.get("hourly") or []:
        try:
            t = parse_ts(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        h = int((t - start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out[h] = float(viewers)
    return out


def solve_linear(mat, vec):
    """Gaussian elimination for a tiny dense system (2x2 here)."""
    n = len(vec)
    m = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for cc in range(col, n + 1):
                    m[r][cc] -= f * m[col][cc]
    return [m[i][n] / m[i][i] for i in range(n)]


def wls_multi(cols, ys):
    """Least squares y ~ a + b1*(x1 - x1bar) + b2*(x2 - x2bar) ...

    cols is a list of regressor columns (all the same length as ys).
    Returns (a, coefs, sigma, se_b1, sxx1): the intercept at the columns'
    means, one coefficient per column, the residual SD, the slope's
    standard error and centred sum of squares (for the first column only,
    which is the log-size slope the shrinkage needs).
    """
    n = len(ys)
    k = len(cols)
    bars = [sum(c) / n for c in cols]
    ybar = sum(ys) / n
    mat = [[0.0] * k for _ in range(k)]
    vec = [0.0] * k
    for t in range(n):
        d = [cols[j][t] - bars[j] for j in range(k)]
        for i in range(k):
            vec[i] += d[i] * (ys[t] - ybar)
            for j in range(k):
                mat[i][j] += d[i] * d[j]
    coefs = solve_linear(mat, vec)
    if coefs is None:
        return None
    resid = [ys[t] - (ybar + sum(coefs[j] * (cols[j][t] - bars[j]) for j in range(k)))
             for t in range(n)]
    dof = max(n - k - 1, 1)
    s2 = sum(r * r for r in resid) / dof
    sigma = math.sqrt(max(s2, 0.0))
    # Standard error of the first slope, from the (0,0) entry of the
    # inverse of the centred cross-product matrix.
    inv00 = None
    if k == 1:
        inv00 = 1.0 / mat[0][0] if mat[0][0] > 0 else None
    elif k == 2:
        det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        inv00 = mat[1][1] / det if abs(det) > 1e-12 else None
    se_b = math.sqrt(s2 * inv00) if inv00 else float("inf")
    sxx = mat[0][0]
    return ybar, coefs, sigma, se_b, sxx, n


def smooth(values, weights):
    """Light neighbour smoothing over hours 2..24 (dict hour -> value)."""
    out = {}
    for h in range(2, 25):
        num = 0.0
        den = 0.0
        for dh, w in ((-1, weights[0]), (0, weights[1]), (1, weights[2])):
            if h + dh in values:
                num += w * values[h + dh]
                den += w
        out[h] = num / den if den else values[h]
    return out


def is_rolling(dh):
    """True when a launch's day curve looks like a rolling-midnight
    console launch: its audience is already there in hour 1 and no later
    hour runs far above it. A simultaneous global unlock has almost
    nobody in hour 1 and explodes later (ratios of 10x-1000x)."""
    if 1 not in dh:
        return False
    p1 = dh[1]
    return all(dh[h] / p1 <= MAX_DAY_RATIO for h in dh if h >= 2)


def is_console_only_platforms(platforms):
    """True when a launch's platform list holds no PC (and no Stadia):
    a console-only launch. Unknown or missing platforms count as not
    console-only -- the conservative side for the regression."""
    if not platforms:
        return False
    text = str(platforms)
    return "PC" not in text and "Stadia" not in text


def target_console_only(case):
    """Is the case's own game a console-only launch?

    First, the target game's own comparable row, when the case carries one
    (the final case does: the GTA VI category's pre-release history, with
    platforms "PS5, Xbox Series X|S"). Otherwise the case notes, which
    state the launch platforms ("released ... on PS5 only", "on PS4, Xbox
    One (console only; PC 2019-11)"). Failing both, True: the rules
    guarantee every case in this competition is a rolling-midnight
    console launch.
    """
    name = str((case.get("game") or {}).get("name") or "").strip().lower()
    for comp in case.get("comparables") or []:
        cname = str(comp.get("name") or "").strip().lower()
        if name and (cname.startswith(name) or name in cname):
            return is_console_only_platforms(comp.get("platforms"))
    notes = str(case.get("notes") or "")
    if notes:
        import re
        m = re.search(r"released\s+\S+\s+on\s+([^;.\n]+)", notes)
        if m:
            clause = re.sub(r"\([^)]*\)", "", m.group(1))
            return "PC" not in clause and "Stadia" not in clause
    return True


def fit_model(comparables, x0, z0):
    """Fit the per-hour log-ratio model. Returns {hour: (median_ratio_log, sigma)}.

    median_ratio_log is ln(viewers_h / viewers_1) at the target's scale;
    sigma is the log-space spread for that hour's quantiles. z0 is the
    target's console-only flag (1.0 or 0.0).
    """
    # Collect (x, z, {hour: y}) over usable, same-kind comparables.
    curves = []
    for comp in comparables:
        dh = day_hours(comp)
        if 1 not in dh:
            continue
        if sum(1 for h in range(2, 25) if h in dh) < MIN_DAY_HOURS:
            continue
        if not is_rolling(dh):
            continue
        p1 = dh[1]
        z = 1.0 if is_console_only_platforms(comp.get("platforms")) else 0.0
        curves.append((math.log(p1), z, {h: math.log(dh[h] / p1) for h in dh if h >= 2}))
    if not curves:
        return None
    xs = [c[0] for c in curves]
    xbar = sum(xs) / len(xs)
    zbar = sum(c[1] for c in curves) / len(curves)

    raw = {}
    for h in range(2, 25):
        pts = [(x, z, ys[h]) for x, z, ys in curves if h in ys]
        if len(pts) < MIN_POINTS_PER_HOUR:
            continue
        res = wls_multi([[p[0] for p in pts], [p[1] for p in pts]],
                        [p[2] for p in pts])
        if res is None:
            continue
        a, coefs, sigma, se_b, sxx, n = res
        raw[h] = {"a": a, "b": coefs[0], "cc": coefs[1] * CC_SHRINK,
                  "se_b": se_b, "sxx": sxx, "n": n, "sigma": sigma}

    if not raw:
        return None

    # Pooled slope: precision-weighted mean of the per-hour slopes.
    num = 0.0
    den = 0.0
    for r in raw.values():
        if math.isfinite(r["se_b"]) and r["se_b"] > 0:
            w = 1.0 / (r["se_b"] ** 2)
            num += w * r["b"]
            den += w
    beta_pool = num / den if den else sum(r["b"] for r in raw.values()) / len(raw)

    # Shrink each hour's slope toward the pooled slope (empirical Bayes).
    tau2 = BETA_PRIOR_SD ** 2
    fitted = {}
    for h, r in raw.items():
        se2 = r["se_b"] ** 2 if math.isfinite(r["se_b"]) else float("inf")
        if math.isfinite(se2) and se2 > 0:
            w = tau2 / (tau2 + se2)
            b_star = w * r["b"] + (1 - w) * beta_pool
        else:
            b_star = beta_pool
        fitted[h] = {"a": r["a"], "b": b_star, "cc": r["cc"],
                     "sigma": r["sigma"], "sxx": r["sxx"], "n": r["n"]}

    # Hours with too few points borrow the pooled shape: the median ratio of
    # the hours around them, at the target's scale, with a wide spread.
    have = sorted(fitted)
    pooled_a = sum(fitted[h]["a"] for h in have) / len(have)
    pooled_cc = sum(fitted[h]["cc"] for h in have) / len(have)
    pooled_sigma = max(fitted[h]["sigma"] for h in have)
    for h in range(2, 25):
        if h not in fitted:
            near = [fitted[k]["a"] for k in have if abs(k - h) <= 2]
            a = sum(near) / len(near) if near else pooled_a
            fitted[h] = {"a": a, "b": beta_pool, "cc": pooled_cc,
                         "sigma": pooled_sigma, "sxx": 0.0, "n": 0}

    alphas = smooth({h: fitted[h]["a"] for h in fitted}, SMOOTH_ALPHA)
    sigmas = smooth({h: fitted[h]["sigma"] for h in fitted}, SMOOTH_SIGMA)

    out = {}
    for h in range(2, 25):
        f = fitted[h]
        yhat = alphas[h] + f["b"] * (x0 - xbar) + f["cc"] * (z0 - zbar)
        # Leverage: the prediction is less certain the further the target
        # sits from the comparables' centre (matters on extrapolation).
        lev = 1.0
        if f["sxx"] > 0 and f["n"] > 0:
            lev = 1.0 + 1.0 / f["n"] + (x0 - xbar) ** 2 / f["sxx"]
        sigma = sigmas[h] * SIGMA_SCALE * math.sqrt(lev)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        out[h] = (yhat, sigma)
    return out


def baseline_quantiles(p1, h, levels):
    """House persistence baseline: every hour at the hour-1 print, spread
    widening like a random walk. Used only when no comparable fits."""
    zs = [norm_ppf(t) for t in levels]
    spread = BASELINE_SIGMA * math.sqrt(h - 1)
    return [round2(p1 * math.exp(z * spread)) for z in zs]


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    zs = [Z[i] if levels[i] in LEVELS and abs(levels[i] - LEVELS[i]) < 1e-9 else norm_ppf(t)
          for i, t in enumerate(levels)] if len(levels) == 5 else [norm_ppf(t) for t in levels]
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    model = None
    try:
        z0 = 1.0 if target_console_only(case) else 0.0
        model = fit_model(case.get("comparables") or [], math.log(p1), z0)
    except Exception:
        model = None

    hours = []
    for row in case.get("forecast", []):
        h = int(row["hour"])
        if model is not None and h in model:
            yhat, sigma = model[h]
            qs = [round2(p1 * math.exp(yhat + z * sigma)) for z in zs]
        else:
            qs = baseline_quantiles(p1, h, levels)
        # Enforce the format contract: finite, > 0, <= 1e9, non-decreasing.
        qs = [min(max(q, 0.01), 1e9) for q in qs]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
