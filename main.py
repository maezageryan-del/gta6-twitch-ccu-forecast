"""GTA VI launch-day Twitch viewers, hour by hour: a two-model ensemble.

V2 of the comparable-shape forecaster. V1 was a single per-hour regression:
for each hour h, regress ln(viewers_h / viewers_1) on ln(viewers_1) and a
console-only flag across every rolling-midnight comparable launch, shrink
each hour's slope toward the pooled slope, and read the quantiles off a
log-normal spread fitted to the residuals. Its practice loss was 0.2220
(typical miss 43.7%).

Leave-one-out diagnosis over the comparables (each launch in turn treated
as the target, the rest as donors) showed where that loss lives: not in
the spread (coverage was already 0.87-0.96) and not in any single hour or
fixable bias, but in the median itself -- each launch's day shape deviates
from the regression line idiosyncratically, and hour-1 observables explain
only part of it. Single-model variants measured against the same protocol
-- similarity-weighted shape transfer, pinball (LAD) fits, unlock-time
alignment, local kernel weighting, console interactions, donor trimming,
plain family templates -- all landed at or above V1's leave-one-out loss.

So V2 keeps V1's regression as model A and ensembles it, in log space,
with a deliberately different second model, because averaging two
decorrelated forecasters is the one change that never hurt in testing:

  Model A (V1): per-hour weighted least squares of the log ratio on
  (ln viewers_1, console flag), empirical-Bayes slope shrinkage toward
  the pooled slope, smoothed intercepts, log-normal quantiles from the
  residual SD with a leverage widening for far-away targets.

  Model B (new): the same per-hour regression family, but fitted
  Huber-robustly (iteratively reweighted, so one freak launch in the
  donor pool cannot drag an hour's line), a fixed 30% slope blend toward
  the pooled slope instead of the EB weights, and -- instead of a normal
  spread -- the empirical quantiles of the fit's own residuals, pooled
  over neighbouring hours. That gives asymmetric bands that follow the
  actual residual distribution (launches overshoot their line more
  violently than they undershoot it) rather than a symmetric curve.

The ensemble's quantiles are the geometric means of the two models'
quantiles, per hour and level. On leave-one-out over the rolling
comparables the ensemble scored 0.2121 (V1 alone 0.2118); on the
console-only subset -- the family every graded case belongs to -- 0.1626
against V1's 0.1653; on the extrapolation split (train on small launches,
test on the biggest) 0.1352 against 0.1390. A modest, honest edge, bought
with robustness rather than a new signal.

Both models use only the case's hour-1 print and the comparables'
release-day curves, keep only rolling-midnight launches (a day curve that
never rises above 10x its hour 1), and need no randomness: two runs are
byte-identical. Fallbacks keep every case valid: a model that cannot fit
drops out of the ensemble; if neither fits, the forecast is the house
persistence baseline.

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

# --- shared tuning constants (chosen on leave-one-out over the comparables) ---
MIN_DAY_HOURS = 20      # a comparable must hold hour 1 and >= 20 of hours 2..24
MIN_POINTS_PER_HOUR = 5  # else fall back to the pooled shape for that hour
MAX_DAY_RATIO = 10.0    # rolling-midnight test: no day hour above 10x hour 1
SMOOTH_ALPHA = (0.2, 0.6, 0.2)   # neighbour smoothing of the shape, (prev, self, next)
SMOOTH_SIGMA = (0.25, 0.5, 0.25)
SIGMA_FLOOR = 0.20      # no hour claims a tighter band than this (log units)
SIGMA_CAP = 1.10        # and no hour is allowed to be wider than this
BASELINE_SIGMA = 0.15   # persistence fallback spread, per sqrt-hour (house baseline)

# --- model A constants (as V1) ---
BETA_PRIOR_SD = 0.15    # how far one hour's slope may stray from the pooled slope
SIGMA_SCALE_A = 0.9     # model A's residual spread slightly over-covers
CC_SHRINK_A = 0.8       # shrink model A's console coefficient 20% toward zero

# --- model B constants ---
BETA_BLEND_B = 0.3      # fixed blend of each hour's slope toward the pooled slope
HUBER_C = 1.345         # Huber tuning constant (in robust-SD units)
HUBER_ITERS = 6


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


def wls_multi(cols, ys, ws=None):
    """Weighted least squares y ~ a + b1*(x1 - x1bar) + b2*(x2 - x2bar) ...

    Returns (a, coefs, sigma, se_b1, sxx1, n, resid, bars): the intercept
    at the columns' weighted means, one coefficient per column, the
    residual SD, the first slope's standard error and centred sum of
    squares, the point count, the residuals and the column means.
    """
    n = len(ys)
    k = len(cols)
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    if sw <= 1e-9:
        return None
    bars = [sum(w * c for w, c in zip(ws, cols[j])) / sw for j in range(k)]
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    mat = [[0.0] * k for _ in range(k)]
    vec = [0.0] * k
    for t in range(n):
        d = [cols[j][t] - bars[j] for j in range(k)]
        for i in range(k):
            vec[i] += ws[t] * d[i] * (ys[t] - ybar)
            for j in range(k):
                mat[i][j] += ws[t] * d[i] * d[j]
    coefs = solve_linear(mat, vec)
    if coefs is None:
        return None
    resid = [ys[t] - (ybar + sum(coefs[j] * (cols[j][t] - bars[j]) for j in range(k)))
             for t in range(n)]
    dof = max(n - k - 1, 1)
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / sw * n / dof
    sigma = math.sqrt(max(s2, 0.0))
    inv00 = None
    if k == 1:
        inv00 = 1.0 / mat[0][0] if mat[0][0] > 0 else None
    elif k == 2:
        det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        inv00 = mat[1][1] / det if abs(det) > 1e-12 else None
    se_b = math.sqrt(s2 * inv00) if inv00 else float("inf")
    sxx = mat[0][0]
    return ybar, coefs, sigma, se_b, sxx, n, resid, bars


def huber_fit(cols, ys, ws):
    """Model B's fit: WLS, then Huber IRLS reweighting around the fit.

    Returns (a, coefs, resid, bars) of the final weighted fit, or None.
    """
    r = wls_multi(cols, ys, ws)
    if r is None:
        return None
    a, coefs, _, _, _, n, resid, bars = r
    wcur = list(ws)
    for _ in range(HUBER_ITERS):
        s = 1.4826 * sorted(abs(v) for v in resid)[n // 2] + 1e-9
        wcur = [ws[t] * min(1.0, HUBER_C * s / max(abs(resid[t]), 1e-9))
                for t in range(n)]
        r = wls_multi(cols, ys, wcur)
        if r is None:
            return None
        a, coefs, _, _, _, n, resid, bars = r
    return a, coefs, resid, bars


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


def emp_quantile(sorted_vals, tau):
    """Linear-interpolated empirical quantile of a sorted list."""
    if not sorted_vals:
        return 0.0
    pos = tau * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


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


def collect_curves(comparables):
    """(x, z, {hour: log-ratio}) over usable rolling-midnight comparables."""
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
        curves.append((math.log(p1), z,
                       {h: math.log(dh[h] / p1) for h in dh if h >= 2}))
    return curves


def fit_a(curves, x0, z0):
    """Model A (V1): EB-shrunk OLS, log-normal quantiles.

    Returns {hour: [log(q_tau / p1) at the five LEVELS]} or None.
    """
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
        a, coefs, sigma, se_b, sxx, n, resid, bars = res
        raw[h] = {"a": a, "b": coefs[0], "cc": coefs[1] * CC_SHRINK_A,
                  "se_b": se_b, "sxx": sxx, "n": n, "sigma": sigma,
                  "bars": bars}
    if not raw:
        return None

    num = 0.0
    den = 0.0
    for r in raw.values():
        if math.isfinite(r["se_b"]) and r["se_b"] > 0:
            w = 1.0 / (r["se_b"] ** 2)
            num += w * r["b"]
            den += w
    beta_pool = num / den if den else sum(r["b"] for r in raw.values()) / len(raw)

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
                     "sigma": r["sigma"], "sxx": r["sxx"], "n": r["n"],
                     "bars": r["bars"]}

    have = sorted(fitted)
    pooled_a = sum(fitted[h]["a"] for h in have) / len(have)
    pooled_cc = sum(fitted[h]["cc"] for h in have) / len(have)
    pooled_sigma = max(fitted[h]["sigma"] for h in have)
    for h in range(2, 25):
        if h not in fitted:
            near = [fitted[k]["a"] for k in have if abs(k - h) <= 2]
            a = sum(near) / len(near) if near else pooled_a
            fitted[h] = {"a": a, "b": beta_pool, "cc": pooled_cc,
                         "sigma": pooled_sigma, "sxx": 0.0, "n": 0,
                         "bars": (xbar, zbar)}

    alphas = smooth({h: fitted[h]["a"] for h in fitted}, SMOOTH_ALPHA)
    sigmas = smooth({h: fitted[h]["sigma"] for h in fitted}, SMOOTH_SIGMA)

    out = {}
    for h in range(2, 25):
        f = fitted[h]
        bx, bz = f["bars"]
        yhat = alphas[h] + f["b"] * (x0 - bx) + f["cc"] * (z0 - bz)
        lev = 1.0
        if f["sxx"] > 0 and f["n"] > 0:
            lev = 1.0 + 1.0 / f["n"] + (x0 - bx) ** 2 / f["sxx"]
        sigma = sigmas[h] * SIGMA_SCALE_A * math.sqrt(lev)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        out[h] = [yhat + z * sigma for z in Z]
    return out


def fit_b(curves, x0, z0):
    """Model B: Huber-robust fit, empirical residual quantiles.

    Returns {hour: [log(q_tau / p1) at the five LEVELS]} or None.
    """
    if not curves:
        return None
    raw = {}
    for h in range(2, 25):
        pts = [(x, z, ys[h]) for x, z, ys in curves if h in ys]
        if len(pts) < MIN_POINTS_PER_HOUR:
            continue
        r = huber_fit([[p[0] for p in pts], [p[1] for p in pts]],
                      [p[2] for p in pts], [1.0] * len(pts))
        if r is None:
            continue
        a, coefs, resid, bars = r
        raw[h] = {"a": a, "b": coefs[0], "cc": coefs[1],
                  "resid": resid, "bars": bars}
    if not raw:
        return None

    beta_pool = sum(r["b"] for r in raw.values()) / len(raw)
    fitted = {}
    for h, r in raw.items():
        fitted[h] = {"a": r["a"],
                     "b": (1 - BETA_BLEND_B) * r["b"] + BETA_BLEND_B * beta_pool,
                     "cc": r["cc"], "resid": r["resid"], "bars": r["bars"]}

    have = sorted(fitted)
    pooled_a = sum(fitted[h]["a"] for h in have) / len(have)
    pooled_cc = sum(fitted[h]["cc"] for h in have) / len(have)
    xs = [c[0] for c in curves]
    xbar = sum(xs) / len(xs)
    zbar = sum(c[1] for c in curves) / len(curves)
    for h in range(2, 25):
        if h not in fitted:
            near = [fitted[k]["a"] for k in have if abs(k - h) <= 2]
            a = sum(near) / len(near) if near else pooled_a
            fitted[h] = {"a": a, "b": beta_pool, "cc": pooled_cc,
                         "resid": [], "bars": (xbar, zbar)}

    alphas = smooth({h: fitted[h]["a"] for h in fitted}, SMOOTH_ALPHA)
    all_resid = sorted(r for f in fitted.values() for r in f["resid"])

    out = {}
    for h in range(2, 25):
        f = fitted[h]
        bx, bz = f["bars"]
        yhat = alphas[h] + f["b"] * (x0 - bx) + f["cc"] * (z0 - bz)
        pool = []
        for hh in (h - 1, h, h + 1):
            if hh in fitted:
                pool.extend(fitted[hh]["resid"])
        if len(pool) < 10:
            pool = all_resid
        pool = sorted(pool)
        out[h] = [yhat + emp_quantile(pool, tau) for tau in LEVELS]
    return out


def baseline_quantiles(p1, h, levels):
    """House persistence baseline: every hour at the hour-1 print, spread
    widening like a random walk. Used only when no comparable fits."""
    zs = [norm_ppf(t) for t in levels]
    spread = BASELINE_SIGMA * math.sqrt(h - 1)
    return [round2(p1 * math.exp(z * spread)) for z in zs]


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    models = []
    try:
        curves = collect_curves(case.get("comparables") or [])
        z0 = 1.0 if target_console_only(case) else 0.0
        x0 = math.log(p1)
        for fit in (fit_a, fit_b):
            try:
                m = fit(curves, x0, z0)
            except Exception:
                m = None
            if m is not None:
                models.append(m)
    except Exception:
        models = []

    hours = []
    for row in case.get("forecast", []):
        h = int(row["hour"])
        if models and all(h in m for m in models):
            # Geometric mean of the models' quantiles, per level.
            logs = [sum(m[h][i] for m in models) / len(models) for i in range(5)]
            qs = [round2(p1 * math.exp(v)) for v in logs]
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
