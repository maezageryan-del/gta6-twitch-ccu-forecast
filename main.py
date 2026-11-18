"""Hour-by-hour Twitch audience forecast for the GTA VI launch day.

Entry for make.fun's gta6-twitch-ccu-forecast-01, second version. The job:
given a launch's hour-1 concurrent-viewer print and a library of earlier
launches' full release-day curves ("comparables"), predict five quantiles
of concurrent viewers for every remaining hour of day one.

Design in one paragraph. Earlier launches trace out a characteristic day
shape, and the shape's growth over hour 1 depends on how large hour 1 was
and on whether the launch was console-only. So, hour by hour, this program
regresses the log growth ln(viewers_h / viewers_1) on (ln viewers_1,
console-only flag) across the comparables and reads the target launch off
the fitted lines. Version 1 was exactly that single regression, with each
hour's slope shrunk toward the pooled slope and log-normal bands from the
residual spread; it scored 0.2220 on the practice board. Leave-one-out
testing over the comparables (every launch in turn hidden and predicted
from the rest) showed the residual loss sits in the median rather than
the bands -- coverage was already 0.87-0.96 -- because each launch's shape
wanders off any one line in its own way, and no extra signal or variant
tried (similarity-weighted transfer, LAD fits, unlock-time alignment,
kernel weighting, console interactions, donor trimming, family templates)
beat the plain regression on that protocol.

Version 2 therefore keeps the regression but builds it twice, in two
deliberately different ways, and averages the two quantile sets in log
space -- the one change that never made leave-one-out worse:

  * Model A is the version-1 fit: weighted least squares per hour,
    empirical-Bayes shrinkage of each hour's slope toward the pooled
    slope, the console coefficient shrunk 20% toward zero, and log-normal
    quantiles from the residual SD, widened by a leverage factor when the
    target sits far from the donor pool.

  * Model B refits the same per-hour regression with Huber-robust
    iteratively reweighted least squares, so a single freak launch cannot
    bend an hour's line; blends each slope a fixed 30% toward the pooled
    slope; and takes its bands from the empirical quantiles of the fit's
    own residuals pooled over neighbouring hours, which produces the
    asymmetric bands the residuals actually show (launches overshoot
    their line harder than they undershoot it).

Leave-one-out scores: ensemble 0.2121 vs 0.2118 for model A alone on all
rolling comparables; 0.1626 vs 0.1653 on the console-only subset (the
family every graded case belongs to); 0.1352 vs 0.1390 on an
extrapolation split (fit on small launches, predict the biggest). A small
edge, and it comes from robustness, not from a new signal.

Data discipline: only launches whose day looks like a rolling-midnight
console launch are used as donors -- a comparable qualifies when its curve
never exceeds 10x its own hour 1 (a simultaneous global unlock starts
near zero and explodes later, and mixing the families poisons the fit).
The forecast is a pure function of the case: no randomness anywhere, so
repeated runs produce identical files. If one model cannot be fitted it
drops out of the average; if neither can, the answer degrades to the
house persistence baseline (flat at the hour-1 print, random-walk
spread).

Input/output: reads every *.in in $MAKEFUN_INPUT_DIR (default /input)
and writes one *.out per case in $MAKEFUN_OUTPUT_DIR (default /output),
in the format the bounty rules specify. Standard library only.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# The grader imports this file with the script's directory missing from
# sys.path; restore it so any sibling module would resolve. (The entry is
# a single file, but the guard costs nothing.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# ---------------------------------------------------------------------------
# Configuration. Every number below was picked by leave-one-out testing over
# the comparables; the two models share the first block.
# ---------------------------------------------------------------------------

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
                 0.6744897501960817, 1.6448536269514722]

MIN_RECORDED_HOURS = 20    # donor must show hour 1 plus >= 20 of hours 2..24
MIN_DONORS_PER_HOUR = 5    # below this an hour falls back to the pooled shape
ROLLING_MAX_RATIO = 10.0   # rolling-midnight test: no hour above 10x hour 1
INTERCEPT_KERNEL = (0.2, 0.6, 0.2)   # neighbour smoothing for the shape level
SPREAD_KERNEL = (0.25, 0.5, 0.25)    # neighbour smoothing for the spread
SPREAD_MIN = 0.20          # tightest band any hour may claim (log units)
SPREAD_MAX = 1.10          # widest band any hour may claim
FALLBACK_SPREAD = 0.15     # persistence fallback spread per sqrt-hour

# Model A.
SLOPE_PRIOR_SD = 0.15        # plausible distance of an hour's slope from pooled
MODEL_A_SPREAD_SCALE = 0.9   # model A's bands slightly over-cover on purpose
MODEL_A_CONSOLE_SHRINK = 0.8  # console coefficient pulled 20% toward zero

# Model B.
MODEL_B_SLOPE_BLEND = 0.3    # fixed pull of each slope toward the pooled slope
HUBER_K = 1.345              # Huber tuning constant, in robust-SD units
HUBER_ROUNDS = 6             # IRLS reweighting rounds


# ---------------------------------------------------------------------------
# Small numeric helpers.
# ---------------------------------------------------------------------------

def parse_instant(text):
    """Parse an ISO-8601 timestamp with a trailing Z into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round_output(x):
    """Round to two decimals, halves away from zero (the house format), and
    return an int when the result is whole."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def normal_quantile(p):
    """Standard-normal quantile via Acklam's rational approximation."""
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


def solve_system(matrix, rhs):
    """Solve a tiny dense linear system by Gauss-Jordan elimination with
    partial pivoting. Returns the solution list, or None when singular."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(n):
            if r != col:
                factor = rows[r][col] / rows[col][col]
                for cc in range(col, n + 1):
                    rows[r][cc] -= factor * rows[col][cc]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def smooth_series(values, kernel):
    """Blend each hour 2..24 with its immediate neighbours per the kernel
    (previous, self, next); hours missing from `values` are skipped and the
    remaining weights renormalise."""
    out = {}
    for h in range(2, 25):
        num = 0.0
        den = 0.0
        for dh, w in ((-1, kernel[0]), (0, kernel[1]), (1, kernel[2])):
            if h + dh in values:
                num += w * values[h + dh]
                den += w
        out[h] = num / den if den else values[h]
    return out


def interpolated_quantile(sorted_values, tau):
    """Empirical quantile of an ascending list, linearly interpolated."""
    if not sorted_values:
        return 0.0
    pos = tau * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


# ---------------------------------------------------------------------------
# Reading cases and donor curves.
# ---------------------------------------------------------------------------

def launch_day_viewers(comp):
    """Map hour (1..24) -> viewers for one comparable's release day.

    Records that are not launch days at all come back empty: for instance
    the target category's own pre-release history, whose day_start is a
    premiere date, has no hour 1 and too few day hours, and the caller's
    completeness filter discards it.
    """
    try:
        start = parse_instant(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for row in comp.get("hourly") or []:
        try:
            t = parse_instant(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        h = int((t - start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out[h] = float(viewers)
    return out


def looks_like_rolling_launch(day):
    """Rolling-midnight shape test: the audience is already present in
    hour 1 and no later hour runs far above it. A simultaneous global
    unlock fails this -- almost nobody in hour 1, then a 10x-1000x spike."""
    if 1 not in day:
        return False
    first = day[1]
    return all(day[h] / first <= ROLLING_MAX_RATIO for h in day if h >= 2)


def platforms_console_only(platforms):
    """True when the platform list contains no PC and no Stadia. A missing
    or unknown list counts as NOT console-only: the conservative side for
    the regression flag."""
    if not platforms:
        return False
    text = str(platforms)
    return "PC" not in text and "Stadia" not in text


def case_is_console_only(case):
    """Decide the target launch's console-only flag.

    Preference order: the target game's own row among the comparables,
    when the case carries one (the final case does -- the GTA VI
    category's pre-release history, platforms "PS5, Xbox Series X|S");
    otherwise the case notes, which name the launch platforms ("released
    ... on PS5 only", "on PS4, Xbox One (console only; PC 2019-11)");
    otherwise True, since the rules guarantee every case in this
    competition is a rolling-midnight console launch.
    """
    name = str((case.get("game") or {}).get("name") or "").strip().lower()
    for comp in case.get("comparables") or []:
        cname = str(comp.get("name") or "").strip().lower()
        if name and (cname.startswith(name) or name in cname):
            return platforms_console_only(comp.get("platforms"))
    notes = str(case.get("notes") or "")
    if notes:
        m = re.search(r"released\s+\S+\s+on\s+([^;.\n]+)", notes)
        if m:
            clause = re.sub(r"\([^)]*\)", "", m.group(1))
            return "PC" not in clause and "Stadia" not in clause
    return True


def usable_donor_curves(comparables):
    """Distil the comparables into regression-ready donor curves:
    a list of (ln hour-1 viewers, console flag, {hour: ln growth over
    hour 1}) for every comparable that passes the completeness and
    rolling-shape filters."""
    curves = []
    for comp in comparables:
        day = launch_day_viewers(comp)
        if 1 not in day:
            continue
        if sum(1 for h in range(2, 25) if h in day) < MIN_RECORDED_HOURS:
            continue
        if not looks_like_rolling_launch(day):
            continue
        p1 = day[1]
        z = 1.0 if platforms_console_only(comp.get("platforms")) else 0.0
        curves.append((math.log(p1), z,
                       {h: math.log(day[h] / p1) for h in day if h >= 2}))
    return curves


# ---------------------------------------------------------------------------
# Regression machinery shared by both models.
# ---------------------------------------------------------------------------

def fit_weighted_line(columns, targets, weights=None):
    """Weighted least squares of targets on an intercept plus the given
    predictor columns, each column centred at its weighted mean.

    Returns a dict with the intercept (at the column means), the slopes,
    the residual SD, the first slope's standard error and centred sum of
    squares, the point count, the residuals, and the column means -- or
    None when the system is singular or the weights vanish.
    """
    n = len(targets)
    k = len(columns)
    if weights is None:
        weights = [1.0] * n
    total_w = sum(weights)
    if total_w <= 1e-9:
        return None
    means = [sum(w * c for w, c in zip(weights, columns[j])) / total_w
             for j in range(k)]
    y_mean = sum(w * y for w, y in zip(weights, targets)) / total_w
    normal = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for t in range(n):
        centred = [columns[j][t] - means[j] for j in range(k)]
        for i in range(k):
            rhs[i] += weights[t] * centred[i] * (targets[t] - y_mean)
            for j in range(k):
                normal[i][j] += weights[t] * centred[i] * centred[j]
    slopes = solve_system(normal, rhs)
    if slopes is None:
        return None
    residuals = [targets[t] - (y_mean + sum(
        slopes[j] * (columns[j][t] - means[j]) for j in range(k)))
        for t in range(n)]
    dof = max(n - k - 1, 1)
    var = sum(w * r * r for w, r in zip(weights, residuals)) / total_w * n / dof
    sigma = math.sqrt(max(var, 0.0))
    inv00 = None
    if k == 1:
        inv00 = 1.0 / normal[0][0] if normal[0][0] > 0 else None
    elif k == 2:
        det = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0]
        inv00 = normal[1][1] / det if abs(det) > 1e-12 else None
    se_slope = math.sqrt(var * inv00) if inv00 else float("inf")
    return {"intercept": y_mean, "slopes": slopes, "sigma": sigma,
            "se_slope": se_slope, "sxx": normal[0][0], "n": n,
            "residuals": residuals, "means": means}


def huber_refit(columns, targets, weights):
    """Model B's fitting routine: an initial weighted fit, then HUBER_ROUNDS
    of IRLS reweighting that caps each point's pull at HUBER_K robust SDs.

    Returns (intercept, slopes, residuals, column means) of the final fit,
    or None."""
    fit = fit_weighted_line(columns, targets, weights)
    if fit is None:
        return None
    n = fit["n"]
    current = list(weights)
    for _ in range(HUBER_ROUNDS):
        scale = 1.4826 * sorted(abs(v) for v in fit["residuals"])[n // 2] + 1e-9
        current = [weights[t] * min(1.0, HUBER_K * scale / max(abs(fit["residuals"][t]), 1e-9))
                   for t in range(n)]
        fit = fit_weighted_line(columns, targets, current)
        if fit is None:
            return None
    return fit["intercept"], fit["slopes"], fit["residuals"], fit["means"]


def hour_points(curves, h):
    """The (x, z, log-growth) donor points available for one hour."""
    return [(x, z, ys[h]) for x, z, ys in curves if h in ys]


def point_columns(pts):
    """Split (x, z, y) donor points into the predictor-column lists and
    the target list the regression routines take."""
    return [[p[0] for p in pts], [p[1] for p in pts]], [p[2] for p in pts]


def fill_missing_hours(fitted, pooled_slope, pooled_console, means,
                       fill_sigma=None):
    """Give every hour 2..24 a fit: hours the donors could not support
    inherit the pooled slope and console effect, an intercept averaged
    from fitted neighbours within two hours (else the pooled intercept),
    the donor pool's column means (so the prediction formula centres the
    same way a real fit would), and -- for model A -- the widest fitted
    spread, with no leverage information."""
    have = sorted(fitted)
    pooled_intercept = sum(fitted[h]["intercept"] for h in have) / len(have)
    for h in range(2, 25):
        if h not in fitted:
            near = [fitted[k]["intercept"] for k in have if abs(k - h) <= 2]
            entry = {"intercept": sum(near) / len(near) if near else pooled_intercept,
                     "slope": pooled_slope, "console": pooled_console,
                     "residuals": [], "means": means}
            if fill_sigma is not None:
                entry["sigma"] = fill_sigma
                entry["sxx"] = 0.0
                entry["n"] = 0
            fitted[h] = entry
    return fitted


# ---------------------------------------------------------------------------
# Model A: empirical-Bayes shrunk least squares, log-normal bands.
# ---------------------------------------------------------------------------

def model_a_quantiles(curves, log_p1, z_target):
    """Return {hour: [log quantile / hour-1 at QUANTILES]} or None."""
    if not curves:
        return None
    xs = [c[0] for c in curves]
    x_mean = sum(xs) / len(xs)
    z_mean = sum(c[1] for c in curves) / len(curves)

    raw = {}
    for h in range(2, 25):
        pts = hour_points(curves, h)
        if len(pts) < MIN_DONORS_PER_HOUR:
            continue
        columns, targets = point_columns(pts)
        fit = fit_weighted_line(columns, targets)
        if fit is None:
            continue
        raw[h] = {"intercept": fit["intercept"], "slope": fit["slopes"][0],
                  "console": fit["slopes"][1] * MODEL_A_CONSOLE_SHRINK,
                  "se_slope": fit["se_slope"], "sxx": fit["sxx"],
                  "n": fit["n"], "sigma": fit["sigma"], "means": fit["means"]}
    if not raw:
        return None

    # Pooled slope: inverse-variance weighted mean of the per-hour slopes.
    num = 0.0
    den = 0.0
    for r in raw.values():
        if math.isfinite(r["se_slope"]) and r["se_slope"] > 0:
            w = 1.0 / (r["se_slope"] ** 2)
            num += w * r["slope"]
            den += w
    pooled_slope = num / den if den else \
        sum(r["slope"] for r in raw.values()) / len(raw)

    # Shrink each hour's slope toward the pooled one, empirical-Bayes style:
    # an hour whose slope is measured precisely keeps it; a noisy hour is
    # pulled most of the way to the pool.
    prior_var = SLOPE_PRIOR_SD ** 2
    fitted = {}
    for h, r in raw.items():
        se2 = r["se_slope"] ** 2 if math.isfinite(r["se_slope"]) else float("inf")
        if math.isfinite(se2) and se2 > 0:
            w = prior_var / (prior_var + se2)
            slope = w * r["slope"] + (1 - w) * pooled_slope
        else:
            slope = pooled_slope
        fitted[h] = {"intercept": r["intercept"], "slope": slope,
                     "console": r["console"], "sigma": r["sigma"],
                     "sxx": r["sxx"], "n": r["n"], "means": r["means"]}

    have = sorted(fitted)
    pooled_console = sum(fitted[h]["console"] for h in have) / len(have)
    widest_sigma = max(fitted[h]["sigma"] for h in have)
    fill_missing_hours(fitted, pooled_slope, pooled_console,
                       (x_mean, z_mean), fill_sigma=widest_sigma)

    levels = smooth_series({h: fitted[h]["intercept"] for h in fitted},
                           INTERCEPT_KERNEL)
    spreads = smooth_series({h: fitted[h]["sigma"] for h in fitted},
                            SPREAD_KERNEL)

    out = {}
    for h in range(2, 25):
        f = fitted[h]
        mx, mz = f["means"]
        centre = levels[h] + f["slope"] * (log_p1 - mx) + \
            f["console"] * (z_target - mz)
        leverage = 1.0
        if f["sxx"] > 0 and f["n"] > 0:
            leverage = 1.0 + 1.0 / f["n"] + (log_p1 - mx) ** 2 / f["sxx"]
        spread = spreads[h] * MODEL_A_SPREAD_SCALE * math.sqrt(leverage)
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
        out[h] = [centre + z * spread for z in NORMAL_SCORES]
    return out


# ---------------------------------------------------------------------------
# Model B: Huber-robust fit, empirical residual bands.
# ---------------------------------------------------------------------------

def model_b_quantiles(curves, log_p1, z_target):
    """Return {hour: [log quantile / hour-1 at QUANTILES]} or None."""
    if not curves:
        return None
    raw = {}
    for h in range(2, 25):
        pts = hour_points(curves, h)
        if len(pts) < MIN_DONORS_PER_HOUR:
            continue
        columns, targets = point_columns(pts)
        fit = huber_refit(columns, targets, [1.0] * len(pts))
        if fit is None:
            continue
        intercept, slopes, residuals, means = fit
        raw[h] = {"intercept": intercept, "slope": slopes[0],
                  "console": slopes[1], "residuals": residuals,
                  "means": means}
    if not raw:
        return None

    pooled_slope = sum(r["slope"] for r in raw.values()) / len(raw)
    fitted = {}
    for h, r in raw.items():
        fitted[h] = {"intercept": r["intercept"],
                     "slope": (1 - MODEL_B_SLOPE_BLEND) * r["slope"]
                     + MODEL_B_SLOPE_BLEND * pooled_slope,
                     "console": r["console"], "residuals": r["residuals"],
                     "means": r["means"]}

    have = sorted(fitted)
    pooled_console = sum(fitted[h]["console"] for h in have) / len(have)
    xs = [c[0] for c in curves]
    x_mean = sum(xs) / len(xs)
    z_mean = sum(c[1] for c in curves) / len(curves)
    fill_missing_hours(fitted, pooled_slope, pooled_console, (x_mean, z_mean))

    levels = smooth_series({h: fitted[h]["intercept"] for h in fitted},
                           INTERCEPT_KERNEL)
    every_residual = sorted(r for f in fitted.values() for r in f["residuals"])

    out = {}
    for h in range(2, 25):
        f = fitted[h]
        mx, mz = f["means"]
        centre = levels[h] + f["slope"] * (log_p1 - mx) + \
            f["console"] * (z_target - mz)
        # Bands from the residuals of this hour and its neighbours; a thin
        # pool falls back to every residual the model produced.
        pool = []
        for hh in (h - 1, h, h + 1):
            if hh in fitted:
                pool.extend(fitted[hh]["residuals"])
        if len(pool) < 10:
            pool = every_residual
        pool = sorted(pool)
        out[h] = [centre + interpolated_quantile(pool, tau) for tau in QUANTILES]
    return out


# ---------------------------------------------------------------------------
# Forecast assembly.
# ---------------------------------------------------------------------------

def persistence_quantiles(p1, h, levels):
    """The house persistence baseline: flat at the hour-1 print with a
    random-walk spread. Only used when no donor curve can be fitted."""
    spread = FALLBACK_SPREAD * math.sqrt(h - 1)
    return [round_output(p1 * math.exp(normal_quantile(t) * spread))
            for t in levels]


def forecast_one(case, label):
    """Produce the {"hours": [...]} forecast dict for one case."""
    levels = case.get("levels") or QUANTILES
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{label} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    models = []
    try:
        curves = usable_donor_curves(case.get("comparables") or [])
        z_target = 1.0 if case_is_console_only(case) else 0.0
        log_p1 = math.log(p1)
        for model in (model_a_quantiles, model_b_quantiles):
            try:
                fitted = model(curves, log_p1, z_target)
            except Exception:
                fitted = None
            if fitted is not None:
                models.append(fitted)
    except Exception:
        models = []

    hours = []
    for row in case.get("forecast", []):
        h = int(row["hour"])
        if models and all(h in m for m in models):
            # Geometric mean of the models' quantiles, level by level: the
            # ensemble lives in log space, like everything else here.
            logs = [sum(m[h][i] for m in models) / len(models) for i in range(5)]
            qs = [round_output(p1 * math.exp(v)) for v in logs]
        else:
            qs = persistence_quantiles(p1, h, levels)
        # Format contract: finite, positive, at most 1e9, non-decreasing.
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
        result = forecast_one(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
