"""GTA VI trophy-rarity forecaster (gta6-trophy-rarity-forecast-01).

Model: counts, not raw rates — read at refresh-consistent moments.

Exophase refreshes its counts on its own schedule, so the players-tracked
denominator jumps in steps while earners only accumulate. A rate can fall
sharply even though nobody un-earned a trophy, and a page frozen between
refreshes repeats the same numbers for hours. This program therefore
works only with refresh-consistent reads: the first read, and every read
whose page changed from the previous one (tracked count or any rate).
Stale repeats carry no new information, and at a refresh instant the
earners and the tracked count belong to the same moment, so the rate then
is the true rate at that moment — progress minus dilution, net.

On those reads it fits, per trophy, logit(rate) ~= a + b * ln(1 + hours
since release). Progress saturates in log time; a flat history collapses
to one distinct read, gives slope 0, and the forecast falls back to
persistence. (An earlier draft fitted the adjusted share
earners / tracked_last in logit space; that link is wrong — the share can
exceed 1, since earners at the target can outnumber the tracked count at
the checkpoint — so it was replaced by this fit on refresh-consistent
rates, which nets dilution out directly.)

Slopes are pooled: trophies are grouped by tier, kind of trophy (story /
grind / one-shot activity, read off the description) and starting-rate
bucket, and a trophy's slope is shrunk toward its group's median slope,
with the shrinkage weight growing in the spread of log-time actually
observed. Thin trophies borrow strength; well-observed ones keep their own.

Spread (logit units) is horizon aware and honest about refresh risk:

    s^2 = s_rw^2 + s_resid^2 + s_refresh^2

with s_rw = 0.5 * sqrt(horizon in days) as the floor (the house baseline's
random walk), s_resid from the fit residuals, and s_refresh a refresh-risk
term: a 0.40 prior combined with the typical (median) log-jump of
players_tracked actually seen in the history, capped at 1.2. The 0.40
prior is not tuned on the practice cases: the rules themselves state
Exophase refreshes on its own schedule and that one refresh moved tracked
638 -> 3868, so a forecast that pretends the denominator cannot jump
between checkpoint and target is overconfident by construction; 0.40
prices roughly a 50% odds move per refresh cycle, well under that
documented first-refresh move. The term is constant in horizon because
it prices the unknown time since the last refresh before the target, not
a drift.

Special cases: a trophy at 100% stays at ~100% (its earners equal the
tracked count by construction); a trophy with zero earners starts from a
half-earner floor instead of logit(0). Every quantile is clamped to
0.01..99.99, non-decreasing, and rounded to 4 decimals half-up.

Long-horizon shape: a straight line in logit-vs-log-time over-extrapolates
a saturating curve — progress slows as a trophy's ceiling is approached,
so the line keeps climbing where the curve flattens. The forecast
therefore also fits, per trophy, a saturating curve
rate(t) = C * (1 - exp(-(t + 3h) / tau)) on the refresh-consistent reads
(the 3h shift is a fixed launch offset: rates are already ~1% at release,
not 0), over a grid of tau, and blends its logit with the line's logit
(0.65 saturating / 0.35 line) when the saturating fit is at least as good
in logit space. On a flat history the saturating fit is unavailable and
the forecast is persistence, as it should be.

Python 3.13, standard library only. Deterministic: no randomness, no
clock reads, files processed in name order.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Refresh-risk prior, in logit units (see module docstring).
REFRESH_PRIOR = 0.40
REFRESH_CAP = 1.2
# Random-walk floor per square-root day, as in the house baseline.
RW_SIGMA = 0.5
# Residual spread floor / cap, in logit units.
RESID_FLOOR = 0.15
RESID_CAP = 1.0
SPREAD_CAP = 2.5
# Slope clamps, in logit per unit of ln(1 + hours).
SLOPE_LO = -1.0
SLOPE_HI = 2.5
# Saturating-curve fit: fixed launch offset (hours) and tau grid (hours).
SAT_SHIFT_H = 3.0
SAT_TAUS = [6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0]
SAT_WEIGHT = 0.65
# Weight half-life for reads, in hours: recent reads matter more.
WEIGHT_HALF_LIFE_H = 36.0


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(logit_value):
    if logit_value >= 0:
        return 1.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# --------------------------------------------------------------------------
# history preparation
# --------------------------------------------------------------------------

def release_moment(case):
    """Midnight UTC on the game's release date; fall back to first read."""
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        try:
            return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    history = case.get("history") or []
    if history:
        return parse_time(history[0]["at"])
    return parse_time(case["at"])


def log_hours(moment, origin):
    hours = (moment - origin).total_seconds() / 3600.0
    if hours < 0.0:
        hours = 0.0
    return math.log1p(hours)


def read_weights(history, checkpoint):
    """Recency weights, half-life WEIGHT_HALF_LIFE_H, oldest to newest."""
    weights = []
    for read in history:
        age_h = (checkpoint - parse_time(read["at"])).total_seconds() / 3600.0
        if age_h < 0.0:
            age_h = 0.0
        weights.append(0.5 ** (age_h / WEIGHT_HALF_LIFE_H))
    return weights


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*(x - xbar); returns (a, b, sxx, resid_rms).

    a is the fitted value at the weighted mean xbar. resid_rms is the
    weighted root-mean-square residual.
    """
    sw = sum(ws)
    if sw <= 0.0 or not xs:
        return 0.0, 0.0, 0.0, 0.0
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - ybar) ** 2 for w, y in zip(ws, ys)) / sw)
        return ybar, 0.0, 0.0, resid
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    resid = math.sqrt(
        sum(w * (y - (ybar + b * (x - xbar))) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    )
    return ybar, b, sxx, resid


def trophy_kind(description):
    """Coarse kind from the description: story, grind, or one-shot activity."""
    text = (description or "").lower()
    if "chapter" in text or "final mission" in text or "100% completion" in text:
        return "story"
    if any(token in text for token in ("250 ", "100 ", "50 ", "gold medal", "20 different",
                                       "10 ", "8 mission", "5 bank", "3 collectible")):
        return "grind"
    return "activity"


def saturating_fit(t_hours, rates_frac, ws):
    """Fit rate(t) = C * (1 - exp(-(t + SAT_SHIFT_H) / tau)) by grid over tau.

    For each tau, C is solved by weighted least squares in rate space and
    tau is picked by weighted SSE in logit space. Returns (C, tau, sse)
    or None when there is too little movement to identify the shape.

    Reads from the first hours after release are excluded by the caller:
    rates there are already ~1% while the curve starts at 0, so they
    reflect launch-night early adopters, not the progress curve, and a
    single near-zero point would dominate the logit SSE.
    """
    distinct = sorted(set(round(r, 6) for r in rates_frac))
    if len(t_hours) < 5 or len(distinct) < 3:
        return None
    best = None
    for tau in SAT_TAUS:
        gs = [1.0 - math.exp(-(t + SAT_SHIFT_H) / tau) for t in t_hours]
        num = sum(w * g * r for w, g, r in zip(ws, gs, rates_frac))
        den = sum(w * g * g for w, g in zip(ws, gs))
        if den <= 0.0:
            continue
        c = min(max(num / den, max(rates_frac)), 0.98)
        sse = 0.0
        for w, g, r in zip(ws, gs, rates_frac):
            pred = min(max(c * g, 0.0001), 0.9999)
            obs = min(max(r, 0.0001), 0.9999)
            d = (math.log(obs / (1.0 - obs))
                 - math.log(pred / (1.0 - pred)))
            sse += w * d * d
        if best is None or sse < best[2]:
            best = (c, tau, sse)
    return best


def rate_bucket(last_rate):
    if last_rate is None:
        return "mid"
    if last_rate <= 0.8:
        return "low"
    if last_rate <= 1.3:
        return "mid"
    return "high"


# --------------------------------------------------------------------------
# per-case forecast
# --------------------------------------------------------------------------

def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    zetas = [norm_ppf(level) for level in levels]
    horizon_min = case.get("horizon_minutes")
    if horizon_min is None:
        horizon_min = (parse_time(case["target"]) - parse_time(case["at"])).total_seconds() / 60.0
    horizon_days = max(horizon_min, 0.0) / 1440.0

    if not history:
        # No reads at all: fall back to a flat 1% with a wide spread.
        rows = []
        for trophy in trophies:
            centre = logit_pct(1.0)
            rows.append({"id": trophy["id"], "q": _quantiles(centre, 1.5, zetas)})
        return {"trophies": rows}

    origin = release_moment(case)
    checkpoint = parse_time(case["at"])
    target_x = log_hours(parse_time(case["target"]), origin)
    weights = read_weights(history, checkpoint)
    xs = [log_hours(parse_time(read["at"]), origin) for read in history]

    # --- refresh-consistent reads: first read, plus every read whose page
    # changed from the previous one. Frozen repeats between refreshes are
    # dropped: they are the same observation restated, and at a refresh
    # instant earners and tracked belong to the same moment. -------------
    kept = [0]
    for j in range(1, len(history)):
        prev, cur = history[j - 1], history[j]
        if (cur.get("players_tracked") != prev.get("players_tracked")
                or (cur.get("rates") or []) != (prev.get("rates") or [])):
            kept.append(j)

    # --- observed refresh jumps in tracked ------------------------------
    jumps = []
    prev_n = None
    for read in history:
        n = read.get("players_tracked") or 0
        if prev_n and n > 0 and n > prev_n:
            jumps.append(math.log(n / prev_n))
        if n > 0:
            prev_n = n
    typical_jump = median(jumps) if jumps else 0.0
    s_refresh = min(math.sqrt(REFRESH_PRIOR ** 2 + typical_jump ** 2), REFRESH_CAP)

    # --- per-trophy fits: logit(rate) vs log time, on kept reads only ----
    fits = []
    for i, trophy in enumerate(trophies):
        t_xs, t_ys, t_ws = [], [], []
        sat_t, sat_r, sat_w, sat_x, sat_y = [], [], [], [], []
        last_rate = None
        for j in kept:
            read = history[j]
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            n = read.get("players_tracked") or 0
            if rate is None or n <= 0:
                continue
            if rate <= 0.0:
                # Zero earners: start from a half-earner floor, not logit(0).
                rate = 100.0 * 0.5 / n
            t_xs.append(xs[j])
            t_ys.append(logit_pct(rate))
            t_ws.append(weights[j])
            t_h = (parse_time(read["at"]) - origin).total_seconds() / 3600.0
            if max(t_h, 0.0) >= 6.0:
                # Saturating-curve inputs only: skip launch-night reads.
                sat_t.append(max(t_h, 0.0))
                sat_r.append(rate / 100.0)
                sat_w.append(weights[j])
                sat_x.append(xs[j])
                sat_y.append(logit_pct(rate))
        if not t_xs:
            fits.append(None)
            continue
        a, b, sxx, resid = wls(t_xs, t_ys, t_ws)
        xbar = sum(w * x for w, x in zip(t_ws, t_xs)) / sum(t_ws)
        # Linear SSE on the same reads the saturating fit sees, so the
        # two shapes are compared like for like.
        sse_lin_sat = sum(
            w * (y - (a + b * (x - xbar))) ** 2
            for w, x, y in zip(sat_w, sat_x, sat_y))
        # Last non-missing rate, for fallbacks and the 100% special case.
        for read in reversed(history):
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is not None:
                last_rate = rate
                break
        fits.append({
            "a": a, "b": b, "sxx": sxx, "resid": resid, "xbar": xbar,
            "sse_lin": sse_lin_sat,
            "sat": saturating_fit(sat_t, sat_r, sat_w),
            "kind": trophy_kind(trophy.get("description")),
            "tier": trophy.get("tier"),
            "bucket": rate_bucket(last_rate),
            "last_rate": last_rate,
        })

    # --- group median slopes (pooling) ----------------------------------
    def group_slope(keyfn):
        groups = {}
        for fit in fits:
            if fit is None or fit["sxx"] <= 0.05:
                continue
            groups.setdefault(keyfn(fit), []).append(fit["b"])
        return {key: median(vals) for key, vals in groups.items() if len(vals) >= 2}

    by_fine = group_slope(lambda f: (f["tier"], f["kind"], f["bucket"]))
    by_tier_bucket = group_slope(lambda f: (f["tier"], f["bucket"]))
    by_tier = group_slope(lambda f: f["tier"])
    global_slopes = [f["b"] for f in fits if f is not None and f["sxx"] > 0.05]
    global_slope = median(global_slopes) if global_slopes else 0.0

    def pooled_slope(fit):
        key_fine = (fit["tier"], fit["kind"], fit["bucket"])
        if key_fine in by_fine:
            prior = by_fine[key_fine]
        elif (fit["tier"], fit["bucket"]) in by_tier_bucket:
            prior = by_tier_bucket[(fit["tier"], fit["bucket"])]
        elif fit["tier"] in by_tier:
            prior = by_tier[fit["tier"]]
        else:
            prior = global_slope
        w_own = fit["sxx"] / (fit["sxx"] + 0.75) if fit["sxx"] > 0 else 0.0
        shrunk = w_own * fit["b"] + (1.0 - w_own) * prior
        return min(max(shrunk, SLOPE_LO), SLOPE_HI)

    group_resid = median([f["resid"] for f in fits if f is not None]) if any(
        f is not None for f in fits) else 0.0
    s_rw = RW_SIGMA * math.sqrt(horizon_days)

    # --- assemble rows ----------------------------------------------------
    tier_last_rates = {}
    for fit in fits:
        if fit is not None and fit["last_rate"] is not None:
            tier_last_rates.setdefault(fit["tier"], []).append(fit["last_rate"])

    rows = []
    for i, trophy in enumerate(trophies):
        fit = fits[i]
        if fit is None or fit["last_rate"] is None:
            # Trophy never observed: borrow its tier's median last rate.
            pool = tier_last_rates.get(trophy.get("tier")) or [1.0]
            centre_rate = median(pool)
            centre = logit_pct(centre_rate)
            rows.append({"id": trophy["id"], "q": _quantiles(centre, 1.5, zetas)})
            continue
        if fit["last_rate"] >= 99.99:
            # Earned by everyone tracked: stays at ~100%.
            rows.append({"id": trophy["id"], "q": [99.9, 99.95, 99.99, 99.99, 99.99]})
            continue
        b = pooled_slope(fit)
        centre = fit["a"] + b * (target_x - fit["xbar"])
        centre = min(max(centre, -9.21), 9.21)
        sat = fit.get("sat")
        if sat is not None and sat[2] <= fit["sse_lin"]:
            # The saturating curve fits at least as well: trust it more for
            # the long-horizon extrapolation, keep the line as a hedge.
            c_sat, tau_sat, _sse = sat
            t_target_h = (parse_time(case["target"]) - origin).total_seconds() / 3600.0
            rate_sat = 100.0 * c_sat * (
                1.0 - math.exp(-(max(t_target_h, 0.0) + SAT_SHIFT_H) / tau_sat))
            centre = (1.0 - SAT_WEIGHT) * centre + SAT_WEIGHT * logit_pct(rate_sat)
            centre = min(max(centre, -9.21), 9.21)
        s_resid = min(max(fit["resid"], 0.5 * group_resid, RESID_FLOOR), RESID_CAP)
        spread = math.sqrt(s_rw * s_rw + s_resid * s_resid + s_refresh * s_refresh)
        spread = min(spread, SPREAD_CAP)
        rows.append({"id": trophy["id"], "q": _quantiles(centre, spread, zetas)})
    return {"trophies": rows}


def _quantiles(centre_logit, spread, zetas):
    out = []
    prev = None
    for z in zetas:
        q = sigmoid(centre_logit + z * spread) * 100.0
        q = round4(clamp_pct(q))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
