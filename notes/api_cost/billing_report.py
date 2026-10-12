"""Estimate Tinker spend from the billing-usage API.

The API returns hourly token / GB-hour quantities but no dollar amounts, so this
script multiplies them by the *current* public price list. Historical discounts
are not reflected; the console invoice is the source of truth.

Usage:
    set -a && source .env && set +a
    .venv/bin/python notes/api_cost/billing_report.py --days 14
    .venv/bin/python notes/api_cost/billing_report.py --start 2026-09-13 --end 2026-10-12
    .venv/bin/python notes/api_cost/billing_report.py --cache /tmp/usage.json   # reuse a dump
"""

import argparse
import collections
import json
import os
import urllib.request
from datetime import UTC, datetime, timedelta

PRICES_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
SERVERLESS_URL = "https://tinker-docs.thinkingmachines.ai/tinker/serverless.json"
STORAGE_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 730
MAX_WINDOW = timedelta(days=14)
KINDS = ["prefill_cached", "prefill_uncached", "sample", "train", "storage"]


def _get_json(url: str) -> list[dict]:
    # The docs CDN rejects urllib's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _usd(s: str | None) -> float | None:
    return float(s.lstrip("$")) if s else None


def load_prices() -> tuple[dict[str, dict], dict[str, dict]]:
    train = {
        m["tinker_id"]: {
            "prefill": _usd(m["prefill"]),
            "cached": _usd(m["cached_prefill"]),
            "sample": _usd(m["sample"]),
            "train": _usd(m["train"]),
        }
        for m in _get_json(PRICES_URL)
    }
    serverless = {
        m["tinker_id"]: {
            "prefill": _usd(m["input"]),
            "cached": _usd(m["cached_input"]),
            "sample": _usd(m["output"]),
        }
        for m in _get_json(SERVERLESS_URL)
    }
    return train, serverless


def _parse_hour(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def fetch_usage(start: datetime, end: datetime) -> dict:
    from tinker import ServiceClient

    rest = ServiceClient().create_rest_client()
    rows, sessions = [], {}
    cur = start
    while cur < end:
        nxt = min(cur + MAX_WINDOW, end)
        resp = rest.get_billing_usage(
            cur.strftime("%Y-%m-%dT%H:%M:%SZ"), nxt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ).result()
        rows += [r.model_dump(mode="json") for r in resp.data]
        sessions.update({sid: s.model_dump(mode="json") for sid, s in resp.sessions.items()})
        cur = nxt
    return {"data": rows, "sessions": sessions}


def price_row(r: dict, prices: dict[str, dict]) -> tuple[float, str]:
    e = r["event_info"]
    t = e["type"]
    if t == "storage":
        return e["gigabyte_hours"] * STORAGE_PER_GB_MONTH / HOURS_PER_MONTH, "storage"
    if t == "checkpoint":
        return 0.0, "checkpoint"
    p = prices.get(r["base_model"])
    if p is None:
        return 0.0, f"unpriced:{r['base_model']}"
    n = e["token_count"] / 1e6
    if t == "training":
        return n * p["train"], "train"
    if t == "sampling_sample":
        return n * p["sample"], "sample"
    if e.get("cached"):
        return n * p["cached"], "prefill_cached"
    return n * p["prefill"], "prefill_uncached"


def serverless_price(r: dict, serverless: dict[str, dict]) -> float | None:
    """What this sampling row would cost on the nvfp4 serverless endpoint, if one exists."""
    e = r["event_info"]
    if not e["type"].startswith("sampling") or not r["base_model"]:
        return None
    p = serverless.get(r["base_model"] + ":sampling-nvfp4")
    if p is None:
        return None
    n = e["token_count"] / 1e6
    if e["type"] == "sampling_sample":
        return n * p["sample"]
    return n * (p["cached"] if e.get("cached") else p["prefill"])


def report(usage: dict, prices: dict, serverless: dict, top: int) -> None:
    rows, sessions = usage["data"], usage["sessions"]
    if not rows:
        print("No usage in window.")
        return

    def label(sid: str | None) -> str:
        if sid is None:
            return "(org-level)"
        md = (sessions.get(sid) or {}).get("user_metadata") or {}
        return md.get("recipe_name") or "(no metadata)"

    def short(model: str | None) -> str:
        return (model or "-").split("/")[-1]

    total = 0.0
    by_kind = collections.Counter()
    by_model_kind = collections.Counter()
    by_label = collections.Counter()
    by_day = collections.defaultdict(collections.Counter)
    by_sess = collections.Counter()
    sess_span: dict[str, list] = {}
    sampling_now = sampling_serverless = 0.0
    tok = collections.Counter()
    storage = []

    for r in rows:
        c, kind = price_row(r, prices)
        total += c
        day = r["bucket_start"][:10]
        by_kind[kind] += c
        by_model_kind[(short(r["base_model"]), kind)] += c
        by_label[label(r["session_id"])] += c
        by_day[day][kind] += c
        if kind.startswith(("prefill", "sample")):
            tok[kind] += r["event_info"]["token_count"]
        if kind == "storage":
            storage.append((r["bucket_start"], r["event_info"]["gigabyte_hours"]))
        if sid := r["session_id"]:
            by_sess[sid] += c
            span = sess_span.setdefault(sid, [day, day, short(r["base_model"])])
            span[1] = max(span[1], day)
        s = serverless_price(r, serverless)
        if s is not None:
            sampling_now += c
            sampling_serverless += s

    days = sorted(by_day)
    print(f"Estimated spend {days[0]}..{days[-1]} at current list price: ${total:,.2f}\n")

    print("By usage kind")
    for k, v in by_kind.most_common():
        print(f"  {k:24s} ${v:10,.2f}  {v / total:6.1%}")

    print("\nBy model x kind (>= $1)")
    for (m, k), v in by_model_kind.most_common():
        if v >= 1:
            print(f"  {m:40s} {k:18s} ${v:10,.2f}")

    print("\nBy day")
    print(f"  {'day':10s}  {'total':>9s}  " + "  ".join(f"{k[:12]:>12s}" for k in KINDS))
    for d in days:
        print(
            f"  {d}  ${sum(by_day[d].values()):8,.2f}  "
            + "  ".join(f"{by_day[d][k]:12,.2f}" for k in KINDS)
        )

    print("\nBy recipe_name metadata")
    for k, v in by_label.most_common():
        print(f"  {k:40s} ${v:10,.2f}  {v / total:6.1%}")

    print(f"\nTop {top} sessions")
    for sid, v in by_sess.most_common(top):
        first, last, model = sess_span[sid]
        print(f"  {sid:44s} ${v:9,.2f}  {model:32s} {first}..{last}  {label(sid)}")

    prefill = tok["prefill_cached"] + tok["prefill_uncached"]
    if tok["sample"]:
        print(
            f"\nSampling: input/output token ratio {prefill / tok['sample']:.1f}, "
            f"prefill cache hit {tok['prefill_cached'] / max(prefill, 1):.1%}"
        )
    if sampling_now:
        print(
            f"Sampling on models with a serverless endpoint: ${sampling_now:,.2f} now, "
            f"${sampling_serverless:,.2f} if served via :sampling-nvfp4"
        )
    if storage:
        storage.sort()
        gb = storage[-1][1]
        print(f"Checkpoint storage: ~{gb:,.0f} GB now (~${gb * STORAGE_PER_GB_MONTH:,.2f}/month)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--days", type=int, default=14, help="Look back this many days (ignored with --start)"
    )
    ap.add_argument("--start", help="UTC date or RFC 3339 time, inclusive")
    ap.add_argument("--end", help="UTC date or RFC 3339 time, exclusive (default: now)")
    ap.add_argument(
        "--cache",
        help="JSON file: read usage from it if it exists, else write the fetched usage to it",
    )
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        with open(args.cache) as f:
            usage = json.load(f)
    else:
        end = _parse_hour(args.end or datetime.now(UTC).isoformat())
        start = _parse_hour(args.start) if args.start else end - timedelta(days=args.days)
        usage = fetch_usage(start, end)
        if args.cache:
            with open(args.cache, "w") as f:
                json.dump(usage, f)

    prices, serverless = load_prices()
    report(usage, prices, serverless, args.top)


if __name__ == "__main__":
    main()
