"""Runs the held-out set through the router and reports honestly.

No tuning happens here. The prototype bank and negative bank are frozen in
router_v2.py before this file was written, so this is the generalization number.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heldout_router_set import HELD_OUT  # noqa: E402
from router_v2 import CLARIFY, RouterV2  # noqa: E402


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--model", default="minilm.tflite")
  p.add_argument("--abstain-margin", type=float, default=0.02)
  p.add_argument("--tag", default="v2")
  p.add_argument("--out", default="measurements/router_heldout.json")
  args = p.parse_args()

  router = RouterV2(args.model, args.abstain_margin, clarify=True)

  rows = []
  for utterance, expected, bucket in HELD_OUT:
    got = router.route(utterance)
    ok = got["tool"] == expected
    rows.append({
        "utterance": utterance,
        "expected": expected,
        "got": got["tool"],
        "bucket": bucket,
        "margin": got["margin"],
        "reason": got["reason"],
        "correct": ok,
    })
    print(
        f"  {'OK ' if ok else 'BAD'} {bucket:13} {utterance!r:56} "
        f"got={got['tool']:16} want={expected:16}"
    )

  buckets = {}
  for bucket in sorted({r["bucket"] for r in rows}):
    sub = [r for r in rows if r["bucket"] == bucket]
    hits = sum(1 for r in sub if r["correct"])
    margins = [r["margin"] for r in sub if r["margin"] is not None]
    buckets[bucket] = {
        "n": len(sub),
        "correct": hits,
        "accuracy": round(hits / len(sub), 4),
        "median_margin": round(statistics.median(margins), 4) if margins else None,
    }

  # Safety metric: acting when it should not have. Two kinds.
  false_act = [r for r in rows if r["expected"] == CLARIFY and r["got"] != CLARIFY]
  abstain_overfire = [r for r in rows if r["expected"] != CLARIFY and r["got"] == CLARIFY]
  wrong_tool = [r for r in rows if r["expected"] != CLARIFY and r["got"] not in (CLARIFY, r["expected"])]

  total_correct = sum(1 for r in rows if r["correct"])
  report = {
      "tag": args.tag,
      "abstain_margin": args.abstain_margin,
      "n": len(rows),
      "overall": round(total_correct / len(rows), 4),
      "safety": {
          "false_act": len(false_act),
          "abstain_overfire": len(abstain_overfire),
          "wrong_tool": len(wrong_tool),
      },
      "buckets": buckets,
      "failures": [
          {k: r[k] for k in ("utterance", "expected", "got", "bucket", "margin")}
          for r in rows if not r["correct"]
      ],
      "detail": rows,
  }

  print(f"\n| bucket | n | correct | accuracy | median margin |")
  print("|---|---|---|---|---|")
  for b, s in buckets.items():
    print(f"| {b} | {s['n']} | {s['correct']} | {s['accuracy']:.0%} | {s['median_margin']} |")
  print(
      f"\noverall {report['overall']:.0%} ({total_correct}/{len(rows)})  "
      f"false_act={len(false_act)}  abstain_overfire={len(abstain_overfire)}  "
      f"wrong_tool={len(wrong_tool)}"
  )

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, "w") as handle:
    json.dump(report, handle, indent=2)
  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
