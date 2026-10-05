"""Stress tests the embedding router against the cases a 4-prompt gate cannot see.

`measure_embedding_router.py` reports 18/18, but every prompt there is a clean
single-intent English sentence whose intent is close to one tool description. That
is the region the router is strongest in, so it confirms the hypothesis without
testing its edges. This script builds the adversarial set:

  confusable   open vs close flashlight, send vs note -- tool descriptions that
               sit at ~0.89 cosine, so the decision is a coin flip on wording
  multi_intent two actions in one utterance ("turn on the light and check my
               calendar"); a single-label argmax has no correct answer
  noisy        truncated, filler-heavy, no punctuation, all-caps
  misspelled   real typos and voice-input errors
  out_of_scope intent with no matching tool, which must route to a fallback
  ambiguous    genuinely two-way ("is the light on?")

Reports accuracy per bucket and, for the confusable bucket, the decision margin,
because a correct answer won on a thin margin is not a robust route.

Run: venv/bin/python scripts/stress_embedding_router.py --model minilm.tflite
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_embedding_router import TOOLS  # noqa: E402
from measure_embedding_router import Embedder  # noqa: E402

# `expected=None` means there is no single correct tool: the utterance is
# out_of_scope, multi_intent, or genuinely ambiguous, and the router is expected to
# produce *some* tool. Those rows are reported separately and never counted as
# routing successes, because claiming credit for them would inflate the number.
CASES: list[tuple[str, str | None, str]] = [
    # --- confusable: two tools at ~0.89 cosine --------------------------------
    ("turn on the flashlight", "open_flashlight", "confusable"),
    ("switch the flashlight on", "open_flashlight", "confusable"),
    ("turn off the flashlight", "close_flashlight", "confusable"),
    ("switch the flashlight off", "close_flashlight", "confusable"),
    ("kill the torch", "close_flashlight", "confusable"),
    ("light on", "open_flashlight", "confusable"),
    ("light off", "close_flashlight", "confusable"),
    ("flashlight on please", "open_flashlight", "confusable"),
    # --- multi-intent: no single right answer ----------------------------------
    ("turn on the flashlight and check my calendar", None, "multi_intent"),
    ("take a photo and show my events", None, "multi_intent"),
    ("snap a pic then turn off the light", None, "multi_intent"),
    # --- noisy: real utterance shapes ------------------------------------------
    ("can u turn the torch on", "open_flashlight", "noisy"),
    ("pls flashlight off thx", "close_flashlight", "noisy"),
    ("TURN THE FLASHLIGHT OFF", "close_flashlight", "noisy"),
    ("ok so anyway lights on now", "open_flashlight", "noisy"),
    ("umm turn off the flashlight i guess", "close_flashlight", "noisy"),
    # --- misspelled: typos and voice-input errors ------------------------------
    ("turn on the flaslight", "open_flashlight", "misspelled"),
    ("flahslight on", "open_flashlight", "misspelled"),
    ("calender today", "query_calendar", "misspelled"),
    ("shw my evnets", "query_calendar", "misspelled"),
    ("tak a photo", "take_photo", "misspelled"),
    # --- out_of_scope: must not silently pick a tool ---------------------------
    ("what is the weather in nairobi", None, "out_of_scope"),
    ("who won the match on sunday", None, "out_of_scope"),
    ("play some music", None, "out_of_scope"),
    ("set a reminder for my dentist", None, "out_of_scope"),
    ("how tall is mount kenya", None, "out_of_scope"),
    # --- ambiguous: two readings, both defensible ------------------------------
    ("is the light on", None, "ambiguous"),
    ("should i turn the torch off", None, "ambiguous"),
]


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="minilm.tflite")
  parser.add_argument(
      "--out", default="measurements/embedding_router_stress.json"
  )
  args = parser.parse_args()

  embedder = Embedder(args.model)
  names = list(TOOLS)
  tool_vecs = embedder.embed([f"{n}: {TOOLS[n]}" for n in names])

  rows = []
  for prompt, expected, bucket in CASES:
    vec = embedder.embed([prompt])[0]
    sims = tool_vecs @ vec
    order = sorted(range(len(names)), key=lambda i: -sims[i])
    got = names[order[0]]
    margin = float(sims[order[0]] - sims[order[1]])
    judged = expected is None
    rows.append({
        "prompt": prompt,
        "bucket": bucket,
        "expected": expected,
        "got": got,
        "margin": round(margin, 4),
        "unscoreable": judged,
        "correct": None if judged else got == expected,
    })
    mark = "---" if judged else ("OK " if got == expected else "BAD")
    print(
        f"  {mark} {bucket:13} {prompt!r:48} got={got:16} "
        f"margin={margin:+.3f} expected={expected or 'n/a'}"
    )

  print()
  by_bucket: dict[str, dict[str, object]] = {}
  for bucket in sorted({r["bucket"] for r in rows}):
    sub = [r for r in rows if r["bucket"] == bucket]
    scored = [r for r in sub if not r["unscoreable"]]
    acc = (
        sum(1 for r in scored if r["correct"]) / len(scored) if scored else None
    )
    margins = [r["margin"] for r in scored]
    by_bucket[bucket] = {
        "n": len(sub),
        "scoreable": len(scored),
        "accuracy": round(acc, 4) if acc is not None else None,
        "median_margin": round(statistics.median(margins), 4) if margins else None,
        "min_margin": round(min(margins), 4) if margins else None,
    }

  print("| bucket | n | scoreable | accuracy | median margin | min margin |")
  print("|---|---|---|---|---|---|")
  for bucket, stats in by_bucket.items():
    acc = (
        f"{stats['accuracy']:.0%}"
        if stats["accuracy"] is not None
        else "n/a"
    )
    print(
        f"| {bucket} | {stats['n']} | {stats['scoreable']} | {acc} | "
        f"{stats['median_margin']} | {stats['min_margin']} |"
    )

  # What threshold would keep every scoreable row correct? The confusable bucket
  # is where a thin margin actually costs something.
  scoreable = [r for r in rows if not r["unscoreable"]]
  correct = [r for r in scoreable if r["correct"]]
  wrong = [r for r in scoreable if not r["correct"]]
  report = {
      "router": "all-MiniLM-L6-v2 via LiteRT",
      "buckets": by_bucket,
      "separability": {
          "min_margin_when_correct": round(
              min((r["margin"] for r in correct), default=None), 4
          ),
          "max_margin_when_wrong": round(
              max((r["margin"] for r in wrong), default=None), 4
          ),
      },
      "detail": rows,
  }

  print()
  if wrong:
    print(
        f"wrong answers exist: max margin when wrong = "
        f"{report['separability']['max_margin_when_wrong']}"
    )
  else:
    print("no wrong answers in this set, so a margin threshold cannot be calibrated")

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as handle:
    json.dump(report, handle, indent=2)
  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())