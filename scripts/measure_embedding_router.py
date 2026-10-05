"""Tests the embedding-routing hypothesis against FunctionGemma 270M's failures.

Measured baseline (mobile-1b-model, CPU, 4 tools):
  unconstrained  28% routing accuracy, 72% refusal rate
  literal prompts 75%, paraphrase prompts 15%, terse 0%

Hypothesis: the failures are lexical, not semantic -- "do i have meetings today"
shares few tokens with "Lists the events on the user's calendar for today", so a
keyword matcher misses it even though the intent is identical. An embedding model
over tool descriptions should recover exactly those cases.

This script measures whether cosine similarity over MiniLM sentence embeddings
routes the same 18 prompts better than FunctionGemma does, so the answer is a
comparison rather than a claim. It does not need the LLM at all.

Run: venv/bin/python scripts/measure_embedding_router.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import numpy as np
from transformers import AutoTokenizer
from tflite_runtime.interpreter import Interpreter

MODEL = "minilm.tflite"

TOOLS = {
    "open_flashlight": "Turns the phone's flashlight on.",
    "close_flashlight": "Turns the phone's flashlight off.",
    "query_calendar": "Lists the events on the user's calendar for today.",
    "take_photo": "Takes a photo with the phone's camera.",
}

# The same 18 prompts and the same labels the LiteRT-LM gate uses.
PROBES: list[tuple[str, str, str]] = [
    ("what is on my calendar today", "query_calendar", "literal"),
    ("whats on my calender today", "query_calendar", "paraphrase"),
    ("calendar today", "query_calendar", "terse"),
    ("what's my schedule", "query_calendar", "paraphrase"),
    ("list my calendar events", "query_calendar", "paraphrase"),
    ("do i have meetings today", "query_calendar", "paraphrase"),
    ("am i busy today", "query_calendar", "paraphrase"),
    ("show my events", "query_calendar", "paraphrase"),
    ("turn on the flashlight", "open_flashlight", "literal"),
    ("i need some light in here", "open_flashlight", "paraphrase"),
    ("it is dark in here", "open_flashlight", "paraphrase"),
    ("lights on please", "open_flashlight", "paraphrase"),
    ("turn off the flashlight", "close_flashlight", "literal"),
    ("kill the torch now", "close_flashlight", "paraphrase"),
    ("shut the light", "close_flashlight", "paraphrase"),
    ("no more light", "close_flashlight", "paraphrase"),
    ("take a photo", "take_photo", "literal"),
    ("snap a picture of the room", "take_photo", "paraphrase"),
]


class Embedder:
  """MiniLM-L6-v2 through LiteRT, mean-pooled and L2-normalized.

  The graph takes already-tokenized int32 input and emits a [1,384] sentence
  vector. Normalizing at load time makes every later comparison a plain dot
  product, and makes the argmax invariant to vector length.
  """

  def __init__(self, model_path: str, max_length: int = 128) -> None:
    self.tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    self.max_length = max_length
    self.interp = Interpreter(model_path=model_path, num_threads=4)
    self.interp.allocate_tensors()
    self._in_id = self.interp.get_input_details()[0]["index"]
    self._in_mask = self.interp.get_input_details()[1]["index"]
    self._out = self.interp.get_output_details()[0]["index"]

  def embed(self, texts: list[str]) -> np.ndarray:
    """Embeds each text separately: this graph is fixed at batch 1.

    Output 0 is the pooled [384] sentence vector; it is L2-normalized here so
    callers compare with a plain dot product and the argmax is invariant to
    vector length.
    """
    out = []
    for text in texts:
      enc = self.tokenizer(
          text,
          padding="max_length",
          truncation=True,
          max_length=self.max_length,
          return_tensors="np",
      )
      # set_tensor enforces int64 here; tflite_runtime's summary reports the
      # logical element type, which is not what it validates against.
      ids = enc["input_ids"].astype(np.int64)
      mask = enc["attention_mask"].astype(np.int64)
      self.interp.set_tensor(self._in_id, ids)
      self.interp.set_tensor(self._in_mask, mask)
      self.interp.invoke()
      # The graph's output is ALREADY mean-pooled: the sentence-transformers
      # export bakes pooling into the graph, so output 0 is a single [384]
      # vector, not [128, 384]. Only L2 normalization is needed here.
      # (An earlier version of this file divided by the token count before
      # normalizing; that is a no-op, since a scalar factor cancels out of an
      # L2 normalization. The scores were correct, the comment was wrong.)
      vec = self.interp.get_tensor(self._out)[0].astype(np.float32)
      norm = float(np.linalg.norm(vec))
      out.append(vec / norm if norm else vec)
    return np.vstack(out)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default=MODEL)
  parser.add_argument("--out", default="measurements/embedding_router.json")
  args = parser.parse_args()

  embedder = Embedder(args.model)

  t0 = time.monotonic()
  tool_vecs = embedder.embed([f"{n}: {d}" for n, d in TOOLS.items()])
  init_ms = (time.monotonic() - t0) * 1000

  rows = []
  for prompt, expected, kind in PROBES:
    t1 = time.monotonic()
    vec = embedder.embed([prompt])[0]
    sims = tool_vecs @ vec
    order = np.argsort(-sims)
    got = list(TOOLS)[order[0]]
    ms = (time.monotonic() - t1) * 1000
    rows.append({
        "prompt": prompt,
        "expected": expected,
        "kind": kind,
        "got": got,
        "ok": got == expected,
        "margin": round(float(sims[order[0]] - sims[order[1]]), 4),
        "ms": round(ms, 1),
    })
    print(
        f"  {'OK ' if got == expected else 'BAD'} {kind:11} {prompt!r:38} "
        f"want={expected:16} got={got:16} margin={rows[-1]['margin']:+.3f} {ms:.0f}ms"
    )

  def acc(subset: list[dict[str, Any]]) -> float:
    return round(sum(r["ok"] for r in subset) / len(subset), 4) if subset else 0.0

  by_kind: dict[str, Any] = {}
  for kind in {r["kind"] for r in rows}:
    sub = [r for r in rows if r["kind"] == kind]
    by_kind[kind] = {"n": len(sub), "accuracy": acc(sub)}

  latencies = [r["ms"] for r in rows]
  # Margin is the router's own confidence signal: how far the top tool is ahead
  # of the runner-up. If that separates correct from incorrect, it can gate
  # whether the LLM is consulted at all.
  ok_margins = [r["margin"] for r in rows if r["ok"]]
  bad_margins = [r["margin"] for r in rows if not r["ok"]]

  report = {
      "router": "all-MiniLM-L6-v2 via LiteRT (.tflite, 89.9 MB)",
      "tools": list(TOOLS),
      "prompts": len(rows),
      "accuracy": acc(rows),
      "by_kind": by_kind,
      "median_ms": round(statistics.median(latencies), 1),
      "max_ms": round(max(latencies), 1),
      "tool_embed_init_ms": round(init_ms, 1),
      "margin": {
          "mean_when_correct": round(statistics.mean(ok_margins), 4) if ok_margins else None,
          "mean_when_wrong": round(statistics.mean(bad_margins), 4) if bad_margins else None,
      },
      "detail": rows,
  }

  print()
  print(f"accuracy {report['accuracy']:.0%} over {len(rows)} prompts")
  for kind, stats in sorted(by_kind.items()):
      print(f"  {kind:11} {stats['accuracy']:.0%} (n={stats['n']})")
  print(
      f"median {report['median_ms']:.0f} ms/prompt, max {report['max_ms']:.0f} ms "
      f"(vs 3120 ms for the 270M on device)"
  )
  print(
      f"margin: {report['margin']['mean_when_correct']} when correct, "
      f"{report['margin']['mean_when_wrong']} when wrong"
  )

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as handle:
    json.dump(report, handle, indent=2)
  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())