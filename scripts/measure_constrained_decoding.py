"""Measures whether grammar-constrained decoding fixes FunctionGemma 270M's
paraphrase failures.

The unconstrained model emits prose refusals for anything that does not share
tokens with a tool description (8% routing accuracy on paraphrases). Constrained
decoding restricts the output to a JSON schema whose `name` is an enum of the
declared tools, so the model must emit a valid tool call and can only pick from
the tools actually on offer.

Two arms, same prompts, same tool set, same conversation lifecycle:
  unconstrained  -- plain send_message, tool_calls compared
  constrained    -- response_format = JSON schema with name.enum = tools

Run: venv/bin/python scripts/measure_constrained_decoding.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import litert_lm
from litert_lm import interfaces
from litert_lm._ffi import LiteRtLmConstraintProviderType

MODEL = "assets/function-gemma-q8-ekv1024.litertlm"

TOOLS = {
    "open_flashlight": "Turns the phone's flashlight on.",
    "close_flashlight": "Turns the phone's flashlight off.",
    "query_calendar": "Lists the events on the user's calendar for today.",
    "take_photo": "Takes a photo with the phone's camera.",
}

# The 18 prompts the 4-tool desktop cell scored, with the literal/paraphrase
# labels that make the failure mode legible.
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


def schema_for(names: list[str]) -> dict[str, Any]:
  return {
      "type": "object",
      "properties": {
          "name": {"type": "string", "enum": names},
          # Deliberately permissive: the argument schema is what this run is not
          # measuring. Constraining it would conflate two variables.
          "arguments": {"type": "object"},
      },
      "required": ["name", "arguments"],
  }


def ask(
    engine: Any,
    constrained: bool,
    prompt: str,
    expected: str,
    kind: str,
) -> dict[str, Any]:
  schema_tools = [make_tool(name) for name in TOOLS]

  kwargs: dict[str, Any] = {
      "tools": schema_tools,
      "automatic_tool_calling": False,
      "sampler_config": interfaces.SamplerConfig(temperature=0.0, seed=0),
  }
  if constrained:
    kwargs["constrained_decoding_config"] = interfaces.ConstrainedDecodingConfig(
        enable=True,
        provider=LiteRtLmConstraintProviderType.LL_GUIDANCE,
    )

  with engine.create_conversation(**kwargs) as convo:
    send: dict[str, Any] = {}
    if constrained:
      send["response_format"] = interfaces.ResponseFormat.json(
          schema_for(list(TOOLS))
      )
    start = time.monotonic()
    response = convo.send_message(prompt, **send)
    elapsed = time.monotonic() - start

  got = extract_name(response)
  return {
      "prompt": prompt,
      "expected": expected,
      "kind": kind,
      "got": got,
      "ok": got == expected,
      "seconds": round(elapsed, 3),
  }


def extract_name(response: Any) -> str | None:
  """Tool name from either arm: a real tool_call, or a constrained JSON body."""
  calls = response.get("tool_calls") or []
  if calls:
    return calls[0].get("function", {}).get("name")
  text = "".join(p.get("text", "") for p in response.get("content") or [])
  if not text.strip():
    return None
  try:
    parsed = json.loads(text)
  except json.JSONDecodeError:
    return None
  name = parsed.get("name") if isinstance(parsed, dict) else None
  return name if isinstance(name, str) else None


class SchemaTool(interfaces.Tool):
  """Hand-written OpenAPI schema; litert_lm.tools cannot express these."""

  def __init__(self, spec: dict[str, Any]) -> None:
    self._spec = spec

  def get_tool_description(self) -> dict[str, Any]:
    return self._spec

  def execute(self, param: Any) -> str:
    return "ok"


def make_tool(name: str) -> SchemaTool:
  return SchemaTool({
      "type": "function",
      "function": {
          "name": name,
          "description": TOOLS[name],
          "parameters": {"type": "object", "properties": {}, "required": []},
      },
  })


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
  if not rows:
    return {"n": 0}
  by_kind: dict[str, list[bool]] = {}
  for row in rows:
    by_kind.setdefault(row["kind"], []).append(row["ok"])
  times = [r["seconds"] for r in rows]
  return {
      "n": len(rows),
      "accuracy": round(sum(r["ok"] for r in rows) / len(rows), 4),
      "refusal_rate": round(
          sum(1 for r in rows if r["got"] is None) / len(rows), 4
      ),
      "by_kind": {
          kind: {"n": len(hits), "accuracy": round(sum(hits) / len(hits), 4)}
          for kind, hits in sorted(by_kind.items())
      },
      "median_seconds": round(statistics.median(times), 3),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default=MODEL)
  parser.add_argument("--out", default="measurements/constrained_decoding.json")
  args = parser.parse_args()

  engine = litert_lm.Engine(
      model_path=args.model,
      backend=interfaces.Backend.CPU(),
      max_num_tokens=2048,
      cache_dir=os.path.abspath(".litertlm-cache"),
  )

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  arms: dict[str, list[dict[str, Any]]] = {"unconstrained": [], "constrained": []}
  for arm in ("unconstrained", "constrained"):
    constrained = arm == "constrained"
    print(f"== {arm}")
    for prompt, expected, kind in PROBES:
      row = ask(engine, constrained, prompt, expected, kind)
      arms[arm].append(row)
      print(
          f"  {'OK ' if row['ok'] else 'BAD'} {kind:11} {prompt!r:38} "
          f"want={expected:16} got={row['got'] or 'refusal'} {row['seconds']}s"
      )
    # Written after every arm, not once at the end: with the LLGuidance provider
    # enabled this process can abort in native teardown (glibc "corrupted size
    # vs. prev_size" / "free(): invalid pointer") *after* every measurement has
    # already succeeded. See the note in the recipe. Persisting per arm means a
    # teardown abort cannot destroy a completed run.
    with open(args.out, "w") as handle:
      json.dump(
          {
              "model": os.path.basename(args.model),
              "backend": "cpu",
              "prompts": len(PROBES),
              "summaries": {a: summarize(r) for a, r in arms.items()},
              "detail": arms,
          },
          handle,
          indent=2,
      )

  summaries = {arm: summarize(rows) for arm, rows in arms.items()}
  print("\n| arm | accuracy | refusal | literal | paraphrase | terse | median s |")
  print("|---|---|---|---|---|---|---|")
  for arm, stats in summaries.items():
    kind = stats["by_kind"]
    print(
        f"| {arm} | {stats['accuracy']:.0%} | {stats['refusal_rate']:.0%} | "
        f"{kind.get('literal', {}).get('accuracy', 0):.0%} | "
        f"{kind.get('paraphrase', {}).get('accuracy', 0):.0%} | "
        f"{kind.get('terse', {}).get('accuracy', 0):.0%} | "
        f"{stats['median_seconds']:.2f} |"
    )

  report = {
      "model": os.path.basename(args.model),
      "backend": "cpu",
      "prompts": len(PROBES),
      "summaries": summaries,
      "detail": arms,
  }
  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as handle:
    json.dump(report, handle, indent=2)
  print(f"\nwrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())