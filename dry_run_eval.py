"""
Dry-run for evaluate_offline.py — no model required.

Loads the same JSONL test files that evaluate_offline.py reads, builds the
exact same (system, user) prompt that would be sent to the trained model,
hands it to a pluggable "agent" callable, and compares the agent's reply
to the gold tactic from the test data.

Use this to:
  * Verify the test file has the right schema and is parsed correctly.
  * Sanity-check the prompt format end-to-end without loading a GPU adapter.
  * Plug in any agent (a stub, a local HTTP endpoint, an LLM CLI) and score
    its outputs against the gold tactics.

Usage:
    # Default: MockAgent always returns the gold tactic (sanity check, 100%).
    python training/dry_run_eval.py \
        --test-proofs training/test-proofs-even-odd.jsonl

    # Chat-format file (one independent step per line):
    python training/dry_run_eval.py \
        --test-proofs training/test-even-or-odd-holdout.jsonl

    # Just print the prompts that would be sent — no scoring.
    python training/dry_run_eval.py \
        --test-proofs training/test-even-or-odd-holdout.jsonl --echo

    # Use an OpenAI-compatible HTTP agent (Ollama, vLLM, …):
    python training/dry_run_eval.py \
        --test-proofs training/test-even-or-odd-holdout.jsonl \
        --agent http://localhost:11434/v1 --agent-model llama3

    # Use a shell command as the agent (stdin = user content, stdout = reply):
    python training/dry_run_eval.py \
        --test-proofs training/test-even-or-odd-holdout.jsonl \
        --agent-cmd "my-agent --one-shot"
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time

# Force UTF-8 stdout so Unicode (e.g. Pie subscripts like x₁) works on Windows.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from collections import defaultdict
from pathlib import Path

# Reuse the prompt + scoring helpers so this script can never drift from the
# real evaluator.
from evaluate_offline import (
    SYSTEM_PROMPT,
    format_proof_state,
    tactic_head,
    tactic_category,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Agents ──────────────────────────────────────────────────────────────────

class Agent:
    """Anything that maps a user message → a tactic string."""
    def reply(self, user_content: str) -> str:
        raise NotImplementedError


class MockAgent(Agent):
    """Returns a fixed string for every step. Default = gold (perfect score)."""
    def __init__(self, fixed: str | None = None):
        self.fixed = fixed

    def reply(self, user_content: str) -> str:
        return self.fixed if self.fixed is not None else ""


class GoldAgent(Agent):
    """Special agent: caller passes the gold in via reply()'s second arg."""
    def reply(self, user_content: str, gold: str = "") -> str:  # type: ignore[override]
        return gold


class HTTPAgent(Agent):
    """OpenAI-compatible /chat/completions endpoint."""
    def __init__(self, base_url: str, model: str = ""):
        import requests
        self._requests = requests
        self.base = base_url.rstrip("/")
        self.model = model or self._detect()

    def _detect(self) -> str:
        r = self._requests.get(f"{self.base}/models", timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise RuntimeError(f"No models at {self.base}/models — pass --agent-model")
        return data[0]["id"]

    def reply(self, user_content: str) -> str:
        r = self._requests.post(
            f"{self.base}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
                "max_tokens": 128,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class CmdAgent(Agent):
    """Spawns a shell command per step. stdin = user content, stdout = reply."""
    def __init__(self, cmd: str):
        self.cmd = cmd

    def reply(self, user_content: str) -> str:
        proc = subprocess.run(
            self.cmd, shell=True, input=user_content,
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"agent cmd failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        return proc.stdout.strip()


# ── Loading ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def iter_steps(records: list[dict]):
    """Yield (label, user_content, gold) for every step in the file,
    auto-detecting per-proof eval format vs. chat training format."""
    if not records:
        return
    if "messages" in records[0]:
        # Chat format: each line = 1 independent step.
        for i, rec in enumerate(records):
            msgs = rec["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            gold = next(m["content"] for m in msgs if m["role"] == "assistant").strip()
            yield (f"step#{i+1}", user, gold)
    else:
        # Per-proof eval format.
        for rec in records:
            name = rec.get("theoremName", "<unnamed>")
            for step in rec.get("steps", []):
                user = format_proof_state(
                    step["goal"],
                    step.get("globalContext", []),
                    step.get("localContext", []),
                )
                gold = step["goldTactic"]
                yield (f"{name}:step{step['stepIndex']}", user, gold)


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dry-run the offline evaluator")
    p.add_argument("--test-proofs", required=True, help="Path to test JSONL")
    p.add_argument("--max-examples", type=int, default=0,
                   help="Limit number of steps (0 = all)")
    p.add_argument("--echo", action="store_true",
                   help="Just print the prompts that would be sent — no scoring")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--agent", type=str,
                   help="OpenAI-compatible base URL (HTTP agent)")
    g.add_argument("--agent-cmd", type=str,
                   help="Shell command to spawn per step (stdin=user, stdout=reply)")
    g.add_argument("--mock", type=str,
                   help="MockAgent always returns this string")

    p.add_argument("--agent-model", type=str, default="",
                   help="Model name for HTTP agent")
    p.add_argument("--output", type=str, default="",
                   help="Write per-step results JSON")
    return p.parse_args()


def build_agent(args) -> tuple[Agent, bool]:
    """Returns (agent, is_gold_agent)."""
    if args.agent:
        return HTTPAgent(args.agent, args.agent_model), False
    if args.agent_cmd:
        return CmdAgent(args.agent_cmd), False
    if args.mock is not None:
        return MockAgent(args.mock), False
    # Default: gold agent — proves the pipeline end-to-end with 100% match.
    return GoldAgent(), True


def main():
    args = parse_args()
    test_path = Path(args.test_proofs)
    if not test_path.exists():
        print(f"Test file not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(test_path)
    steps = list(iter_steps(records))
    if args.max_examples > 0:
        steps = steps[:args.max_examples]

    print(f"Loaded {len(records)} records → {len(steps)} steps from {test_path.name}")

    if args.echo:
        for label, user, gold in steps:
            print(f"\n── {label} ─────────────────────────────────")
            print(f"[system] {SYSTEM_PROMPT}")
            print(f"[user]\n{user}")
            print(f"[gold] {gold}")
        return

    agent, is_gold = build_agent(args)
    agent_kind = type(agent).__name__
    print(f"Agent: {agent_kind}\n")

    t0 = time.time()
    total = exact = head = cat = 0
    per_tactic = defaultdict(lambda: {"total": 0, "exact": 0, "head": 0})
    results = []
    failures = []

    for label, user, gold in steps:
        if is_gold:
            predicted = agent.reply(user, gold)  # type: ignore[call-arg]
        else:
            predicted = agent.reply(user)
        predicted = (predicted or "").strip()

        is_exact = predicted == gold
        is_head = tactic_head(predicted) == tactic_head(gold)
        is_cat = tactic_category(predicted) == tactic_category(gold)

        total += 1
        exact += is_exact
        head += is_head
        cat += is_cat

        gh = tactic_head(gold)
        per_tactic[gh]["total"] += 1
        if is_exact: per_tactic[gh]["exact"] += 1
        if is_head:  per_tactic[gh]["head"] += 1

        status = "OK" if is_exact else ("~" if is_head else "X")
        print(f"[{status}] {label}  gold={gold!r}  pred={predicted!r}")

        results.append({
            "label": label, "gold": gold, "predicted": predicted,
            "exact": is_exact, "head": is_head, "category": is_cat,
        })
        if not is_exact and len(failures) < 20:
            failures.append((label, gold, predicted))

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  Dry-run results ({agent_kind})")
    print(f"{'=' * 60}")
    print(f"  Total steps:  {total}")
    if total:
        print(f"  Exact-match:  {exact}/{total} ({exact/total:.1%})")
        print(f"  Tactic-head:  {head}/{total} ({head/total:.1%})")
        print(f"  Category:     {cat}/{total} ({cat/total:.1%})")
    print(f"  Time:         {elapsed:.1f}s")

    if per_tactic:
        print(f"\n── Per-tactic ──")
        for t in sorted(per_tactic, key=lambda k: per_tactic[k]["total"], reverse=True):
            s = per_tactic[t]
            ex = s["exact"] / s["total"] * 100 if s["total"] else 0
            hd = s["head"]  / s["total"] * 100 if s["total"] else 0
            print(f"  {t:16s}  n={s['total']:4d}  exact={ex:5.1f}%  head={hd:5.1f}%")

    if failures:
        print(f"\n── First failures ({min(10, len(failures))}/{len(failures)}) ──")
        for label, g, p in failures[:10]:
            print(f"  {label}: gold={g!r}  pred={p!r}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "agent": agent_kind,
                "total": total, "exact": exact, "head": head, "category": cat,
                "elapsed_seconds": elapsed,
                "per_tactic": {k: dict(v) for k, v in per_tactic.items()},
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
