"""Notebook cell: distil the 26B teacher into small Gemma 3 students, one automated session, two arms.

No teacher labelling happens here. The teacher's soft targets are the published dataset
(j-raghavan/s1-decision-data, train.jsonl), downloaded in setup() and reused unchanged, so this session only
trains and evaluates small students. Each arm, in turn:

  1. download the student (google/gemma-3-1b-it, then google/gemma-3-270m-it)
  2. LoRA full run on the published rows, snapshots every SAVE steps, bounded by ARM_MINUTES (cost guard)
  3. score the untuned student and every snapshot on jevbench_dev + custom_dev2 through HF (run_decoder.py,
     option-letter logits, one pass) -- no vLLM: these models are small enough that HF scoring is fast
  4. choose the snapshot with decide.choose_snapshot_ni (best JevBench-dev within the custom-dev margin of the
     untuned student); score the chosen snapshot and the untuned student once on the jevbench + custom test sets
  5. results packed into /content/gemma3_out.tgz

Expects /content/s1_full.tgz (the repo + eval data + JevBench harness/data, built by scripts/colab_upload.sh)
extracted to /content/s1, and HF_TOKEN set in the environment (the students are gated). Holds the VM until
/content/downloaded.flag appears so the watcher can pull the results before release.

    colab exec -s <session> -f colab/session_gemma3.py

Dry run before spending the full budget: a small, cheap Colab session with
    F_MODELS=1b:google/gemma-3-270m-it F_FULL_STEPS=20 F_ARM_MINUTES=3 F_LIMIT=40 F_BATCH=4
exercises the whole flow (download, train, dev scoring, selection, test) end to end in a few minutes. The F_*
variables also shrink everything for a local check.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Arms: tag -> Hub id. The -it variants carry the chat_template; the base gemma-3-270m does not.
DEFAULT_ARMS = [("1b", "google/gemma-3-1b-it"), ("270m", "google/gemma-3-270m-it")]
ARMS = [tuple(a.split(":", 1)) for a in os.environ["F_MODELS"].split()] if os.environ.get("F_MODELS") else DEFAULT_ARMS

BASE = Path(os.environ.get("F_BASE", "/content"))
CUDA = os.environ.get("F_DEVICE", "cuda") == "cuda"
FULL_STEPS = int(os.environ.get("F_FULL_STEPS", 3200))  # ~1 epoch of ~100K rows at 32/step
FULL_SAVE = int(os.environ.get("F_FULL_SAVE", 500))
# Per-arm wall-clock cap (the trainer fits its step count into 85% of this after timing 10 steps). The 270M
# arm trains faster, so it gets less; both together bound the paid time.
ARM_MINUTES = {"1b": float(os.environ.get("F_ARM_MINUTES", 55)), "270m": float(os.environ.get("F_ARM_MINUTES_270M", 30))}
BATCH = os.environ.get("F_BATCH", "16")
VAL_LIMIT = os.environ.get("F_VAL_LIMIT", "400")
LIMIT = ["--limit", os.environ["F_LIMIT"]] if os.environ.get("F_LIMIT") else []
DATASET = os.environ.get("F_DATASET", "j-raghavan/s1-decision-data")
TARGETS = "data/train/train.jsonl"  # downloaded from the dataset in setup()
DEV = ("jevbench_dev", "custom_dev2")
TEST = ("jevbench", "custom")

W = BASE / "s1"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "decisions.json"
decisions: dict = {}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    decisions[key] = value
    REPORT.write_text(json.dumps(decisions, indent=1))


def run(args: list[str], log: str, timeout_min: float) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60).returncode
    except subprocess.TimeoutExpired:
        return -9


def rules():
    sys.path.insert(0, str(W))
    import s1.decide as d
    return d


def preds(suite: str, run_id: str) -> dict:
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    return rules().load_predictions(p) if p.exists() else {}


def n_dev_cases(suite: str) -> int:
    """Yes/no and choice cases the untuned model must answer (score slices are excluded from selection)."""
    sys.path.insert(0, str(W / "eval" / "jevbench"))
    from paths import SUITES
    lines = open(SUITES[suite]["cases"]).readlines()
    lines = lines[: int(LIMIT[1])] if LIMIT else lines
    return sum(1 for line in lines if '"task_type": "score"' not in line)


def hf_score(model: str, run_id: str, suite: str, adapter: Path | None) -> int:
    args = [sys.executable, "eval/jevbench/run_decoder.py", "--model", model, "--run-id", run_id, "--suite", suite,
            "--batch", BATCH, *LIMIT] + (["--adapter", str(adapter)] if adapter else [])
    return run(args, f"hf_{run_id}_{suite}.log", 30)


def train(model: str, out: Path, minutes: float, log: str) -> int:
    # In-training evaluation is off; every decision is taken on the dev splits afterwards.
    return run([sys.executable, "-m", "s1.train_decoder", "--model", model, "--targets", TARGETS, "--out", str(out),
                "--max-steps", str(FULL_STEPS), "--batch", BATCH, "--accum", "2", "--lr", "1e-4", "--rank", "64",
                "--save-every", str(FULL_SAVE), "--eval-every", "1000000", "--val-limit", VAL_LIMIT,
                "--skip-initial-eval", "--max-minutes", str(minutes)], log, minutes + 30)


def steps_run(log: str, planned: int) -> int:
    for line in reversed(open(BASE / log, errors="replace").read().splitlines()):
        if line.startswith("{") and '"done": true' in line:
            try:
                return int(json.loads(line)["steps"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return planned


def snapshots(out: Path, steps: int) -> dict[str, Path]:
    """step-N snapshots plus the final adapter; a step-N right before the end duplicates the final one."""
    margin = max(1, min(10, FULL_SAVE // 2))
    found = {p.name: p for p in sorted(out.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
             if (p / "adapter_model.safetensors").exists() and int(p.name.split("-")[1]) < steps - margin}
    if (out / "last" / "adapter_model.safetensors").exists():
        found["last"] = out / "last"
    return found


def setup() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_full.tgz -C {W}", shell=True, check=True)
    if CUDA:
        status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                              capture_output=True, text=True).stdout.strip())
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "peft==0.21.2",
                        "accelerate==1.15.0", "httpx", "tqdm"], check=True)
        # Colab preinstalls torchao, which PEFT rejects when it builds LoRA layers; nothing here uses it.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    from huggingface_hub import hf_hub_download
    dst = W / TARGETS
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(DATASET, "data/train.jsonl", repo_type="dataset")
    dst.write_bytes(Path(src).read_bytes())
    n = sum(1 for _ in dst.open())
    record("setup", {"arms": ARMS, "targets": TARGETS, "target_rows": n, "batch": BATCH,
                     "eval_limit": LIMIT[1] if LIMIT else None, "full_steps": FULL_STEPS})
    # Echo the effective scale up front: a leaked F_LIMIT/F_BATCH (e.g. from reusing a dry-run kernel) shows here.
    status(f"setup done: {n} training rows from {DATASET}; arms {[a for a, _ in ARMS]}; "
           f"batch {BATCH}, eval limit {LIMIT[1] if LIMIT else 'none'}, max-steps {FULL_STEPS}")


def dev_and_select(tag: str, model: str, snaps: dict[str, Path]):
    """Untuned student + every snapshot on both dev splits (HF), then choose_snapshot_ni."""
    d = rules()
    cands = {f"{tag}-{k}": p for k, p in snaps.items()}
    for suite in DEV:
        status(f"[{tag}] dev {suite}: untuned exit {hf_score(model, f'base-{tag}', suite, None)}")
        for name, path in cands.items():
            status(f"[{tag}] dev {suite}: {name} exit {hf_score(model, f'{name}-adapt', suite, path)}")
    base_c, base_j = preds("custom_dev2", f"base-{tag}"), preds("jevbench_dev", f"base-{tag}")
    if not d.base_complete(base_c, base_j, n_dev_cases("custom_dev2"), n_dev_cases("jevbench_dev")):
        status(f"[{tag}] untuned dev scoring incomplete ({len(base_c)} custom, {len(base_j)} jev); cannot select")
        return None, []
    best, report = d.choose_snapshot_ni(base_c, base_j, {n: (preds("custom_dev2", f"{n}-adapt"),
                                                            preds("jevbench_dev", f"{n}-adapt")) for n in cands})
    status(f"[{tag}] untuned dev jev {d.slice_mean_accuracy(base_j):.3f} custom {d.slice_mean_accuracy(base_c):.3f}")
    for r in report:
        status(f"[{tag}] {r['name']:16s} jev_dev {r['jev_acc']:.3f} cov {r['coverage']:.2f} custom "
               f"{r['custom_change'] * 100:+.2f} pts {'eligible' if r['eligible'] else 'NOT eligible'}")
    return best, report


def test_once(tag: str, model: str, chosen: str, path: Path) -> None:
    """The chosen snapshot and the untuned student, scored once on the test sets."""
    d = rules()
    for suite in TEST:
        status(f"[{tag}] TEST {suite}: {chosen} exit {hf_score(model, f'{tag}-test', suite, path)}")
        status(f"[{tag}] TEST {suite}: untuned exit {hf_score(model, f'base-{tag}-test', suite, None)}")
    jev, cust = preds("jevbench", f"{tag}-test"), preds("custom", f"{tag}-test")
    base_cust = preds("custom", f"base-{tag}-test")
    reg = d.regression(base_cust, cust)
    reg["accuracy_change"] = d.accuracy_change(base_cust, cust)
    reg["non_inferior"] = d.non_inferior(base_cust, cust)
    record(f"test_{tag}", {"snapshot": chosen, "adapter": str(path), "jevbench_acc": d.slice_mean_accuracy(jev),
                           "custom_acc": d.slice_mean_accuracy(cust), "custom_vs_untuned": reg})
    status(f"[{tag}] TEST jevbench {d.slice_mean_accuracy(jev):.3f}, custom {d.slice_mean_accuracy(cust):.3f}, "
           f"custom vs untuned {reg['accuracy_change'] * 100:+.2f} pts, within margin: {reg['non_inferior']}")


def run_arm(tag: str, model: str) -> None:
    status(f"==== arm {tag} ({model}) ====")
    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*"])
    out = BASE / f"adapter_{tag}"
    rc = train(model, out, ARM_MINUTES.get(tag, 45), f"train_{tag}.log")
    snaps = snapshots(out, steps_run(f"train_{tag}.log", FULL_STEPS))
    status(f"[{tag}] train exit {rc}, {time.time() - t0:.0f}s, snapshots {list(snaps)}")
    record(f"train_{tag}", {"model": model, "exit": rc, "snapshots": list(snaps)})
    if not snaps:
        status(f"[{tag}] no snapshots; skipping")
        return
    best, report = dev_and_select(tag, model, snaps)
    fallback = None
    if best is None and report:  # pre-registered: test the least-regressing snapshot with complete dev scores
        complete = [r for r in report if r["coverage"] >= rules().MIN_COVERAGE]
        fallback = max(complete, key=lambda r: (r["custom_change"], r["jev_acc"]))["name"] if complete else None
    record(f"select_{tag}", {"best": best, "fallback_no_eligible": fallback, "report": report})
    chosen = best or fallback
    status(f"[{tag}] chosen: {chosen}" + (" (no snapshot met the custom-dev margin)" if best is None else ""))
    if chosen:
        test_once(tag, model, chosen, {f"{tag}-{k}": p for k, p in snaps.items()}[chosen])


def pack() -> None:
    adapters = " ".join(str(Path(decisions[k]["adapter"]).relative_to(BASE))
                        for k in decisions if k.startswith("test_") and decisions[k].get("adapter"))
    rc = subprocess.run(f"cd {W} && tar czf {BASE}/gemma3_out.tgz results/raw -C {BASE} decisions.json {adapters} "
                        f"$(cd {BASE} && ls *.log)", shell=True).returncode
    status(f"packed gemma3_out.tgz exit {rc}, {(BASE / 'gemma3_out.tgz').stat().st_size / 1e6:.0f} MB")


def main() -> None:
    setup()
    for tag, model in ARMS:
        try:
            run_arm(tag, model)
        except Exception as exc:  # one arm failing must not lose the other's results
            status(f"[{tag}] ABORTED: {exc!r}"[:300])


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:
    status(f"ABORTED: {exc!r}"[:300])
try:
    pack()
except Exception as exc:
    status(f"pack failed: {exc!r}"[:200])
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("F_HOLD_MIN", 10)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
