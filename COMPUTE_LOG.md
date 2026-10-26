# Compute log

Every paid Colab run for s1-embedded, logged before it starts and after it ends. Free local runs (the pre-flight
checks, smoke trains and the watcher dry-run) are not logged here.

Balance at the start of s1-embedded work: 374.63 units (2026-10-26). The teacher labelling is reused from the parent
project (j-raghavan/s1-decision-data), so no labelling run is needed here.

| Date | Job | GPU | Est. units | Before | After | Used | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-26 | Dry run: session_gemma3.py at tiny scale (F_FULL_STEPS=20, F_ARM_MINUTES=2, F_LIMIT=40) over both arms, to validate the eval wiring before the full run | A100 40GB | ~1 | 374.63 | 373.92 | 0.71 | Session s1e-dry 05:14–05:20 UTC. **PASS**: both arms ran end to end (setup → train → dev scoring → choose_snapshot_ni → test), all steps exit 0; the eligible (270m) and fallback (1b) selection branches both exercised. Accuracy numbers are noise at 20 steps / 40 cases. 7 min wall |
| 2026-10-26 | Full two-arm run: session_gemma3.py, 1B + 270M | A100 40GB | ~5–7 | 373.92 | 367.73 | 6.19 | Session s1e-dry (reused the dry-run VM). **INVALID — discarded, nothing published.** `colab exec --env` sets persistent kernel env vars, and reusing the dry-run's kernel leaked F_LIMIT=40 and F_BATCH=4 into this run: every predictions file holds 40 rows (eval ran on 40 cases, not the full suites) and training ran at batch 4 (~0.25 epoch), so all reported accuracies are noise. VM stopped. Fix for the re-run: restart the kernel first and pass F_BATCH and an empty F_LIMIT explicitly |
