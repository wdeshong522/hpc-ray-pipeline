# Schedule

Last updated: Wed 2026-11-25, evening. Build target Sun 11/29. Presentation
Thu 12/3. Scope is `PROPOSAL.md`; changes go in `DECISIONS.md`.

## Status

| Piece | State |
|---|---|
| Repo, proposal, decisions log | done (public repo on GitHub, cloned on Wahab) |
| Timing module (`src/common/timing.py`) | done, tested |
| Container image (`wdeshong/hpc-ray-pipeline:0.1`) | done, validated on a V100 |
| 8-bit judging | decided: not possible on V100, run the pipeline-parallel case (D19) |
| Two-node network check (InfiniBand) | done: NCCL uses IB, 3.93 GB/s |
| Training data module (`src/train/data.py`) | done, tested, corpus verified (4,000 usable records) |
| Model module, trainer, Ray launcher | done |
| Two-node Ray cluster inside the container | done (1, 2, 4, 8 GPUs) |
| Stage 1 scaling grid (7 configs x 3 repeats) | done Wed 11/25; strong 100/97/90% at 2/4/8 GPUs, weak 95-98% |
| Llama PP2 + LoRA serving test (one replica) | works; see D1 and D31 for the open questions it raised |
| Generation stage, judging stage, driver | not started |
| Training cost (Llama, measured) | measured: about 61 min of processing for 4,000 examples on 1 GPU |
| Per-user GPU cap (14) | open: admin email, decision Fri 11/27 (D28) |

## Day by day

| Day | Build and review | Runs on Wahab |
|---|---|---|
| Tue 11/24 (evening) | Model module; trainer; Ray launcher | End-of-turn label check; two-node Ray cluster test |
| Wed 11/25 | Generation script (PP2 replica servers, sharded client) | 1-GPU Llama smoke test; LoRA pilots at 1, 2, 4, 8 GPUs on a small subset (measures training cost); PP2 + LoRA serving test |
| Thu 11/26 (Thanksgiving) | Judging script (Qwen and Granite, 6-GPU units) | LoRA scaling runs start (unattended); generation pilot |
| Fri 11/27 | Driver (SLURM dependency chain); analysis scripts | Generation scaling runs; judging pilot |
| Sat 11/28 | Fix failures; analysis scripts | Judging scaling runs; full-pipeline pilot and runs; 24-GPU judging point in a quiet window |
| Sun 11/29 | Analysis: tables and plots from raw data | Last repeats only |
| Mon 11/30 | Reserve for spilled runs; finish analysis | Reserve |
| Tue 12/1 to Thu 12/3 | Slides, rehearsal | None |

The 11/29 build target has no slack left. Monday 11/30 is the real buffer.

## Gates

- **End of Wed 11/25:** Stage 1 runs at 1, 2, 4, 8 GPUs and the training cost is measured. If not, decide cuts that night, not later.
- **End of Fri 11/27:** every stage has run at least once at small scale. If not, drop full-pipeline strong scaling first.

## Cut order if time runs short

1. Repeats beyond 3 at the largest points.
2. Full-pipeline strong scaling.
3. Full-pipeline weak scaling.
4. Smaller strong-scaling workload sizes (the proposal fixes sizes by pilot).

Protected: stage-level strong and weak scaling for LoRA, generation and the
one judging case, with at least 3 repeats where possible.

## Risks

- **Training cost** may exceed the Granite-based estimate; measured Wednesday.
- **Review pace**: each module is walked through before the next starts.
- **Queue waits**: the 24-GPU judging point needs 6 idle nodes at once.
- **Run count**: about 100 runs including repeats, many with multi-minute model loads.
