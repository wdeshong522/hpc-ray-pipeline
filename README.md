# Hybrid Parallel Dental Hygiene Feedback Pipeline

HPC class project: a three-stage pipeline (LoRA fine-tuning with Ray Train,
response generation with vLLM pipeline parallelism, judging by two small LLM
judges) scaled across GPU nodes on ODU's Wahab cluster.

- Scope: see [`PROPOSAL.md`](PROPOSAL.md) (frozen as submitted).
- Deviations and open decisions: see [`DECISIONS.md`](DECISIONS.md).
- Data is **not** included in this repository. See `data/README.md` (to be added).
- Results are synthetic-data timing experiments, not clinical or model-quality
  evaluations.

Status: under construction.
