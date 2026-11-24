"""Per-GPU training worker: the function Ray Train runs once per GPU.

Proposal link: Stage 1 is Ray Train data-parallel LoRA fine-tuning.  Timing
follows DECISIONS D24: startup is everything before the first training step,
processing is trainer.train() only.  One timing file per worker (tag=rankN).
"""
FIXED = {
    "epochs": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lr": 3e-4,
    "max_seq_len": 2048,
}

TUNABLE = {
    "per_device_batch": 1,
    "grad_accum": 8,
}

REQUIRED = ("model_name", "data_dir", "n_examples", "output_dir",
            "results_dir", "run_id")


def with_defaults(cfg):
    """Fill defaults; the proposal's tunables may be overridden, FIXED not."""
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    overridden = [k for k in FIXED if k in cfg and cfg[k] != FIXED[k]]
    if overridden:
        raise ValueError(f"fixed settings cannot be overridden: {overridden}")
    return {**FIXED, **TUNABLE, **cfg}


def sft_kwargs(cfg):
    """Trainer settings: no eval, no checkpoints, no reporting (D23)."""
    return dict(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        learning_rate=cfg["lr"],
        weight_decay=0.001,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        packing=False,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_len"],
        seed=42,
    )


def train_worker(user_cfg):
    import ray.train
    import ray.train.torch
    from trl import SFTConfig, SFTTrainer

    from src.common.timing import RunRecorder
    from src.train.data import build_dataset, load_records, ordered_pool, take
    from src.train.model import add_lora, load_base_model, load_tokenizer

    cfg = with_defaults(user_cfg)
    rank = ray.train.get_context().get_world_rank()
    device = ray.train.torch.get_device()
    rec = RunRecorder(
        cfg["results_dir"], cfg["run_id"], "lora", tag=f"rank{rank}",
        params={
            "world_size": ray.train.get_context().get_world_size(),
            "n_examples": cfg["n_examples"],
            "per_device_batch": cfg["per_device_batch"],
            "grad_accum": cfg["grad_accum"],
        })

    with rec.interval("startup"):
        tokenizer = load_tokenizer(cfg["model_name"])
        records, problems = load_records(cfg["data_dir"])
        dataset = build_dataset(
            take(ordered_pool(records), cfg["n_examples"]), tokenizer)
        model = load_base_model(cfg["model_name"], device.index)
        model = add_lora(model, cfg["lora_r"], cfg["lora_alpha"],
                         cfg["lora_dropout"])
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(**sft_kwargs(cfg)),
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    if rank == 0:
        print(f"examples={len(dataset)} skipped_lines={len(problems)}")
        model.print_trainable_parameters()

    with rec.interval("processing"):
        result = trainer.train()

    if rank == 0:
        trainer.save_model(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])

    ray.train.report({"train_loss": float(result.training_loss)})
