"""Judge prompt templates, JSON schemas, and score parsing."""
import json
from pathlib import Path
from string import Template

METRICS = ("correctness", "completeness", "relevance", "faithfulness", "hallucination")
REASONING_MAX_CHARS = 600


def load_prompts(prompts_dir, metrics=METRICS):
    """Read one template file per metric: {metric: {template, requires_ground_truth, score_type}}."""
    registry = {}
    for name in metrics:
        text = (Path(prompts_dir) / f"{name}.txt").read_text(encoding="utf-8")
        _, front, body = text.split("---", 2)
        meta = dict(line.split(":", 1) for line in front.strip().splitlines())
        meta = {k.strip(): v.strip() for k, v in meta.items()}
        registry[name] = {"template": Template(body.strip()),
                          "requires_ground_truth": meta.get("requires_ground_truth") == "true",
                          "score_type": meta.get("score_type", "continuous")}
    return registry


def split_record(record):
    """Scenario is the user turns, ground truth is the first assistant turn; system turns are left out."""
    scenario, truth = [], ""
    for turn in record["conversations"]:
        role = turn.get("from", "").lower()
        value = turn.get("value", "").strip()
        if role in ("human", "user"):
            scenario.append(value)
        elif role in ("gpt", "assistant") and not truth:
            truth = value
    return "\n\n".join(scenario), truth


def response_instruction(score_type):
    if score_type == "binary":
        return ('\n\nRespond with only a JSON object:\n'
                '{"reasoning": "<brief explanation>", "score": 0} if no issue is present, or\n'
                '{"reasoning": "<brief explanation>", "score": 1} if a meaningful issue is present.')
    return ('\n\nRespond with only a JSON object:\n'
            '{"reasoning": "<brief explanation>", "score": <value between 0.0 and 1.0>}')


def build_prompt(entry, scenario, answer, ground_truth):
    body = entry["template"].safe_substitute(scenario=scenario, answer=answer,
                                             ground_truth=ground_truth)
    return body.strip() + response_instruction(entry["score_type"])


def score_schema(score_type):
    score = ({"type": "integer", "enum": [0, 1]} if score_type == "binary"
             else {"type": "number", "minimum": 0.0, "maximum": 1.0})
    return {"type": "object",
            "properties": {"reasoning": {"type": "string", "maxLength": REASONING_MAX_CHARS},
                           "score": score},
            "required": ["reasoning", "score"], "additionalProperties": False}


def parse_score(content, score_type):
    """Return (score, reasoning) or None when the reply is not a valid score object."""
    try:
        obj = json.loads(content)
        score = float(obj["score"])
    except (ValueError, KeyError, TypeError):
        return None
    if score_type == "binary":
        score = int(round(score))
        if score not in (0, 1):
            return None
    elif not 0.0 <= score <= 1.0:
        return None
    return score, obj.get("reasoning", "")
