"""
Metric extraction from paper text — Phase 24.

The Novelty Check compares the user's own numbers against the numbers the
retrieved papers REPORT. That comparison needs STRUCTURED metrics out of
free-form PDF text:

    "Our method reaches 34.8 BLEU-4 on the WMT14 test set."
        -> {"name": "BLEU-4", "value": 34.8}

Two shapes are recognised, matching how papers actually write:

  * "METRIC = 34.8" / "METRIC: 94.2%"      (named-value)
  * "34.8 BLEU-4" / "94.2% Accuracy"       (value-then-name)

Only metrics from a KNOWN vocabulary are extracted. This is deliberate: a
bare number with no metric name ("we trained for 100 epochs") is not a
metric, and treating it as one would produce nonsense comparisons. The
vocabulary is broad enough for the common benchmark families and is
extensible without touching the matching logic.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from src.novelty.scoring import normalize_metric_name, parse_numeric

# ---------------------------------------------------------------------------
# Metric vocabulary
# ---------------------------------------------------------------------------

# Base metric families. A variant suffix ("-4", "-L", "-2") is allowed by
# the regex, so "BLEU" covers BLEU-1..BLEU-4 and "ROUGE" covers ROUGE-L.
_KNOWN_METRIC_FAMILIES: list[str] = [
    # Translation / summarisation
    "BLEU",
    "ROUGE",
    "METEOR",
    "chrF",
    "BERTScore",
    "CIDEr",
    # Classification
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "F1-score",
    "AUC",
    "AUROC",
    "AP",
    "mAP",
    "EM",
    "Exact Match",
    # Language modelling
    "Perplexity",
    "PPL",
    "Bits Per Byte",
    "BPB",
    "LogLoss",
    "Cross-Entropy",
    # Generation quality
    "FID",
    "IS",
    "CLIPScore",
    "BLEURT",
    # Retrieval / IR
    "NDCG",
    "MRR",
    "Recall@K",
    "Precision@K",
    "MAP",
    "HitRate",
    # Other
    "Throughput",
    "Latency",
    "Error Rate",
    "WER",
    "CER",
    "IoU",
    "mIoU",
    "Dice",
    "PSNR",
    "SSIM",
    "MAE",
    "RMSE",
    "R2",
]

# Deliberately EXCLUDED from the vocabulary: training hyperparameters like
# "Epochs", "Steps", "Samples", "Parameters". They are numbers in a paper,
# but they describe the TRAINING RECIPE, not the RESULT — and treating
# "we trained for 100 epochs" as a metric would manufacture a bogus
# comparison against a user's own metric list.

# How many values to keep per metric name when extracting from a document.
_MAX_PER_METRIC = 3

# A year must never be read as a metric value.
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")

# Metric-name token pattern: letters/digits/hyphens, allowing the variant
# suffix. Requires at least 2 chars and must start with a letter.
_METRIC_TOKEN = r"[A-Za-z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*"

# "NAME = 34.8" / "NAME: 94.2%" / "NAME of 34.8"
_NAMED_VALUE_RE = re.compile(
    rf"(?P<name>{_METRIC_TOKEN})\s*(?:=|:|\bof\b)\s*"
    r"(?P<value>[-+]?\d*\.?\d+(?:\s?%|\s?[kKmMbB])?)"
)

# "34.8 BLEU-4" / "94.2% Accuracy" — value FIRST, then the metric name.
_VALUE_FIRST_RE = re.compile(
    r"(?P<value>[-+]?\d*\.?\d+(?:\s?%)?)\s*"
    rf"(?P<name>{_METRIC_TOKEN})"
)

# A number with a metric name on EITHER side, tolerating intervening words.
# "34.8 points BLEU-4" / "BLEU-4 of 34.8"
#
# The name group is ATOMIC ((?>...)). Without it, the engine backtracks:
# "BLEU-4" matches the base "BLEU", then the optional variant group gives
# back the "-4" so the following `\W{0,12}?` can consume it — yielding the
# nonsense metric "BLEU" with value "-4". The atomic group forbids that
# backtrack, so the variant stays attached to its name.
_LOOSE_RE = re.compile(
    rf"(?P<name>(?>{_METRIC_TOKEN}))\W{{0,12}}?(?P<value>[-+]?\d*\.?\d+(?:\s?%)?)"
)


def known_metric_names() -> list[str]:
    """The metric vocabulary this extractor recognises."""
    return list(_KNOWN_METRIC_FAMILIES)


def _normalise_family(name: str) -> str:
    """Normalise a candidate name into the vocabulary's canonical spelling."""
    raw = (name or "").strip()
    if not raw:
        return ""
    # Match against the vocabulary case-insensitively, preserving the
    # vocabulary's own spelling (so "bleu-4" becomes "BLEU-4").
    low = raw.lower()
    for fam in _KNOWN_METRIC_FAMILIES:
        if fam.lower() == low:
            return fam
    # "BLEU4" / "BLEU 4" / "BLEU-4" all mean BLEU-4 -> match "BLEU" family
    # with the variant preserved.
    base = re.split(r"[-/\s]", low)[0]
    for fam in _KNOWN_METRIC_FAMILIES:
        if fam.lower().split()[0] == base:
            variant = re.sub(r"^[A-Za-z]+", "", raw[len(base):])
            if variant:
                return f"{fam}{variant}"
            return fam
    return ""


def _is_known(name: str) -> bool:
    """True when `name` belongs to the metric vocabulary."""
    return bool(_normalise_family(name))


def _clean_value(raw: str) -> float | None:
    """
    Parse a captured value, rejecting years and other non-metrics.

    "2024" is rejected outright — a year is not a metric. "94.2%" and
    "1.2k" parse normally (the k/m/b suffix is dropped).
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Strip magnitude suffix (k/m/b) before parsing.
    text = re.sub(r"\s*[kKmMbB]$", "", text).strip()
    if _YEAR_RE.match(text):
        return None
    return parse_numeric(text)


def extract_named_value(sentence: str) -> dict[str, Any] | None:
    """
    Extract a single "METRIC = value" / "METRIC: value" pair from a line.

    Returns None when the line has no known metric name or no number.
    """
    if not sentence:
        return None
    for m in _NAMED_VALUE_RE.finditer(sentence):
        name = _normalise_family(m.group("name"))
        if not name:
            continue
        value = _clean_value(m.group("value"))
        if value is None:
            continue
        return {"name": name, "value": value}
    return None


def extract_metrics_from_text(sentence: str) -> list[dict[str, Any]]:
    """
    Extract every known-metric value pair from ONE line of text.

    Recognises both the named-value and the value-first styles. A metric
    name that is not in the vocabulary is ignored — this keeps "we trained
    for 100 epochs on 8 GPUs" from producing fake metrics.
    """
    if not sentence:
        return []

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()

    def _add(raw_name: str, raw_value: str) -> None:
        name = _normalise_family(raw_name)
        if not name:
            return
        value = _clean_value(raw_value)
        if value is None:
            return
        key = (name, value)
        if key in seen:
            return
        seen.add(key)
        out.append({"name": name, "value": value})

    # Named-value first: "BLEU-4 = 34.8".
    for m in _NAMED_VALUE_RE.finditer(sentence):
        _add(m.group("name"), m.group("value"))

    # Value-first: "34.8 BLEU-4".
    for m in _VALUE_FIRST_RE.finditer(sentence):
        _add(m.group("name"), m.group("value"))

    # Loose: "BLEU-4 of 34.8" / "BLEU-4 at 34.8".
    for m in _LOOSE_RE.finditer(sentence):
        _add(m.group("name"), m.group("value"))

    return out


def extract_metrics(text: str, *, max_per_metric: int = _MAX_PER_METRIC) -> list[dict[str, Any]]:
    """
    Extract every reported metric from a document.

    Returns a list of {"name", "value", "source"} dicts.

    Semantics
    ---------
    The FIRST value seen for a metric name wins and later values for the
    SAME name are dropped: in a paper the headline number is almost always
    the first mention, and subsequent mentions are ablations
    ("our method reaches 34.8 BLEU-4" then "an ablation reports 33.1
    BLEU-4"). Comparing a user's result against the ablation would be
    wrong.

    `max_per_metric` bounds how many DISTINCT metric names are kept, so a
    pathological document cannot flood the report.

    `source` records the line the value came from, so the report can point
    back at the evidence.
    """
    if not text:
        return []

    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for m in extract_metrics_from_text(line):
            name = m["name"]
            if name in seen_names:
                continue
            if len(out) >= max_per_metric * _MAX_PER_METRIC:
                break
            seen_names.add(name)
            out.append({**m, "source": line[:300]})
        if len(out) >= max_per_metric * _MAX_PER_METRIC:
            break

    return out


def metrics_by_name(metrics: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    """Group extracted metrics by their normalised name."""
    grouped: dict[str, list[float]] = {}
    for m in metrics or []:
        name = m.get("name", "")
        value = m.get("value")
        if not name or value is None:
            continue
        grouped.setdefault(normalize_metric_name(name), []).append(float(value))
    return grouped
