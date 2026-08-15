"""Word-level mispronunciation detection on speechocean762: tune and report the phone rule.

    # 1. run the phone recognizer once and cache its frame posteriors (~5 min on CPU)
    python benchmarks/word_detection.py --extract --sample 500

    # 2. grid search of the thresholds on the tuning half (even utterances)
    python benchmarks/word_detection.py --tune

    # 3. precision / recall of the current rule on both halves (report in benchmarks/README.md)
    python benchmarks/word_detection.py --report

    # 4. flagged words on the bundled samples (assets/), as a regression check
    python benchmarks/word_detection.py --assets

The posteriors are cached in ``~/.cache/openpronounce/speechocean762/logits`` (float16,
one ``.npy`` per utterance, plus ``labels.json`` and ``vocab.json``), so steps 2-4 do not
need the model. A word is "mispronounced" for the raters when its accuracy is below 5
(strict) or 7 (lenient); both are reported. The sample is split by parity of the
utterance index: even for tuning, odd held out.
"""

import argparse
import io
import itertools
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speechocean762 import DATASET_DIR, load_rows, sample_rows  # noqa: E402

LOGITS_DIR = os.path.join(DATASET_DIR, "logits")
LABEL_THRESHOLDS = (5, 7)
ASSETS = [
    ("assets/developer.wav", "hello I am a developer"),
    ("assets/example.mp3", "Hello, how are you?"),
    ("assets/harvard.wav", "assets/harvard_text.txt"),
    ("assets/developer1.wav", "hello I am a developer"),
]

logger = logging.getLogger("word_detection")


# ---------------------------------------------------------------------------
# Posteriors cache
# ---------------------------------------------------------------------------

def extract(args):
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    from openpronounce import audio, phones

    rows = sample_rows(load_rows(args.split), args.sample, seed=args.seed)
    os.makedirs(LOGITS_DIR, exist_ok=True)
    labels = [{
        "utt": r["utt"], "speaker": r["speaker"], "text": r["text"], "total": r["total"], "accuracy": r["accuracy"],
        "words": [{"text": w["text"], "accuracy": w["accuracy"]} for w in r["words"]],
    } for r in rows]
    with open(os.path.join(LOGITS_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)
    with open(os.path.join(LOGITS_DIR, "vocab.json"), "w") as f:
        json.dump(list(phones.phone_vocab()), f)
    for k, row in enumerate(rows, 1):
        path = os.path.join(LOGITS_DIR, row["utt"] + ".npy")
        if os.path.exists(path):
            continue
        sound = audio.load(io.BytesIO(row["audio"]["bytes"]))
        np.save(path, phones.phone_log_posteriors(sound).astype(np.float16))
        logger.info("[%d/%d] %s", k, len(rows), row["utt"])


def load_cache():
    """Return ``(labels, vocab)`` and check that every utterance has its posteriors."""
    with open(os.path.join(LOGITS_DIR, "labels.json")) as f:
        labels = json.load(f)
    with open(os.path.join(LOGITS_DIR, "vocab.json")) as f:
        vocab = tuple(json.load(f))
    missing = [lab["utt"] for lab in labels if not os.path.exists(os.path.join(LOGITS_DIR, lab["utt"] + ".npy"))]
    if missing:
        raise SystemExit(f"{len(missing)} utterances without cached posteriors, run --extract first")
    return labels, vocab


def word_rows(labels, vocab, use_posteriors=True):
    """One dict per word: the report of ``phones._word_reports`` plus ``bad<5``/``bad<7`` and ``tune``."""
    import numpy as np

    from openpronounce import phones

    rows = []
    for k, lab in enumerate(labels):
        log_posteriors = np.load(os.path.join(LOGITS_DIR, lab["utt"] + ".npy")).astype(np.float32)
        recognition = phones.decode_ctc(log_posteriors, vocab)
        heard = recognition if use_posteriors else recognition.phones
        for report in phones._word_reports(heard, lab["text"].lower()):
            accuracy = lab["words"][report["position"]]["accuracy"]
            for threshold in LABEL_THRESHOLDS:
                report[f"bad<{threshold}"] = accuracy < threshold
            report["tune"] = k % 2 == 0
            rows.append(report)
    return rows


# ---------------------------------------------------------------------------
# Rules and metrics
# ---------------------------------------------------------------------------

def rule(threshold, min_edits, key="weighted_edits"):
    """Flag a word when ``key`` reaches ``threshold`` of its phones or ``min_edits``."""
    def flag(row):
        edits = row[key]
        return row["distance"] > 0 and (edits / len(row["expected"]) >= threshold or edits >= min_edits)
    return flag


def metrics(rows, flag, label):
    tp = sum(1 for r in rows if flag(r) and r[label])
    fp = sum(1 for r in rows if flag(r) and not r[label])
    fn = sum(1 for r in rows if not flag(r) and r[label])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "flagged": (tp + fp) / len(rows)}


GRID = list(itertools.product([t / 20 for t in range(4, 25)], [1.5, 2, 2.5, 3, 4, 99]))


def precision_at_recall(rows, key, label, target):
    """Best precision over the grid of thresholds among operating points with recall >= target."""
    best = 0.0
    for threshold, min_edits in GRID:
        m = metrics(rows, rule(threshold, min_edits, key), label)
        if m["recall"] >= target:
            best = max(best, m["precision"])
    return best


def tune(args):
    labels, vocab = load_cache()
    rows = [r for r in word_rows(labels, vocab) if r["tune"]]
    label = f"bad<{args.label}"
    results = sorted(((metrics(rows, rule(t, m), label), t, m) for t, m in GRID), key=lambda x: -x[0]["f1"])
    print(f"Tuning half: {len(rows)} words, {sum(r[label] for r in rows)} with human accuracy < {args.label}")
    print(f"{'threshold':>9} {'min_edits':>9} {'F1':>6} {'P':>6} {'R':>6} {'flagged':>8}")
    for m, t, me in results[:15]:
        print(f"{t:9.2f} {me:9} {m['f1']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['flagged']:8.1%}")


def report(args):
    from openpronounce import phones

    labels, vocab = load_cache()
    with_posteriors = word_rows(labels, vocab)
    without = word_rows(labels, vocab, use_posteriors=False)
    current = rule(phones.PHONE_ERROR_THRESHOLD, phones.PHONE_ERROR_MIN_EDITS)
    systems = [
        ("0.2.1 rule (edit distance >= 50 % or >= 3)", without, rule(0.5, 3, "distance"), "distance"),
        ("confidence rule, no posteriors", without, current, "weighted_edits"),
        ("confidence rule + posteriors (default)", with_posteriors, current, "weighted_edits"),
    ]
    n_tune = sum(1 for r in with_posteriors if r["tune"])
    print(f"{len(labels)} utterances, {len(with_posteriors)} words ({n_tune} in the tuning half); "
          f"rule: weighted edits >= {phones.PHONE_ERROR_THRESHOLD:.0%} of the phones or >= {phones.PHONE_ERROR_MIN_EDITS}")
    for threshold in LABEL_THRESHOLDS:
        label = f"bad<{threshold}"
        n_bad = sum(r[label] for r in with_posteriors)
        print(f"\nHuman word accuracy < {threshold} = mispronounced ({n_bad} words, {n_bad / len(with_posteriors):.1%})")
        print(f"{'system':44} {'split':8} {'P':>6} {'R':>6} {'F1':>6} {'flagged':>8} {'P@R.5':>6} {'P@R.7':>6}")
        for name, rows, flag, key in systems:
            for split, keep in (("tune", True), ("held-out", False)):
                subset = [r for r in rows if r["tune"] == keep]
                m = metrics(subset, flag, label)
                print(f"{name:44} {split:8} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['flagged']:8.1%} "
                      f"{precision_at_recall(subset, key, label, 0.5):6.3f} {precision_at_recall(subset, key, label, 0.7):6.3f}")


def assets(args):
    import torch

    torch.set_num_threads(args.threads)
    from openpronounce import audio, phones

    for path, text in ASSETS:
        if os.path.exists(text):
            with open(text) as f:
                text = f.read()
        recognition = phones.recognize_phones(audio.load(path))
        result = phones.compare_phones(recognition, text)
        flagged = ", ".join(f"{e['word']} ({e['confidence']:.2f})" for e in result["errors"]) or "-"
        print(f"{path}: {flagged}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", action="store_true", help="cache the phone posteriors of the sample")
    mode.add_argument("--tune", action="store_true", help="grid search on the tuning half")
    mode.add_argument("--report", action="store_true", help="precision/recall table on both halves")
    mode.add_argument("--assets", action="store_true", help="flagged words on the bundled samples")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", type=int, default=5, choices=LABEL_THRESHOLDS,
                        help="human accuracy below which a word is mispronounced, for --tune")
    parser.add_argument("--threads", type=int, default=6, help="torch threads")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.extract:
        extract(args)
    elif args.tune:
        tune(args)
    elif args.report:
        report(args)
    else:
        assets(args)


if __name__ == "__main__":
    main()
