"""Benchmark OpenPronounce against human scores from speechocean762.

Two modes:

    # run inference on a fixed random sample of the test split (resumable)
    python benchmarks/speechocean762.py --sample 500 --out benchmarks/results/speechocean762.csv

    # analyse an existing CSV: correlations, word-level precision/recall, grid search + CV
    python benchmarks/speechocean762.py --report --out benchmarks/results/speechocean762.csv

The dataset (parquet, with audio) is downloaded from the Hugging Face hub
(``mispeech/speechocean762``) into ``~/.cache/openpronounce/speechocean762``.
"""

import argparse
import csv
import io
import itertools
import logging
import os
import random
import sys
import time
from collections import defaultdict

DATASET_REPO = "mispeech/speechocean762"
DATASET_DIR = os.path.join(os.path.expanduser("~"), ".cache", "openpronounce", "speechocean762")
PARQUET = {"test": "data/test-00000-of-00001.parquet", "train": "data/train-00000-of-00001.parquet"}

# A word is "mispronounced" for the human raters when its accuracy is below this (0-10 scale).
HUMAN_BAD_WORD = 5

FIELDS = [
    "utt", "speaker", "text",
    "human_total", "human_accuracy", "human_fluency", "human_prosodic",
    "score", "acoustic_distance", "phoneme_error_rate", "word_error_rate",
    "n_words", "n_flagged", "n_human_bad", "n_hits",
    "word_recall", "word_precision", "wall_time",
]

logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download(split="test"):
    from huggingface_hub import hf_hub_download

    os.makedirs(DATASET_DIR, exist_ok=True)
    return hf_hub_download(DATASET_REPO, PARQUET[split], repo_type="dataset", local_dir=DATASET_DIR)


def load_rows(split="test"):
    """Return the list of utterance dicts (audio bytes included) of ``split``."""
    import pyarrow.parquet as pq

    path = download(split)
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["utt"] = os.path.splitext(row["audio"]["path"])[0]
    return rows


def sample_rows(rows, n, seed=0):
    """Random sample of ``n`` rows, stratified on the human ``total`` score (proportional allocation)."""
    if n >= len(rows):
        return sorted(rows, key=lambda r: r["utt"])
    rng = random.Random(seed)
    by_total = defaultdict(list)
    for row in rows:
        by_total[row["total"]].append(row)
    picked = []
    # Largest-remainder allocation so that the strata sum to exactly n.
    quotas = {t: n * len(g) / len(rows) for t, g in by_total.items()}
    alloc = {t: int(q) for t, q in quotas.items()}
    for t, _ in sorted(quotas.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[: n - sum(alloc.values())]:
        alloc[t] += 1
    for t in sorted(by_total):
        group = sorted(by_total[t], key=lambda r: r["utt"])
        picked.extend(rng.sample(group, alloc[t]))
    return sorted(picked, key=lambda r: r["utt"])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def read_done(out):
    if not os.path.exists(out):
        return set()
    with open(out, newline="") as f:
        return {r["utt"] for r in csv.DictReader(f)}


def text2speech_with_retry(text, attempts=8):
    """gTTS goes over the network and may answer 429; back off and retry."""
    from openpronounce import audio

    delay = 5
    for attempt in range(attempts):
        try:
            return audio.text2speech(text)
        except Exception as e:  # noqa: BLE001
            logger.warning("gTTS failed for %r (%s), retry %d/%d in %ds", text, e, attempt + 1, attempts, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"gTTS failed for {text!r}")


def normalize_text(text):
    """speechocean762 texts are upper case; give the TTS a normally cased sentence."""
    text = text.strip().lower()
    return text[:1].upper() + text[1:]


def word_metrics(row, errors):
    words = [w["text"].lower() for w in row["words"]]
    human_bad = {i for i, w in enumerate(row["words"]) if w["accuracy"] < HUMAN_BAD_WORD}
    flagged_words = {e["word"].lower() for e in errors}
    flagged = {i for i, w in enumerate(words) if w in flagged_words}
    hits = human_bad & flagged
    return {
        "n_words": len(words),
        "n_flagged": len(flagged),
        "n_human_bad": len(human_bad),
        "n_hits": len(hits),
        "word_recall": round(len(hits) / len(human_bad), 4) if human_bad else "",
        "word_precision": round(len(hits) / len(flagged), 4) if flagged else "",
    }


def run(args):
    import torch

    torch.set_num_threads(args.threads)
    from openpronounce import audio, speech

    rows = sample_rows(load_rows(args.split), args.sample, seed=args.seed)
    done = read_done(args.out)
    todo = [r for r in rows if r["utt"] not in done]
    logger.info("%d utterances in sample, %d already done, %d to run", len(rows), len(done), len(todo))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    failures = []
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for k, row in enumerate(todo, 1):
            text = normalize_text(row["text"])
            t0 = time.time()
            try:
                text2speech_with_retry(text)
                sound = audio.load(io.BytesIO(row["audio"]["bytes"]))
                result = speech.compare_audio_with_text(sound, text)
            except Exception as e:  # noqa: BLE001
                logger.error("utt %s failed: %s", row["utt"], e)
                failures.append(row["utt"])
                continue
            diff = result["differences"]
            out = {
                "utt": row["utt"],
                "speaker": row["speaker"],
                "text": row["text"],
                "human_total": row["total"],
                "human_accuracy": row["accuracy"],
                "human_fluency": row["fluency"],
                "human_prosodic": row["prosodic"],
                "score": result["score"],
                "acoustic_distance": result["acoustic_distance"],
                "phoneme_error_rate": diff["phoneme_error_rate"],
                "word_error_rate": diff["word_error_rate"],
                **word_metrics(row, diff["errors"]),
                "wall_time": round(time.time() - t0, 2),
            }
            writer.writerow(out)
            f.flush()
            logger.info("[%d/%d] %s human=%d score=%.1f (%.1fs)", k, len(todo), row["utt"],
                        row["total"], result["score"], out["wall_time"])
    if failures:
        logger.warning("%d utterances failed: %s", len(failures), " ".join(failures))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FIELDS:
            if k in ("utt", "speaker", "text"):
                continue
            r[k] = float(r[k]) if r[k] != "" else None
    return rows


def score_from_components(ad, per, wer, good, bad, w_ac, w_ph, w_wo):
    """Vectorised copy of ``speech.compute_pronunciation_score`` (arrays in, array out)."""
    import numpy as np

    acoustic = np.clip(100 * (1 - (ad - good) / (bad - good)), 0, 100)
    phon = np.clip(100 * (1 - per), 0, 100)
    word = np.clip(100 * (1 - wer), 0, 100)
    return np.clip(w_ac * acoustic + w_ph * phon + w_wo * word, 0, 100)


# Grid of the score constants explored by ``--report``: weights on a 0.1 step (summing to
# 1), acoustic bounds GOOD in 3..8 and BAD in 12..22.
GRID_WEIGHTS = [(a / 10, p / 10, round((10 - a - p) / 10, 1)) for a in range(11) for p in range(11 - a)]
GRID_GOOD = [float(g) for g in range(3, 9)]
GRID_BAD = [float(b) for b in range(12, 23)]
GRID = [(w, g, b) for w in GRID_WEIGHTS for g in GRID_GOOD for b in GRID_BAD
        if w[0] > 0 or (g, b) == (GRID_GOOD[0], GRID_BAD[0])]  # bounds irrelevant without acoustic weight

# Combinations always reported by the cross-validation, next to the current defaults.
CV_CANDIDATES = [
    ((0.2, 0.5, 0.3), 5.0, 15.0),   # 0.2.1 defaults
    ((0.3, 0.4, 0.3), 6.0, 15.0),   # 0.3.0 defaults
    ((0.6, 0.2, 0.2), 5.0, 18.0),
    ((0.7, 0.2, 0.1), 5.0, 18.0),
    ((0.8, 0.1, 0.1), 5.0, 18.0),
    ((1.0, 0.0, 0.0), 5.0, 18.0),   # acoustic distance alone
]


def current_constants():
    from openpronounce import speech

    w = speech.SCORE_WEIGHTS
    return (w["acoustic"], w["phonemes"], w["words"]), speech.ACOUSTIC_DISTANCE_GOOD, speech.ACOUSTIC_DISTANCE_BAD


def components(rows):
    import numpy as np

    return tuple(np.array([r[k] for r in rows]) for k in ("acoustic_distance", "phoneme_error_rate", "word_error_rate"))


def grid_search(rows, human, top=8):
    import numpy as np
    from scipy.stats import spearmanr

    ad, per, wer = components(rows)

    def rho(combo, idx=slice(None)):
        (w_ac, w_ph, w_wo), good, bad = combo
        s = score_from_components(ad[idx], per[idx], wer[idx], good, bad, w_ac, w_ph, w_wo)
        return spearmanr(s, human[idx]).correlation

    results = sorted(((rho(c), c) for c in GRID), reverse=True)
    default = current_constants()

    print("\nGrid search (Spearman of recomputed score vs human total):")
    print(f"{'spearman':>9} {'w_acoustic':>10} {'w_phonemes':>10} {'w_words':>8} {'AD_GOOD':>8} {'AD_BAD':>7}")
    for r, ((w_ac, w_ph, w_wo), good, bad) in results[:top]:
        print(f"{r:9.4f} {w_ac:10.1f} {w_ph:10.1f} {w_wo:8.1f} {good:8.1f} {bad:7.1f}")
    (w_ac, w_ph, w_wo), good, bad = default
    print(f"{rho(default):9.4f} {w_ac:10.1f} {w_ph:10.1f} {w_wo:8.1f} {good:8.1f} {bad:7.1f}  <- current defaults")

    # Two-fold cross-validation, three random splits: the best combination of one half is
    # scored on the other half, next to fixed candidates.
    n = len(rows)
    folds = []
    for seed in range(3):
        perm = np.random.RandomState(seed).permutation(n)
        folds.append((perm[: n // 2], perm[n // 2:]))
        folds.append((perm[n // 2:], perm[: n // 2]))
    print("\nCross-validation (2 folds x 3 splits), held-out Spearman with human total:")
    tuned = [(rho(max(GRID, key=lambda c: rho(c, train)), test)) for train, test in folds]
    print(f"{'best of the training half':32} mean {np.mean(tuned):.3f}  folds " + " ".join(f"{v:.3f}" for v in tuned))
    for combo in [default] + [c for c in CV_CANDIDATES if c != default]:
        vals = [rho(combo, test) for _, test in folds]
        (w_ac, w_ph, w_wo), good, bad = combo
        name = f"{w_ac:.1f}/{w_ph:.1f}/{w_wo:.1f} bounds {good:.0f}/{bad:.0f}"
        print(f"{name:32} mean {np.mean(vals):.3f}  folds " + " ".join(f"{v:.3f}" for v in vals)
              + ("  <- current defaults" if combo == default else ""))
    return results


def report(args):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    rows = load_csv(args.out)
    n = len(rows)
    print(f"N = {n} utterances, {len({r['speaker'] for r in rows})} speakers, "
          f"mean wall time {np.mean([r['wall_time'] for r in rows]):.2f}s/utt")

    def corr(a, b):
        return pearsonr(a, b)[0], spearmanr(a, b).correlation

    score = np.array([r["score"] for r in rows])
    total = np.array([r["human_total"] for r in rows])
    accuracy = np.array([r["human_accuracy"] for r in rows])
    fluency = np.array([r["human_fluency"] for r in rows])
    prosodic = np.array([r["human_prosodic"] for r in rows])

    # The CSV stores the score computed at run time; when the constants have changed since,
    # also report the score recomputed from the stored components with the current ones.
    weights, good, bad = current_constants()
    recomputed = score_from_components(*components(rows), good, bad, *weights)
    scores = [("score", score)]
    if np.abs(recomputed - score).max() > 0.05:
        scores.append(("score (recomputed, current constants)", recomputed))

    print("\nHuman total: mean %.2f, std %.2f. Our score: mean %.1f, std %.1f"
          % (total.mean(), total.std(), score.mean(), score.std()))
    if len(scores) > 1:
        print("Recomputed score with %.1f/%.1f/%.1f, bounds %.0f/%.0f: mean %.1f, std %.1f"
              % (*weights, good, bad, recomputed.mean(), recomputed.std()))
    print("\nCorrelations (Pearson / Spearman):")
    print(f"{'pair':56} {'pearson':>8} {'spearman':>9}")
    for label, sc in scores:
        for name, h in (("total", total), ("accuracy", accuracy), ("fluency", fluency), ("prosodic", prosodic)):
            p, s = corr(sc, h)
            print(f"{label + ' vs human ' + name:56} {p:8.3f} {s:9.3f}")
    for comp in ("acoustic_distance", "phoneme_error_rate", "word_error_rate"):
        x = np.array([r[comp] for r in rows])
        for hname, h in (("accuracy", accuracy), ("total", total)):
            p, s = corr(x, h)
            print(f"{comp + ' vs human ' + hname:56} {p:8.3f} {s:9.3f}")

    # Utterances with more phone edits than expected phones. Before 0.3.0 this was the
    # signature of a phonemization failure (expected phones empty, PER = len(heard)/1, no
    # word flagged); it is now a handful of very noisy recordings.
    broken = [r for r in rows if r["phoneme_error_rate"] > 1]
    if broken:
        keep = np.array([r["phoneme_error_rate"] <= 1 for r in rows])
        print(f"\n{len(broken)} utterances have PER > 1. On the {int(keep.sum())} others:")
        for name, x in (("score", score), ("acoustic_distance", -np.array([r['acoustic_distance'] for r in rows])),
                        ("phoneme_error_rate", -np.array([r['phoneme_error_rate'] for r in rows])),
                        ("word_error_rate", -np.array([r['word_error_rate'] for r in rows]))):
            p, s = corr(x[keep], total[keep])
            print(f"  {name + ' vs human total':54} {p:8.3f} {s:9.3f}")

    # Word level
    n_bad = sum(r["n_human_bad"] for r in rows)
    n_flag = sum(r["n_flagged"] for r in rows)
    n_hits = sum(r["n_hits"] for r in rows)
    n_words = sum(r["n_words"] for r in rows)
    print(f"\nWord level (human accuracy < {HUMAN_BAD_WORD} = mispronounced): "
          f"{int(n_words)} words, {int(n_bad)} human-bad ({100 * n_bad / n_words:.1f}%), {int(n_flag)} flagged by us")
    print(f"  micro recall    = {n_hits / n_bad:.3f}" if n_bad else "  micro recall    = n/a")
    print(f"  micro precision = {n_hits / n_flag:.3f}" if n_flag else "  micro precision = n/a")
    rec = [r["word_recall"] for r in rows if r["word_recall"] is not None]
    prec = [r["word_precision"] for r in rows if r["word_precision"] is not None]
    print(f"  macro recall    = {np.mean(rec):.3f} (over {len(rec)} utterances with a human-bad word)")
    print(f"  macro precision = {np.mean(prec):.3f} (over {len(prec)} utterances with a flagged word)")
    tp, fp, fn = n_hits, n_flag - n_hits, n_bad - n_hits
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    print(f"  micro F1        = {f1:.3f}")
    if broken:
        ok = [r for r in rows if r["phoneme_error_rate"] <= 1]
        b, fl, h = (sum(r[k] for r in ok) for k in ("n_human_bad", "n_flagged", "n_hits"))
        print(f"  on the {len(ok)} utterances with PER <= 1: "
              f"recall = {h / b:.3f}, precision = {h / fl:.3f} ({int(b)} human-bad, {int(fl)} flagged)")

    # Score distribution by human total (sanity check of monotonicity)
    print("\nMean score by human total:")
    for t in sorted(set(total)):
        m = score[total == t]
        print(f"  total={int(t):2d}: n={len(m):3d} mean score={m.mean():5.1f} std={m.std():4.1f}")

    # Per speaker (at least 3 utterances): mean score vs mean human total
    speakers = defaultdict(list)
    for i, r in enumerate(rows):
        speakers[r["speaker"]].append(i)
    idx = [v for v in speakers.values() if len(v) >= 3]
    for label, sc in scores:
        s = spearmanr([sc[v].mean() for v in idx], [total[v].mean() for v in idx]).correlation
        print(f"Per speaker ({len(idx)} speakers with >= 3 utterances), {label}: Spearman {s:.3f}")

    grid_search(rows, total)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=500, help="number of utterances (default 500)")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmarks/results/speechocean762.csv")
    parser.add_argument("--threads", type=int, default=6, help="torch threads")
    parser.add_argument("--report", action="store_true", help="analyse the CSV instead of running inference")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    if args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
