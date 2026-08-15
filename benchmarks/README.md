# Benchmarks

## speechocean762

[speechocean762](https://www.openslr.org/101/) (Zhang et al., 2021, CC BY 4.0) is a
free corpus of 5,000 English utterances read by 250 Mandarin-speaking learners (half of
them children), each rated by five experts. Every utterance carries sentence-level
`accuracy`, `fluency`, `prosodic` and `total` scores (0-10), and every word carries an
`accuracy` score (0-10) plus phone-level scores. Sentences are short (2 to 12 words).

`benchmarks/speechocean762.py` runs `speech.compare_audio_with_text` on a fixed random
sample of the test split and compares the output with the human ratings.

### How to run

```bash
# inference (resumable: utterances already in the CSV are skipped)
HF_HOME=~/.cache/huggingface python benchmarks/speechocean762.py --sample 500 \
    --out benchmarks/results/speechocean762.csv

# analysis of the CSV (correlations, word-level precision/recall, grid search)
python benchmarks/speechocean762.py --report --out benchmarks/results/speechocean762.csv
```

The parquet file (`mispeech/speechocean762` on the Hugging Face hub, audio included) is
downloaded to `~/.cache/openpronounce/speechocean762/` on first run (~300 MB for the
test split). The sample is stratified on the human `total` score with seed 0, so
`--sample 500` always picks the same 500 utterances. `--threads` sets the number of
torch threads (default 6). gTTS is called once per distinct sentence (network); the
script retries with backoff on failure.

CSV columns: `utt`, `speaker`, `text`, human `total`/`accuracy`/`fluency`/`prosodic`,
our `score`, the three components (`acoustic_distance`, `phoneme_error_rate`,
`word_error_rate`), word counts (`n_words`, `n_flagged`, `n_human_bad`, `n_hits`),
per-utterance word `recall`/`precision` (a word is "bad" for the raters when its
accuracy is below 5) and `wall_time` in seconds.

### Results (2026-08-15)

- N = 500 utterances of the test split (123 speakers), OpenPronounce 0.2.1 (worktree of
  commit ea7f4df), `facebook/wav2vec2-large-960h` + `facebook/wav2vec2-lv-60-espeak-cv-ft`,
  default constants (`SCORE_WEIGHTS = {acoustic: 0.2, phonemes: 0.5, words: 0.3}`,
  `ACOUSTIC_DISTANCE_GOOD/BAD = 5/15`).
- CPU only, 6 torch threads on a 16-core desktop (other jobs running in parallel),
  torch 2.13, transformers 5.15: 3.5 s per utterance on average, 29 min for the run.
- No gTTS failure, no skipped utterance. Raw data: `results/speechocean762.csv`.

Human total: mean 7.26, sd 1.54 (the corpus is skewed toward good readings). Our
score: mean 49.8, sd 27.7.

Correlation with the human ratings:

| pair                                   | Pearson | Spearman |
|----------------------------------------|--------:|---------:|
| score vs human total                   |   0.571 |    0.583 |
| score vs human accuracy                |   0.535 |    0.553 |
| score vs human fluency                 |   0.532 |    0.515 |
| score vs human prosodic                |   0.552 |    0.541 |
| acoustic_distance vs human accuracy    |  -0.608 |   -0.642 |
| acoustic_distance vs human total       |  -0.639 |   -0.666 |
| phoneme_error_rate vs human accuracy   |  -0.018 |   -0.372 |
| phoneme_error_rate vs human total      |  -0.041 |   -0.409 |
| word_error_rate vs human accuracy      |  -0.503 |   -0.548 |
| word_error_rate vs human total         |  -0.519 |   -0.551 |

Mean score per human total: 3 -> 14, 4 -> 21, 5 -> 23, 6 -> 34, 7 -> 40, 8 -> 61,
9 -> 68 (monotonic, but the within-level standard deviation is 20-24 points).
Aggregated per speaker (106 speakers with at least 3 utterances), the Spearman
correlation between mean score and mean human total is 0.78.

Word level, "mispronounced" = human word accuracy < 5 (168 of 3,146 words, 5.3%):

| metric                              | value |
|-------------------------------------|------:|
| micro recall (bad words we flagged) | 0.696 |
| micro precision (flagged words that are bad) | 0.108 |
| micro F1                            | 0.188 |
| words flagged                       | 1,080 (34% of all words) |

Grid search on the score formula (recomputed from the stored components, Spearman with
human total; weights on a 0.1 grid, GOOD in {0..8}, BAD in {8..30}):

| Spearman | w_acoustic | w_phonemes | w_words | AD_GOOD | AD_BAD |
|---------:|-----------:|-----------:|--------:|--------:|-------:|
| 0.674 | 0.8 | 0.1 | 0.1 | 7 | 18 |
| 0.674 | 0.8 | 0.1 | 0.1 | 5 | 18 |
| 0.674 | 0.8 | 0.1 | 0.1 | 6 | 18 |
| 0.674 | 0.8 | 0.1 | 0.1 | 4 | 20 |
| 0.672 | 0.8 | 0.1 | 0.1 | 5 | 15 |
| 0.666 | 1.0 | 0 | 0 | any | any (acoustic distance alone) |
| 0.583 | 0.2 | 0.5 | 0.3 | 5 | 15 (current defaults) |

Two-fold cross-validation (3 random splits, best combination chosen on one half,
evaluated on the other): test Spearman 0.61 to 0.74 for the tuned weights against
0.51 to 0.66 for the defaults on the same halves, i.e. a gain of +0.07 to +0.10 on
every split. The winning weights are the same on every half (0.8/0.1/0.1); the acoustic
bounds are not identified (anything from 4-7 / 15-20 gives the same result).

### Interpretation

- The score is a usable but coarse proxy for human judgment: Spearman 0.58 with the
  human total at the utterance level, 0.78 once averaged per speaker. It ranks a
  learner correctly, it does not rank a single sentence reliably (sd of 20+ points at
  a given human level). For reference, the GOP + SVR baseline of the speechocean762 paper
  is around Pearson 0.64 on the total score and later supervised systems (GOPT and
  followers) reach 0.74 to 0.77 (from memory of the literature, to be checked before
  quoting).
- The signal is carried by the acoustic distance (DTW on Wav2Vec2 embeddings vs the TTS
  reference): alone it reaches Spearman 0.67, more than the combined score. The word
  error rate is second (0.55). The phoneme error rate is the weakest component
  (Spearman 0.41, Pearson close to 0), even though it has the largest weight (0.5),
  which is why the default score under-performs its own best component.
- Part of the PER weakness is a bug in `openpronounce/phones.py`, not a modelling
  limit: when espeak merges neighbouring words ("would have to" comes back as one
  group), `_expected_phones_by_word` falls back to per-word phonemization with
  `Separator(phone=" ", word=" ")`, which phonemizer rejects (`ValueError`, identical
  separators). Every word then gets an empty phone list, PER becomes
  `len(heard) / 1` (values up to 49 in the CSV) and no word can be flagged. This hits
  84 of the 500 utterances (17%). On the 416 unaffected utterances the PER reaches
  Spearman 0.53 with the human total, and word-level recall is 0.88 (precision 0.11).
- Word-level flagging is far too eager: one word in three is flagged while the raters
  reject one in twenty, so precision is 0.11 with recall 0.70. This is the phone
  recognizer plus a 0.5 edit-ratio threshold on very short words on child speech; the
  feedback list is not usable as is on this population.
- Recommendation: `SCORE_WEIGHTS = {acoustic: 0.8, phonemes: 0.1, words: 0.1}` with
  `ACOUSTIC_DISTANCE_GOOD/BAD = 5/18` (Spearman 0.674 vs 0.583). The gain is material
  and stable across splits, so it is not an artefact of the 500-utterance sample.
  Two caveats before adopting it: the corpus is a single population (Mandarin L2,
  many children, very short sentences, one TTS voice as reference), and a heavier
  acoustic weight also makes the score more sensitive to voice/recording mismatch
  with the gTTS reference. Re-running the grid search after fixing the phonemizer
  fallback is advisable, since the PER weight would probably not stay at 0.1 with a
  working PER. The mean score also drops from 50 to 40 with the tuned weights and
  bounds 5/15 (51 with 5/18): the bounds shift the level, the weights the ranking.
