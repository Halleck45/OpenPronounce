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

## Word-level detection

`benchmarks/word_detection.py` calibrates the rule that flags a word as mispronounced,
on the same 500-utterance sample. It runs the phone recognizer once (`--extract`, the
frame posteriors are cached in `~/.cache/openpronounce/speechocean762/logits`), then
tunes and reports without the model (`--tune`, `--report`), and checks the bundled
samples (`--assets`). The 500 utterances are split by parity of their index: **250 for
tuning (even), 250 held out (odd)**; the constants below were chosen on the tuning half only.

### Rule (0.3.0)

Every wrong phone of a word gets an error confidence in [0, 1]:

- substitution: 1, or `NEAR_PHONE_COST = 0.5` when the two phones are close (voicing
  pairs, tense/lax vowels, ð/z/d, θ/s/t, n/ŋ, h/x...);
- deletion: 1, or `FINAL_DELETION_COST = 0.5` for the last phone of the word;
- extra phone next to a correct one: 1, or `FINAL_EXTRA_COST = 0.25` after the last
  phone of the word (epenthetic vowel, onset of the next word caught by the alignment);
- the value is then scaled by `1 - min(1, p / PHONE_PLAUSIBLE_POSTERIOR)` where `p` is
  the highest posterior of the expected phone in the frames it aligned to (its
  neighbours' frames for a deletion) and `PHONE_PLAUSIBLE_POSTERIOR = 0.05`: an
  expected phone the recognizer found plausible is not a confident error.

A word is flagged when the confidences add up to `PHONE_ERROR_THRESHOLD = 0.4` of its
phones or to `PHONE_ERROR_MIN_EDITS = 2` phones. Its `confidence` is
`min(1, max(sum / phones, sum / 2))`. The 0.2.1 rule was: plain edit distance of at
least 50 % of the phones or 3.

The F1 surface on the tuning half is flat (0.305 to 0.321 for thresholds between 0.4
and 0.7 and 2 to 3 minimum edits): 0.4 / 2 keeps recall (0.73 against 0.48 for the
F1-optimal 0.65 / 2.5) and keeps the bundled examples flagged. The recognizer's own
confidence in the heard phone (peak posterior) is not discriminative on this corpus
(same distribution for confirmed and refuted alarms) and does not enter the rule; the
posterior of the *expected* phone and the phone-pair costs carry the gain.

### Results (2026-08-15, N = 500, 3,146 words)

"bad" = human word accuracy < 5 (168 words, 5.3 %); "lenient" = accuracy < 7 (265
words, 8.4 %). P@R.5 / P@R.7 = best precision reachable at recall >= 0.5 / 0.7 by
moving the two thresholds of that system.

| system | split | P | R | F1 | flagged | P@R.5 | P@R.7 |
|---|---|--:|--:|--:|--:|--:|--:|
| 0.2.1 (edit distance, exact old code) | tune | 0.119 | 0.939 | 0.211 | 41.9 % | | |
| 0.2.1 (edit distance, exact old code) | held-out | 0.119 | 0.872 | 0.210 | 39.2 % | | |
| 0.2.1 rule, new normalization | tune | 0.127 | 0.939 | 0.223 | 39.4 % | 0.161 | 0.161 |
| 0.2.1 rule, new normalization | held-out | 0.126 | 0.860 | 0.221 | 36.5 % | 0.171 | 0.171 |
| confidence rule, no posteriors | tune | 0.152 | 0.902 | 0.261 | 31.5 % | 0.206 | 0.173 |
| confidence rule, no posteriors | held-out | 0.148 | 0.837 | 0.251 | 30.4 % | 0.198 | 0.168 |
| **confidence rule + posteriors (default)** | tune | 0.194 | 0.732 | 0.307 | 20.0 % | 0.227 | 0.198 |
| **confidence rule + posteriors (default)** | held-out | 0.201 | 0.721 | 0.314 | 19.3 % | 0.241 | 0.201 |

Same, lenient label (accuracy < 7):

| system | split | P | R | F1 | flagged | P@R.5 | P@R.7 |
|---|---|--:|--:|--:|--:|--:|--:|
| 0.2.1 (edit distance, exact old code) | tune | 0.179 | 0.866 | 0.297 | 41.9 % | | |
| 0.2.1 (edit distance, exact old code) | held-out | 0.170 | 0.817 | 0.282 | 39.2 % | | |
| 0.2.1 rule, new normalization | tune | 0.188 | 0.851 | 0.307 | 39.4 % | 0.232 | 0.232 |
| 0.2.1 rule, new normalization | held-out | 0.181 | 0.809 | 0.296 | 36.5 % | 0.241 | 0.231 |
| confidence rule, no posteriors | tune | 0.224 | 0.813 | 0.352 | 31.5 % | 0.285 | 0.262 |
| confidence rule, no posteriors | held-out | 0.207 | 0.771 | 0.327 | 30.4 % | 0.301 | 0.262 |
| **confidence rule + posteriors (default)** | tune | 0.269 | 0.619 | 0.375 | 20.0 % | 0.303 | 0.261 |
| **confidence rule + posteriors (default)** | held-out | 0.269 | 0.634 | 0.377 | 19.3 % | 0.328 | 0.262 |

"New normalization" = 0.2.1 thresholds on the plain edit distance, with the 0.3.0
phone normalization (Mandarin tone numbers and aspiration marks dropped, `ɔɹ`/`oɹ`,
`ɜ`/`ɚ` merged, `ai`/`ei`/`au`/`ou` spelled as diphthongs). "No posteriors" = the
0.3.0 costs without the plausibility scaling (what a plain list of phones gives).

Bundled samples (`--assets`), before -> after: `developer.wav` none -> none;
`example.mp3` hello, how -> hello (0.89), how (0.50); `harvard.wav` beer, al -> none;
`developer1.wav` i, developer -> developer (1.00).

### Interpretation

- Half of the false alarms went away (39 % of the words flagged -> 19 %) for a quarter
  of the hits (recall 0.87 -> 0.72 on the held-out half); precision 0.12 -> 0.20 (0.17
  -> 0.27 with the lenient label). Tuning and held-out halves agree within 0.01.
- Precision stays low in absolute terms because the raters are lenient: 90 % of the
  words are rated 10/10, accent traits such as ð -> z, θ -> s, ɪ -> i or a dropped
  final consonant are almost never rated below 5, and 2-phone function words ("the",
  "to", "it", "you") make up most of the remaining alarms. Among words with a given edit
  count, no more than 25 % are rated bad, so an edit-count rule cannot go much further.
- The recognizer's confidence in what it heard does not separate true from false
  alarms; the posterior of the expected phone (GOP-like) and the phone-pair costs do.
  Next steps: a phonetically weighted alignment (the costs inside the DP, not only after
  it), and a small classifier on the per-word features against the human labels.
