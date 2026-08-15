# Changelog

## 0.3.0 (unreleased)

- Added: multi-language support (fr, es, de, it, pt, nl), `lang` parameter everywhere (`--lang` on the CLI, `lang` form field on the API), `GET /languages`. English stays the default and behaves as before.
- Removed: the Streamlit app (`streamlit_app.py`), whose bridge between the embedded UI and Streamlit never worked; use the FastAPI app or the Docker image.
- Fixed: in the transcription-based fallback, extra words heard inside a word ("hell no" for "hello") are now shown in `actual_word` (#6).
- Added: CUDA support (`OPENPRONOUNCE_DEVICE`, automatic when available) and `Dockerfile.gpu`.
- Fixed: sentences that espeak merges into one token ("would have to") lost all expected phones, so no word could be flagged and the phone error rate exploded (17 % of speechocean762 utterances).
- Added: `benchmarks/speechocean762.py`, benchmark of the score against human ratings.
- Changed: word-level mispronunciation detection uses the CTC posteriors of the phone recognizer. `errors[]` entries gain `confidence` (0-1) and `phones[]` (`expected`, `heard`, `confidence` per phone), `differences.heard_phones_confidence` is added; `transcribe_phones(..., return_confidence=True)` and `recognize_phones()` expose the confidences. Close substitutions and word-final slips count less, and an expected phone the recognizer found plausible is not an error. On speechocean762 the precision of the flagged words goes from 0.12 to 0.20 (F1 0.21 to 0.31, one word in five flagged instead of two in five). Also fixed a few normalization gaps (Mandarin tone numbers, aspirated stops, `ɔɹ`/`oɹ` and `ɜ`/`ɚ` merges).
- Added: offline TTS backends (Piper, Kokoro), `OPENPRONOUNCE_TTS` env var, `OPENPRONOUNCE_TTS_VOICE` / `voice=` to pick the voice. `pip install openpronounce[tts-piper]` or `[tts-kokoro]`.

## 0.2.1 (2026-08-15)

- Install from PyPI in the docs and the notebook.
- Release workflow (`Release` action, `vX.Y.Z` input) that bumps, tags, publishes to PyPI and creates the GitHub Release.
- Fix test discovery on CI (`pythonpath`).

## 0.2.0 (2026-08-15)

### Breaking

- The code now lives in the `openpronounce` package: `from openpronounce import audio, speech`
  (was `import audio`, `import speech`). Installable with `pip install .` and shipped with an
  `openpronounce` console script.
- The score is now computed from length-independent measures (mean per-frame DTW distance,
  phoneme error rate, word error rate). Scores are not comparable with 0.1.x: previously a
  perfect reading of a long sentence could score 30/100 because raw DTW distances grow with
  the audio length.
- `compute_pronunciation_score(acoustic_distance, phoneme_error_rate, word_error_rate)` takes
  the normalized measures.
- `audio.webp2wav` is renamed `audio.webm2wav` (old name kept as an alias) and writes `<name>.16k.wav`.

### Fixed

- Upper-case reference text was phonemized letter by letter by espeak (`IT` -> /aɪtiː/), so
  every word of an upper-case sentence was flagged as mispronounced.
- The reference audio was written to a fixed `reference.mp3` in the working directory:
  concurrent web requests overwrote each other's file. References are now cached per
  sentence under `$OPENPRONOUNCE_CACHE_DIR` (default: system temp dir).
- Uploading a `.wav` to the API deleted the converted file before it was read.
- The test suite referenced functions that did not exist and CI had been red since December 2025.

### Added

- Phone-level assessment: `wav2vec2-lv-60-espeak-cv-ft` recognizes the phones actually
  said; `differences.errors` now shows, per word, the expected phones and the phones heard
  (`openpronounce/phones.py`, `transcribe_phones`, `compare_phones`). Set
  `OPENPRONOUNCE_PHONEME_MODEL=off` to fall back to the transcription-based detection.

### Changed

- Models are loaded lazily on first use, and the CTC checkpoint's encoder is reused for
  embeddings: half the memory and load time, and `import openpronounce` is instantaneous.
- `torchaudio` (whose `load`/`save` now require torchcodec) and unused dependencies
  (`coqui-tts`, `dtw-python`, `pydub`, `spacy` download step) are dropped.
- New fields: `acoustic_distance`, `differences.phoneme_error_rate`, `differences.word_error_rate`,
  `errors[].actual_word` is always present.
- Dockerfile, `/health` endpoint, Swagger metadata, CI matrix (3.10 / 3.12).
- Demo notebook rewritten to install from GitHub and run end to end on Colab.
