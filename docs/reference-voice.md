# Reference voice

The acoustic term needs a native reading of the expected sentence: it is synthesized once per sentence, encoded with Wav2Vec2 like the learner's recording, and cached under `$OPENPRONOUNCE_CACHE_DIR` (system temp dir by default). Three synthesizers are available, chosen with the `OPENPRONOUNCE_TTS` environment variable (or `audio.text2speech(..., backend=...)`):

| `OPENPRONOUNCE_TTS` | Engine | Install | Offline | Download |
|---|---|---|---|---|
| `gtts` (default) | Google Translate TTS | included | no: network on the first analysis of each sentence, cached afterwards | none |
| `piper` | [Piper](https://github.com/OHF-Voice/piper1-gpl) (VITS, ONNX, CPU) | `pip install openpronounce[tts-piper]` | yes, after the first voice download | ~60 MB per medium voice, from `rhasspy/piper-voices` |
| `kokoro` | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (PyTorch) | `pip install openpronounce[tts-kokoro]` | yes, after the first model download | ~330 MB model + a few MB per voice (+ the spaCy `en_core_web_sm` model, fetched on first use for English) |

`OPENPRONOUNCE_TTS_VOICE` (or `voice=`) selects the voice: a Piper voice id such as `en_US-lessac-medium` (default) or `en_GB-cori-medium`, a Kokoro voice such as `af_heart` (default) or `bf_emma`, or, for gTTS, the Google domain that sets the accent (`com`, `co.uk`, `com.au`). Piper and Kokoro ship a default voice for the languages they cover (`openpronounce.tts.PIPER_DEFAULT_VOICES`, `openpronounce.tts.KOKORO_LANGUAGES`); models and voices land in the Hugging Face cache (`$HF_HOME`), so `HF_HUB_OFFLINE=1` works once they are there. The reference cache is keyed by backend and voice, so switching engines does not serve stale references.

For self-hosting we recommend Piper: no network at all, small, fast on CPU, and no PyTorch model to load next to Wav2Vec2. Kokoro sounds more natural but costs ~330 MB and a few seconds of warm-up. On the bundled samples the acoustic distance stays on the gTTS scale with Kokoro (6.2 / 11.6 / 10.3 for `developer.wav`, `developer1.wav`, `harvard.wav` versus 6.3 / 11.4 / 10.1 with gTTS) and shifts up by 1 to 2 with Piper on good readings (8.2 / 11.9 / 11.0), which lowers the acoustic term slightly (a 30 % weight in the score) until it is recalibrated for that engine.
