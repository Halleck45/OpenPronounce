---
title: OpenPronounce
emoji: 🎤
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Phoneme-level English pronunciation feedback, open source
---

# OpenPronounce

Record a sentence, get a score, the mispronounced words with the phones you actually said (IPA), and your pitch and energy curves.

Open source (MIT), runs on CPU: [github.com/Halleck45/OpenPronounce](https://github.com/Halleck45/OpenPronounce), `pip install openpronounce`.

This Space runs the FastAPI app from the repository. The first analysis after a cold start takes a while (two Wav2Vec2 models are loaded); the next ones take about 15 s on the free CPU.
