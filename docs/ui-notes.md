# Web demo: UI notes

Files: `templates/index.html` (markup, Tailwind via CDN, a few lines of CSS), `static/ui.js` (all page
logic and server calls), `static/audio.js` (`AudioRecorder`, MediaRecorder + silence detection),
`static/viseme.js` (`Viseme`, mouth animation), `static/assets/` (logo, mouth images, notification sound).
No build step, vanilla JS.

## Layout

One column, `max-w-2xl`, warm accent (`accent-500 = #ea580c`) used only for actions.

1. Header: wordmark, GitHub link.
2. Headline + one-line sub.
3. Practice card: sentence textarea, language select (fed by `GET /languages`, remembered in
   `localStorage`), example chips (per language, `EXAMPLES` in `ui.js`), the mic button, "Hear the
   reference" (`POST /tts`), "Upload an audio file" (also drag and drop on the card), then the audio
   player with an "Analyze" button when the audio came from a file (a recording is analyzed as soon as
   it stops).
4. Loading card: progress line, message changes at 0 s / 4 s / 12 s.
5. Error card: one sentence, "Try again".
6. Result card: score ring + number (count-up), verdict ("2 of 4 words need work"), word chips
   (green ok, red flagged; native tooltip with confidence and phones), a detail panel for the selected
   word (expected phones with wrong ones underlined in red using `phones[].confidence >= 0.5`,
   heard phones, "The speech recognizer understood ..." when `actual_word` is set, mouth animation +
   SpeechSynthesis on "Hear it"), "We heard" (transcription) and "Your sounds" (heard phones, low
   confidence in grey), then a `<details>` with the phoneme trace and the pitch / energy charts,
   then "Try again" / "Replay my recording".
7. Footer: GitHub, MIT, `pip install openpronounce`, link to the article.

## States and events

`AudioRecorder` emits `record:start`, `record:stop`, `record:silence`, `record:ready` (`detail.blob`)
and `record:error` (`detail.reason` in `insecure | denied | notfound | unknown`). `ui.js` maps the
reason to a short inline message under the mic. Empty sentence: inline message under the examples,
textarea focused. Server errors (5xx, network) show the error card; a 422 shows the server detail.
`setView('idle' | 'loading' | 'error' | 'result')` is the only place that shows/hides the three cards.

The first flagged word is selected automatically after an analysis so the detail panel is never empty.

## Test locally

```bash
HF_HOME=$HOME/.cache/huggingface PYTHONPATH=$PWD .venv/bin/uvicorn server:app --port 8811
# then open http://localhost:8811 (localhost counts as a secure context, so the mic works)
PYTHONPATH=$PWD .venv/bin/python -m pytest -q tests/test_server.py
node --check static/ui.js static/audio.js static/viseme.js
```

Without a mic: "Upload an audio file" with `assets/example.mp3` and the sentence "Hello, how are you?"
(two words flagged, score around 56), or `assets/developer.wav` with "hello I am a developer" (all
clear, score around 97).

## Recording the README GIF (1280x800)

1. Open `http://localhost:8811/` (models already warm: run one analysis before recording).
2. Click the example chip "Hello, how are you?" (the textarea already holds it, the click shows the
   affordance).
3. Click the mic, say the sentence, wait for the two seconds of silence (auto stop) or click again.
4. Loading card appears ("Listening to your recording", then "Comparing your sounds...").
5. Result: ring animates, chips appear, "Hello" is selected. Click "how", then "Hear it and watch the
   mouth" so the mouth animates.
6. Optionally open "Details" to show the charts, then click "Try again".
