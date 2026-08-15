import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import server


class TestServer(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(server.app)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_home_serves_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<html", response.text)

    def test_phonemes(self):
        response = self.client.post("/phonemes", data={"text": "hello world"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreater(len(body["phonemes"]), 0)
        self.assertEqual(len(body["phonemes"]), len(body["words"]))

    @patch("server.speech.transcribe", return_value="HELLO")
    def test_speech2text(self, _):
        import io
        import numpy as np
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
        buf.seek(0)
        response = self.client.post("/speech2text", files={"file": ("rec.wav", buf, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"transcript": "HELLO"})

    def test_ui_assets_and_languages(self):
        for path in ("/static/ui.js", "/static/audio.js", "/static/viseme.js", "/static/assets/logo.svg"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        home = self.client.get("/").text
        for element in ("record-btn", "language-select", "expected-text", "word-chips", "score-ring"):
            self.assertIn(f'id="{element}"', home)
        languages = self.client.get("/languages").json()
        self.assertEqual(languages["default"], "en")
        self.assertIn({"code": "en", "name": "English"}, languages["languages"])
