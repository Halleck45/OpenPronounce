
## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## As API

Mount the server:

```
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### Speech2text

```bash
curl -X POST 127.0.0.1:8000/speech2text -F 'file=@assets/harvard.wav'
```

### Pronunciation

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/pronunciation' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@assets/developer.wav' \
  -F 'expected_text=hello I am a developer'
```

## As CLI

```bash
python main.py assets/developer.wav "hello, I am a developer"
```

# References

[Visemes](https://learn.microsoft.com/fr-fr/azure/ai-services/speech-service/how-to-speech-synthesis-viseme?tabs=visemeid&pivots=programming-language-csharp), [Visemes SSML](https://learn.microsoft.com/fr-fr/azure/ai-services/speech-service/speech-ssml-phonetic-sets)