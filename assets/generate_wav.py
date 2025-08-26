from TTS.api import TTS

# Charger un modèle pré-entraîné (anglais)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to("cpu")

# Générer l'audio
tts.tts_to_file(text="Hello, how are you?", file_path="reference.wav")