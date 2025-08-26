import sys
import audio
import speech

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Utilisation : python main.py <file.wav> <text>")
        sys.exit(1)

    audio_path = sys.argv[1]
    expected_text = sys.argv[2]

    sound = audio.load(audio_path)
    json = speech.compare_audio_with_text(sound, expected_text)
    print(json)