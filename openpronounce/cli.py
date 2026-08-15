"""Command-line entry point: ``openpronounce <audio> "<expected text>"``."""

import argparse
import json
import sys

from .languages import DEFAULT_LANGUAGE, LANGUAGES


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="openpronounce",
        description="Score the pronunciation of a recording against the sentence it should contain.",
    )
    parser.add_argument("audio", help="path to a wav/mp3/flac/ogg/webm recording")
    parser.add_argument("text", help="the sentence the speaker was supposed to say")
    parser.add_argument("--lang", default=DEFAULT_LANGUAGE, choices=sorted(LANGUAGES),
                        help="language of the sentence (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="print the full JSON result instead of a summary")
    parser.add_argument("--no-prosody", action="store_true", help="omit prosody contours from the JSON output")
    args = parser.parse_args(argv)

    from . import audio, speech

    sound = audio.load(args.audio)
    result = speech.compare_audio_with_text(sound, args.text, lang=args.lang)

    if args.no_prosody:
        result.pop("prosody", None)

    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"Score        : {result['score']}/100")
    print(f"Transcription: {result['transcribe']}")
    if "heard_phones" in result["differences"]:
        print(f"Heard phones : /{' '.join(result['differences']['heard_phones'])}/")
    errors = result["differences"]["errors"]
    if errors:
        print("Mispronounced:")
        for err in errors:
            actual = f"/{err['actual']}/" if err["actual"] else "(missing)"
            print(f"  - {err['word']}: expected /{err['expected']}/, heard {actual}")
    else:
        print("No mispronounced word detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
