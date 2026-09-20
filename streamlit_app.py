"""Streamlit front-end for the hosted OpenPronounce demo (Hugging Face Space)."""

import os
import tempfile

import streamlit as st

from openpronounce import compare_audio_with_text, load_audio
from openpronounce.languages import DEFAULT_LANGUAGE, LANGUAGES

st.set_page_config(
    page_title="OpenPronounce",
    page_icon="🎤",
    layout="wide",
)

st.title("OpenPronounce")
st.markdown(
    "Upload or record a sentence and get a pronunciation score, the words you "
    "mispronounced (with the phones you actually said), and prosody curves."
)

expected = st.text_input(
    "Expected sentence",
    value="Hello, how are you?",
)
language_options = sorted(LANGUAGES)
lang = st.selectbox(
    "Language",
    options=language_options,
    index=language_options.index(DEFAULT_LANGUAGE) if DEFAULT_LANGUAGE in language_options else 0,
)
uploaded = st.file_uploader(
    "Recording",
    type=["wav", "mp3", "flac", "ogg", "webm", "m4a"],
)

if uploaded is None or not expected.strip():
    st.info("Enter the expected sentence and upload a recording to analyze it.")
elif st.button("Analyze"):
    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(uploaded.read())
        path = handle.name

    try:
        with st.spinner("Loading the model(s) and analyzing the recording..."):
            sound = load_audio(path)
            result = compare_audio_with_text(sound, expected.strip(), lang=lang)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")
    else:
        st.subheader(f"Score: {result['score']:.0f}/100")
        st.write(f"Transcription: {result.get('transcribe') or ''}")

        errors = result.get("differences", {}).get("errors", [])
        if errors:
            st.markdown("**Mispronounced words**")
            for error in errors:
                actual = error.get("actual") or "(missing)"
                confidence = f" (confidence {error['confidence']:.0%})" if "confidence" in error else ""
                st.write(
                    f"- {error['word']}: expected /{error['expected']}/, "
                    f"heard /{actual}/{confidence}"
                )
        else:
            st.success("No mispronounced word detected.")
    finally:
        if os.path.exists(path):
            os.unlink(path)
