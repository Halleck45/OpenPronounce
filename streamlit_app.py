import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import base64
import json
import speech
import audio
import random
import string

# Configuration de la page Streamlit
st.set_page_config(
    page_title="OpenPronounce - Pronunciation Analysis",
    page_icon="🎤",
    layout="wide"
)

# Chemins des fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

def load_html_with_integrated_scripts():
    """Load HTML and integrate JS scripts directly"""
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(html_path):
        return None
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Integrate JS files directly into HTML
    for js_file in ['audio.js', 'viseme.js', 'ui.js']:
        js_path = os.path.join(STATIC_DIR, js_file)
        if os.path.exists(js_path):
            with open(js_path, 'r', encoding='utf-8') as f:
                js_content = f.read()
            # Replace external scripts with inline content
            html_content = html_content.replace(
                f'<script src="/static/{js_file}?v=10"></script>',
                f'<script>{js_content}</script>'
            )
            html_content = html_content.replace(
                f'<script src="/static/{js_file}?v=11"></script>',
                f'<script>{js_content}</script>'
            )
    
    # Add a wrapper that intercepts fetch() calls and redirects them to Streamlit
    api_wrapper = """
    <script>
    // Wrapper to intercept API calls and use window.parent.postMessage
    (function() {
        const originalFetch = window.fetch;
        const pendingRequests = new Map();
        let requestIdCounter = 0;
        
        window.fetch = function(url, options = {}) {
            // Check if it's an API endpoint call
            const apiEndpoints = ['/pronunciation', '/speech2text', '/phonemes', '/tts'];
            const isApiCall = apiEndpoints.some(endpoint => url.includes(endpoint));
            
            if (isApiCall && window.parent && window.parent !== window) {
                // This is an API call from the HTML component
                return new Promise((resolve, reject) => {
                    const requestId = `req_${requestIdCounter++}_${Date.now()}`;
                    
                    // Store the promise
                    pendingRequests.set(requestId, { resolve, reject });
                    
                    // If it's a FormData, convert it to base64
                    if (options.body instanceof FormData) {
                        const formData = options.body;
                        const file = formData.get('file');
                        const expectedText = formData.get('expected_text') || formData.get('text');
                        
                        if (file instanceof File || file instanceof Blob) {
                            // Convert file to base64
                            const reader = new FileReader();
                            reader.onload = function() {
                                const base64 = reader.result.split(',')[1];
                                window.parent.postMessage({
                                    type: 'api_request',
                                    requestId: requestId,
                                    url: url,
                                    method: options.method || 'POST',
                                    fileData: base64,
                                    expectedText: expectedText,
                                    text: expectedText
                                }, '*');
                            };
                            reader.readAsDataURL(file);
                        } else {
                            // No file, just text
                            window.parent.postMessage({
                                type: 'api_request',
                                requestId: requestId,
                                url: url,
                                method: options.method || 'POST',
                                text: expectedText
                            }, '*');
                        }
                    } else {
                        // Simple request
                        window.parent.postMessage({
                            type: 'api_request',
                            requestId: requestId,
                            url: url,
                            method: options.method || 'GET',
                            body: options.body
                        }, '*');
                    }
                    
                    // Listen for response
                    const messageHandler = (event) => {
                        if (event.data && event.data.type === 'api_response' && event.data.requestId === requestId) {
                            window.removeEventListener('message', messageHandler);
                            pendingRequests.delete(requestId);
                            
                            if (event.data.error) {
                                reject(new Error(event.data.error));
                            } else {
                                // Create a fetch-like response
                                let responseData = event.data.data;
                                
                                // For TTS, convert base64 to blob
                                if (url.includes('/tts') && responseData.audio) {
                                    const audioBlob = base64ToBlob(responseData.audio, 'audio/wav');
                                    const response = new Response(audioBlob, {
                                        status: 200,
                                        statusText: 'OK',
                                        headers: { 'Content-Type': 'audio/wav' }
                                    });
                                    resolve(response);
                                } else {
                                    const response = new Response(
                                        JSON.stringify(responseData),
                                        {
                                            status: 200,
                                            statusText: 'OK',
                                            headers: { 'Content-Type': 'application/json' }
                                        }
                                    );
                                    resolve(response);
                                }
                            }
                        }
                    };
                    
                    window.addEventListener('message', messageHandler);
                    
                    // Timeout après 60 secondes
                    setTimeout(() => {
                        window.removeEventListener('message', messageHandler);
                        pendingRequests.delete(requestId);
                        reject(new Error('Request timeout'));
                    }, 60000);
                });
            }
            
            // For other requests, use normal fetch
            return originalFetch(url, options);
        };
        
        function base64ToBlob(base64, mimeType) {
            const byteCharacters = atob(base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], { type: mimeType });
        }
        
        // Listen for response messages from Streamlit
        window.addEventListener('message', (event) => {
            if (event.data && event.data.type === 'api_response') {
                const { requestId } = event.data;
                const pending = pendingRequests.get(requestId);
                if (pending) {
                    if (event.data.error) {
                        pending.reject(new Error(event.data.error));
                    } else {
                        pending.resolve(new Response(
                            JSON.stringify(event.data.data),
                            { status: 200, headers: { 'Content-Type': 'application/json' } }
                        ));
                    }
                    pendingRequests.delete(requestId);
                }
            }
        });
    })();
    </script>
    """
    
    # Insert wrapper before closing </body>
    html_content = html_content.replace('</body>', api_wrapper + '</body>')
    
    return html_content

# Functions to handle API calls
def handle_pronunciation_api(file_data_base64, expected_text):
    """Handle /pronunciation endpoint"""
    try:
        file_bytes = base64.b64decode(file_data_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        try:
            wav_file = audio.webp2wav(tmp_path)
            sound = audio.load(wav_file)
            result = speech.compare_audio_with_text(sound, expected_text)
            return result
        finally:
            for f in [tmp_path, wav_file]:
                if os.path.exists(f):
                    os.unlink(f)
    except Exception as e:
        return {"error": str(e)}

def handle_speech2text_api(file_data_base64):
    """Handle /speech2text endpoint"""
    try:
        file_bytes = base64.b64decode(file_data_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        try:
            wav_file = audio.webp2wav(tmp_path)
            sound = audio.load(wav_file)
            transcript = speech.transcribe(sound)
            return {"transcript": transcript}
        finally:
            for f in [tmp_path, wav_file]:
                if os.path.exists(f):
                    os.unlink(f)
    except Exception as e:
        return {"error": str(e)}

def handle_phonemes_api(text):
    """Handle /phonemes endpoint"""
    try:
        phonemes, words = speech.get_phonemes_with_word_mapping(text)
        return {
            "phonemes": phonemes,
            "words": list(words.values())
        }
    except Exception as e:
        return {"error": str(e)}

def handle_tts_api(text):
    """Handle /tts endpoint"""
    try:
        filename = audio.text2speech(text)
        with open(filename, 'rb') as f:
            audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        if os.path.exists(filename) and filename.startswith('/tmp'):
            os.unlink(filename)
        
        return {"audio": audio_base64, "format": "wav"}
    except Exception as e:
        return {"error": str(e)}

# Handle API requests via messages from HTML component
if 'api_requests' not in st.session_state:
    st.session_state.api_requests = {}

# Create HTML component that can communicate with Streamlit
html_content = load_html_with_integrated_scripts()

if html_content:
    # Create HTML component
    components.html(
        html_content,
        height=1200,
        scrolling=True
    )
    
    # Process pending API requests
    # Note: Streamlit doesn't directly support bidirectional callbacks
    # We'll use an approach with query parameters or session_state
    
    # For now, just display the component
    # API calls will be handled via a polling or refresh system
else:
    st.error("Unable to load HTML template")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    st.success("✅ Application active")
    
    st.markdown("---")
    st.markdown("### 📚 Links")
    st.markdown("[📖 Blog article](https://blog.lepine.pro/en/ai-wav2vec-pronunciation-vectorization/)")
    st.markdown("[🐙 GitHub](https://github.com/Halleck45/OpenPronounce)")
    
    st.markdown("---")
    st.markdown("### 🔧 Troubleshooting")
    st.markdown("""
    If the application doesn't load:
    1. Wait a few seconds
    2. Refresh the page
    3. Check the browser console for errors
    
    **For the microphone:**
    - Click on the lock icon in the address bar
    - Allow microphone access
    - If it doesn't work, use audio file upload
    """)
