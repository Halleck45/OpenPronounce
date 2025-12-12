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
            
            if (isApiCall) {
                // This is an API call - use query parameters to communicate with Streamlit
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
                                
                                // Store request data in sessionStorage for Streamlit to access
                                const requestData = {
                                    requestId: requestId,
                                    url: url,
                                    method: options.method || 'POST',
                                    fileData: base64,
                                    expectedText: expectedText,
                                    text: expectedText
                                };
                                
                                // Use query parameters to trigger Streamlit processing
                                const params = new URLSearchParams(window.location.search);
                                params.set('_api_req', JSON.stringify(requestData));
                                window.location.search = params.toString();
                            };
                            reader.onerror = function() {
                                reject(new Error('Failed to read file'));
                            };
                            reader.readAsDataURL(file);
                        } else {
                            // No file, just text
                            const requestData = {
                                requestId: requestId,
                                url: url,
                                method: options.method || 'POST',
                                text: expectedText
                            };
                            
                            const params = new URLSearchParams(window.location.search);
                            params.set('_api_req', JSON.stringify(requestData));
                            window.location.search = params.toString();
                        }
                    } else {
                        // Simple request
                        const requestData = {
                            requestId: requestId,
                            url: url,
                            method: options.method || 'GET',
                            body: options.body
                        };
                        
                        const params = new URLSearchParams(window.location.search);
                        params.set('_api_req', JSON.stringify(requestData));
                        window.location.search = params.toString();
                    }
                    
                    // Poll for response in sessionStorage (set by Streamlit)
                    const checkResponse = setInterval(() => {
                        const responseKey = `api_response_${requestId}`;
                        const responseData = sessionStorage.getItem(responseKey);
                        
                        if (responseData) {
                            clearInterval(checkResponse);
                            sessionStorage.removeItem(responseKey);
                            pendingRequests.delete(requestId);
                            
                            try {
                                const response = JSON.parse(responseData);
                                
                                if (response.error) {
                                    reject(new Error(response.error));
                                } else {
                                    // Create a fetch-like response
                                    let data = response.data;
                                    
                                    // For TTS, convert base64 to blob
                                    if (url.includes('/tts') && data && data.audio) {
                                        const audioBlob = base64ToBlob(data.audio, 'audio/wav');
                                        const fetchResponse = new Response(audioBlob, {
                                            status: 200,
                                            statusText: 'OK',
                                            headers: { 'Content-Type': 'audio/wav' }
                                        });
                                        resolve(fetchResponse);
                                    } else {
                                        const fetchResponse = new Response(
                                            JSON.stringify(data),
                                            {
                                                status: 200,
                                                statusText: 'OK',
                                                headers: { 'Content-Type': 'application/json' }
                                            }
                                        );
                                        resolve(fetchResponse);
                                    }
                                }
                            } catch (e) {
                                reject(new Error('Failed to parse response: ' + e.message));
                            }
                        }
                    }, 100); // Check every 100ms
                    
                    // Timeout after 60 seconds
                    setTimeout(() => {
                        clearInterval(checkResponse);
                        if (pendingRequests.has(requestId)) {
                            pendingRequests.delete(requestId);
                            reject(new Error('Request timeout'));
                        }
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
if 'api_responses' not in st.session_state:
    st.session_state.api_responses = {}

# Create HTML component that can communicate with Streamlit
html_content = load_html_with_integrated_scripts()

if html_content:
    # Add a script in the Streamlit page to listen for postMessage and handle API requests
    message_handler_script = """
    <script>
    (function() {
        console.log('Streamlit API handler loaded');
        
        // Listen for messages from the HTML component
        window.addEventListener('message', async function(event) {
            // Only process messages from our component
            if (event.data && event.data.type === 'api_request') {
                console.log('API request received:', event.data);
                
                const requestId = event.data.requestId;
                const url = event.data.url;
                let result = null;
                let error = null;
                
                try {
                    // Send request to Streamlit backend via a hidden mechanism
                    // Since we can't directly call Python from JS, we'll use a workaround
                    // Store request in a way Streamlit can access it
                    
                    // Create a custom event that Streamlit can catch
                    const requestEvent = new CustomEvent('streamlit_api_request', {
                        detail: event.data
                    });
                    document.dispatchEvent(requestEvent);
                    
                    // For now, we'll need to use a different approach
                    // Let's use query parameters as a bridge
                    const params = new URLSearchParams(window.location.search);
                    params.set('_api_req', JSON.stringify({
                        requestId: requestId,
                        url: url,
                        fileData: event.data.fileData,
                        expectedText: event.data.expectedText,
                        text: event.data.text
                    }));
                    
                    // This will trigger a page reload, but we need a better solution
                    // For now, let's log the request
                    console.log('Storing API request:', requestId);
                    
                } catch (e) {
                    console.error('Error handling API request:', e);
                    error = e.message;
                }
            }
        });
    })();
    </script>
    """
    
    # Inject the message handler script into the page
    st.markdown(message_handler_script, unsafe_allow_html=True)
    
    # Check for API requests via query parameters (workaround)
    query_params = st.query_params
    
    if '_api_req' in query_params:
        try:
            request_data = json.loads(query_params['_api_req'][0])
            request_id = request_data.get('requestId')
            url = request_data.get('url', '')
            
            st.session_state.api_requests[request_id] = request_data
            
            # Process the request
            result = None
            error = None
            
            try:
                if '/pronunciation' in url:
                    if 'fileData' in request_data and 'expectedText' in request_data:
                        result = handle_pronunciation_api(
                            request_data['fileData'],
                            request_data['expectedText']
                        )
                elif '/speech2text' in url:
                    if 'fileData' in request_data:
                        result = handle_speech2text_api(request_data['fileData'])
                elif '/phonemes' in url:
                    if 'text' in request_data:
                        result = handle_phonemes_api(request_data['text'])
                elif '/tts' in url:
                    if 'text' in request_data:
                        result = handle_tts_api(request_data['text'])
            except Exception as e:
                error = str(e)
                st.error(f"API Error: {error}")
                import traceback
                st.code(traceback.format_exc())
            
            # Store response
            st.session_state.api_responses[request_id] = {
                'data': result,
                'error': error
            }
            
            # Store response in sessionStorage for JavaScript to pick up
            response_data = {
                'data': result,
                'error': error
            }
            response_script = f"""
            <script>
            (function() {{
                // Store response in sessionStorage for the component to read
                const responseKey = 'api_response_{request_id}';
                sessionStorage.setItem(responseKey, JSON.stringify({json.dumps(response_data)}));
                console.log('Response stored:', '{request_id}');
                
                // Remove the query parameter to avoid reprocessing
                const params = new URLSearchParams(window.location.search);
                params.delete('_api_req');
                const newUrl = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
                window.history.replaceState({{}}, '', newUrl);
            }})();
            </script>
            """
            st.markdown(response_script, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing API request: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Create HTML component
    components.html(
        html_content,
        height=1200,
        scrolling=True
    )
    
    # Display debug info in sidebar
    with st.sidebar:
        if st.checkbox("🔍 Show Debug Info"):
            st.markdown("### API Requests")
            if st.session_state.api_requests:
                st.json(st.session_state.api_requests)
            else:
                st.info("No API requests yet")
            
            st.markdown("### API Responses")
            if st.session_state.api_responses:
                st.json(st.session_state.api_responses)
            else:
                st.info("No API responses yet")
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
