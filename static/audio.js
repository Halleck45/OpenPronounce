/**
 * Microphone recorder built on MediaRecorder.
 *
 * Emits DOM events on `document`:
 *   record:start          recording began
 *   record:stop           recording ended (user, timeout or silence)
 *   record:silence        recording ended because of silence
 *   record:ready          detail: { blob }  the webm blob is available
 *   record:error          detail: { reason } one of "insecure", "denied", "notfound", "unknown"
 */
class AudioRecorder {
    constructor({ silenceThreshold = 50, silenceDuration = 2000, maxDuration = 60000 } = {}) {
        this.mediaRecorder = null;
        this.stream = null;
        this.audioContext = null;
        this.analyser = null;
        this.audioChunks = [];
        this.silenceTimer = null;
        this.maxTimer = null;
        this.silenceThreshold = silenceThreshold;
        this.silenceDuration = silenceDuration;
        this.maxDuration = maxDuration;
        this.started = false;
    }

    static isSupported() {
        return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
    }

    async start() {
        if (this.started) {
            this.stop();
            return;
        }

        if (!AudioRecorder.isSupported()) {
            this.emit('record:error', { reason: window.isSecureContext ? 'unknown' : 'insecure' });
            return;
        }

        try {
            // Ask for the microphone only now, never on page load
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            const name = err && err.name;
            const reason = (name === 'NotAllowedError' || name === 'SecurityError') ? 'denied'
                : (name === 'NotFoundError' || name === 'OverconstrainedError') ? 'notfound'
                    : 'unknown';
            this.emit('record:error', { reason });
            return;
        }

        this.audioContext = new AudioContext();
        const source = this.audioContext.createMediaStreamSource(this.stream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        source.connect(this.analyser);

        this.audioChunks = [];
        this.mediaRecorder = new MediaRecorder(this.stream);
        this.mediaRecorder.ondataavailable = event => this.audioChunks.push(event.data);
        this.mediaRecorder.onstop = () => {
            const blob = new Blob(this.audioChunks, { type: 'audio/webm' });
            this.emit('record:ready', { blob });
        };

        this.started = true;
        this.mediaRecorder.start();
        requestAnimationFrame(() => this.checkSilence());
        this.maxTimer = setTimeout(() => this.stop(), this.maxDuration);

        this.emit('record:start');
    }

    stop() {
        clearTimeout(this.maxTimer);
        clearTimeout(this.silenceTimer);
        this.maxTimer = null;
        this.silenceTimer = null;

        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }

        if (!this.started) {
            return;
        }
        this.started = false;
        this.emit('record:stop');
    }

    isRecording() {
        return this.started;
    }

    stopDueToSilence() {
        this.stop();
        this.emit('record:silence');
    }

    checkSilence() {
        if (!this.started || !this.analyser) {
            return;
        }
        const data = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(data);
        const average = data.reduce((sum, value) => sum + value, 0) / data.length;

        if (average < this.silenceThreshold) {
            if (!this.silenceTimer) {
                this.silenceTimer = setTimeout(() => this.stopDueToSilence(), this.silenceDuration);
            }
        } else {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }

        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            requestAnimationFrame(() => this.checkSilence());
        }
    }

    emit(name, detail) {
        document.dispatchEvent(new CustomEvent(name, { detail }));
    }
}
