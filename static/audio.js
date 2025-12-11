class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioContext = null;
        this.analyser = null;
        this.silenceTimer = null;
        this.silenceThreshold = 50;
        this.silenceDuration = 2000;
        this.started = false;
    }

    async start() {
        if (this.started) {
            this.stop();
            return;
        }

        try {
            // 🔴 Demande l'accès au micro seulement ici, et non plus au chargement de la page
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new AudioContext();
            const source = this.audioContext.createMediaStreamSource(this.stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);
            this.mediaRecorder = new MediaRecorder(this.stream);

            this.mediaRecorder.ondataavailable = event => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => this.processAudio();

            this.started = true;
            this.audioChunks = [];

            this.mediaRecorder.start();
            requestAnimationFrame(() => this.checkSilence());

            document.dispatchEvent(new Event('record:start'));

            setTimeout(() => this.stop(), 60000);
        } catch (err) {
            console.error("Erreur d'accès au micro: ", err);
        }
    }

    stop() {

        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        this.stream = null;

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }

        if (!this.started) {
            return;
        }

        this.started = false;

        document.dispatchEvent(new Event('record:stop'));
    }

    isRecording() {
        return this.started;
    }

    stopDueToSilence() {
        this.stop();
        document.dispatchEvent(new Event('record:silence'));
    }

    checkSilence() {
        let dataArray = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(dataArray);
        let average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;

        if (average < this.silenceThreshold) {
            if (!this.silenceTimer) {
                this.silenceTimer = setTimeout(() => this.stopDueToSilence(), this.silenceDuration);
            }
        } else {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }

        if (this.mediaRecorder.state === 'recording') {
            requestAnimationFrame(() => this.checkSilence());
        }
    }

    processAudio() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

        // Show audio player with recorded audio
        const audioPlayer = document.getElementById('audio-player');
        const audioPlayerContainer = document.getElementById('audio-player-container');
        const url = URL.createObjectURL(audioBlob);
        audioPlayer.src = url;
        audioPlayerContainer.classList.remove('hidden');

        this.sendAudioToAPI(audioBlob);
    }

    sendAudioToAPI(audioBlob) {
        //this.sendSpeechToAPI(audioBlob);
        this.sendPronunciationToAPI(audioBlob);
    }

    sendSpeechToAPI(audioBlob) {
        const formData = new FormData();
        formData.append("file", audioBlob, "audio.webm");

        fetch("/speech2text", { method: "POST", body: formData })
            .then(response => response.json())
            .then(data => {
                document.querySelector('[data-role="transcription"]').innerText = data.transcript;
            })
            .catch(error => console.error("Erreur d'envoi:", error));
    }

    sendPronunciationToAPI(audioBlob) {
        const formData = new FormData();
        formData.append("file", audioBlob, "audio.webm");
        formData.append("expected_text", document.getElementById('expected-text').value);

        fetch("/pronunciation", { method: "POST", body: formData })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                console.log(JSON.stringify(data));
                document.dispatchEvent(new CustomEvent('analyze:done', {
                    detail: data
                }));
            })
            .catch(error => {
                console.error("Erreur d'envoi:", error);
                document.dispatchEvent(new Event('analyze:error'));
            });
    }

    sendGrammarApi(text) {
        const formData = new FormData();
        formData.append("text", text);

        fetch("/grammar", { method: "POST", body: formData })
            .then(response => response.json())
            .then(data => {
                document.querySelector('[data-role="grammar"]').innerText = JSON.stringify(data);
                this.displayResults();
            })
            .catch(error => console.error("Erreur d'envoi:", error));
    }
}
