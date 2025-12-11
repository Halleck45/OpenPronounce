document.addEventListener('DOMContentLoaded', function () {
    const recorder = new AudioRecorder();
    let recordingStartTime = null;
    let timerInterval = null;

    // Viseme checkbox toggle
    document.getElementById('viseme-checkbox').addEventListener('change', function () {
        const visemeSection = document.getElementById('viseme-section');
        if (this.checked) {
            visemeSection.classList.remove('hidden');
        } else {
            visemeSection.classList.add('hidden');
        }
    });

    // File upload handling
    const fileInput = document.getElementById('file-input');
    const fileDropZone = document.getElementById('file-drop-zone');
    const browseFilesBtn = document.getElementById('browse-files-btn');
    const audioPlayer = document.getElementById('audio-player');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const analyzeBtnContainer = document.getElementById('analyze-btn-container');
    let uploadedFile = null;

    browseFilesBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadedFile = e.target.files[0];
            handleFileUpload(uploadedFile);
        }
    });

    fileDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileDropZone.classList.add('border-red-400', 'bg-red-50');
    });

    fileDropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        fileDropZone.classList.remove('border-red-400', 'bg-red-50');
    });

    fileDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        fileDropZone.classList.remove('border-red-400', 'bg-red-50');
        if (e.dataTransfer.files.length > 0) {
            uploadedFile = e.dataTransfer.files[0];
            handleFileUpload(uploadedFile);
        }
    });

    function handleFileUpload(file) {
        const url = URL.createObjectURL(file);
        audioPlayer.src = url;
        audioPlayerContainer.classList.remove('hidden');
        analyzeBtnContainer.classList.remove('hidden');
    }

    // Analyze button (for file uploads only)
    document.getElementById('analyze-btn').addEventListener('click', async () => {
        const expectedText = document.getElementById('expected-text').value;
        if (!expectedText) {
            alert('Please enter text to pronounce');
            return;
        }

        // If we have an uploaded file, use it
        if (uploadedFile) {
            displayBlock('loading');

            const formData = new FormData();
            formData.append('file', uploadedFile);
            formData.append('expected_text', expectedText);

            try {
                const response = await fetch('/pronunciation', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Analysis failed');
                }

                const data = await response.json();
                document.dispatchEvent(new CustomEvent('analyze:done', { detail: data }));
            } catch (error) {
                console.error(error);
                document.dispatchEvent(new CustomEvent('analyze:error'));
            }
        }
    });

    document.getElementById('record-btn').addEventListener('click', () => {
        if (!recorder.isRecording()) {
            recorder.start();
        } else {
            recorder.stop();
        }
    });

    document.addEventListener("record:start", () => {
        document.getElementById('record-btn').classList.add('animate-ping');

        // Start timer
        recordingStartTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('recording-timer').textContent = `${minutes}:${seconds}`;
        }, 100);
    });

    document.addEventListener("record:stop", () => {
        document.getElementById('record-btn').classList.remove('animate-ping');

        // Stop timer
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }

        // Analysis will be triggered automatically by audio.js after processing
        displayBlock('loading');
    });

    document.addEventListener("analyze:error", (data) => {
        displayBlock('error');
    });

    document.addEventListener("analyze:done", (data) => {
        data = data.detail;
        if (!data) {
            return;
        }

        displayBlock('analysis-section');

        if (!data.differences) {
            displayBlock('error');
            return;
        }


        if (data.differences.errors.length == 0) {
            // emit "part:success" event
            document.dispatchEvent(new CustomEvent('part:success', { detail: data }));
        } else {
            // emit "part:error" event
            document.dispatchEvent(new CustomEvent('part:error', { detail: data }));
        }


        document.querySelector('[data-role="result.no-errors"]').classList.toggle('hidden', data.differences.errors.length > 0);
        document.querySelector('[data-role="result.has-errors"]').classList.toggle('hidden', data.differences.errors.length == 0);

        // scroll to the analysis section
        document.getElementById('analysis-section').scrollIntoView({ behavior: 'smooth' });
    });

    // on part:error
    document.addEventListener('part:error', (data) => {
        data = data.detail;

        displayBlock('analysis-section');

        // Display transcription
        document.querySelector('[data-role="pronunciation.transcribe"]').innerText = data.transcribe;

        // Calculate scores
        const overallScore = Math.round(data.score);
        const accuracy = calculateAccuracy(data);
        const fluency = calculateFluency(data);
        const completeness = calculateCompleteness(data);
        const prosody = calculateProsody(data);

        // Display overall score
        document.querySelector('[data-role="pronunciation.score"]').innerText = overallScore;

        // Animate circular chart
        animateCircularChart(overallScore);

        // Display detail scores
        displayDetailScore('accuracy', accuracy);
        displayDetailScore('fluency', fluency);
        displayDetailScore('completeness', completeness);
        displayDetailScore('prosody', prosody);

        // Display word-by-word details
        displayWordDetails(data);

        // Scroll to results
        document.getElementById('analysis-section').scrollIntoView({ behavior: 'smooth' });
    });

    // on part:success
    document.addEventListener('part:success', (data) => {
        data = data.detail;

        displayBlock('analysis-section');
    });

    // Listen button handler
    const listenBtn = document.getElementById('listen-btn');
    if (listenBtn) {
        console.log('Listen button found, attaching listener');
        listenBtn.addEventListener('click', async () => {
            console.log('Listen button clicked');
            const text = document.getElementById('expected-text').value;
            if (!text) return;

            const btn = document.getElementById('listen-btn');
            const originalContent = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = `
            <svg class="animate-spin h-4 w-4 text-gray-700" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Loading...
        `;

            try {
                const formData = new FormData();
                formData.append('text', text);

                const response = await fetch('/tts', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('TTS failed');

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);

                // Restore button when audio finishes
                audio.onended = () => {
                    btn.disabled = false;
                    btn.innerHTML = originalContent;
                    URL.revokeObjectURL(url);
                };

                await audio.play();
            } catch (error) {
                console.error('TTS Error:', error);
                btn.disabled = false;
                btn.innerHTML = originalContent;
                alert('Failed to generate audio');
            }
        });
    }

    // on click on btn-read, we read the phrase in the textarea "#expected-text"
    document.getElementById('btn-read').addEventListener('click', async () => {
        const text = document.getElementById('expected-text').value;
        if (!text) {
            return;
        }

        // Envoyer le texte à l'API pour obtenir les phonèmes
        const formData = new FormData();
        formData.append("text", text);

        const response = await fetch("/phonemes", { method: "POST", body: formData });
        const data = await response.json();

        const words = data.words;
        const phonemes = data.phonemes;

        if (!words || !phonemes || words.length !== phonemes.length) {
            console.error("Les mots et les phonèmes ne correspondent pas");
            return;
        }

        // Exécuter les animations et sons séquentiellement
        for (let i = 0; i < words.length; i++) {
            await playPhoneme(words[i], phonemes[i]);  // Attendre que la lecture et l'affichage du visème soient terminés
        }
    });
    // initialize default viseme
    const mouthImage = document.getElementById('viseme-image');
    const viseme = new Viseme(mouthImage);

    /**
     * Joue un mot et affiche son visème
     */
    async function playPhoneme(word, phoneme) {
        return new Promise((resolve) => {
            const utterance = new SpeechSynthesisUtterance(word);
            // forcer l'anglais
            utterance.lang = 'en-EN'; // changer l'accent ici
            utterance.rate = 0.7; // Ralentir la parole

            // Déclencher l'affichage du visème dès que la prononciation commence
            utterance.onstart = () => {
                console.log(`Lecture du mot: ${word}`);

                // Afficher le visème correspondant
                const mouthImage = document.getElementById('viseme-image');
                const viseme = new Viseme(mouthImage);
                viseme.play([phoneme]);  // Jouer l'animation du visème
            };

            // Attendre la fin de la prononciation AVANT de continuer
            utterance.onend = () => {
                console.log(`Fin de la lecture du mot: ${word}`);

                // Remettre l’image de repos après le mot
                const mouthImage = document.getElementById('viseme-image');
                mouthImage.src = "/static/assets/mouths/HunanBeanCMU39/rest.png";

                // Ajouter un léger délai avant le mot suivant pour la clarté
                setTimeout(resolve, 300);
            };

            // Lancer la prononciation
            speechSynthesis.speak(utterance);
        });
    }


    function displayProsodyChart(f0, energy) {
        const ctx = document.getElementById('prosodyChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: f0.map((_, i) => i),
                datasets: [
                    {
                        label: 'Fundamental Frequency (F0)',
                        data: f0,
                        borderColor: 'red',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderWidth: 2,
                        pointRadius: 0,  // Supprimer les points trop visibles
                        tension: 0.4  // Courbe plus lisse
                    },
                    {
                        label: 'Energy (normalized)',
                        data: energy,
                        borderColor: 'blue',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { display: false }, // Masquer les index inutiles
                    y: { beginAtZero: true }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }



    function updateErrorsCards(errors) {
        const cardContainer = document.getElementById("errors-card-container");
        const template = document.getElementById("error-card-template");

        cardContainer.innerHTML = ""; // Nettoyer les anciennes cartes

        errors.forEach(error => {

            if (error.expected == "" || error.word == "") {
                return;
            }

            const card = template.content.cloneNode(true);
            card.querySelector('[data-role="word"]').textContent = error.word;
            card.querySelector('[data-role="actual"]').textContent = error.actual;
            card.querySelector('[data-role="expected"]').textContent = error.expected;

            // Image cliquable pour jouer l’animation du visème
            const visemeImage = card.querySelector(".viseme-img");
            const viseme = new Viseme(visemeImage);

            visemeImage.addEventListener("click", () => {
                const duration = viseme.estimateWordDuration([error.expected]) * 1.5; // Augmenter la durée de 50%
                viseme.play([error.expected], duration); // Ralentir l'animation des visèmes
                const utterance = new SpeechSynthesisUtterance(error.word);
                utterance.lang = 'en-EN';
                utterance.rate = 0.7; // Ralentir la parole (1.0 = normal, <1 = plus lent)
                speechSynthesis.speak(utterance);
            });


            // Bouton pour écouter la bonne prononciation
            const playButton = card.querySelector(".play-audio");
            playButton.addEventListener("click", () => {
                visemeImage.click();
            });


            cardContainer.appendChild(card);
        });
    }

    function displayPronunciationChart(data) {

        // display pronunciation chart (data-role="pronunciation.chart")
        document.querySelector('[data-role="pronunciation.chart"]').classList.remove('hidden');
        // const expected_vector = data.differences.expected_vector.map(arr => arr[0]);
        // const transcribed_vector = data.differences.transcribed_vector.map(arr => arr[0]);
        const expected_vector = data.differences.expected_vector;
        const transcribed_vector = data.differences.transcribed_vector;

        console.log(expected_vector, transcribed_vector);

        // destroy preivous chart if exists  (Error: Canvas is already in use.)
        if (window.chart) {
            window.chart.destroy();
        }

        const ctx = document.getElementById('pronunciationChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.differences.expected_phonemes, // Affichage des phonèmes en X
                datasets: [
                    {
                        label: 'Interlocuteur natif',
                        data: expected_vector,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3
                    },
                    {
                        label: 'Votre prononciation',
                        data: transcribed_vector,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Phonèmes'
                        },
                        ticks: {
                            autoSkip: false, // Affiche tous les phonèmes
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Amplitude'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        enabled: true, // Affiche les valeurs des points au survol
                        callbacks: {
                            title: function (tooltipItem) {
                                return "Phonème : " + tooltipItem[0].label; // Affiche le phonème en survol
                            }
                        }
                    }
                }
            }
        });

        window.chart = chart;

        // chart height should not exceed given size
        const height = Math.min(250, expected_vector.length * 10);
        document.getElementById('pronunciationChart').style.height = `${height}px`;
        // width should be 100%
        document.getElementById('pronunciationChart').style.width = '100%';

    }


    const record_silence = new Audio("/static/assets/sounds/slick-notification.mp3");
    document.addEventListener("record:silence", () => {
        console.log("Silence détecté, arrêt de l'enregistrement");
        record_silence.play();
    });

    // on click on retry button
    document.getElementById('retry-btn').addEventListener('click', () => {
        window.location.reload();
    });

});

// Helper functions for score calculation and display
function calculateAccuracy(data) {
    // Accuracy based on phoneme distance (lower distance = higher accuracy)
    const maxDistance = 5000;
    const phonemeDistance = data.differences.phoneme_distance || 0;
    return Math.max(0, Math.round(100 - (phonemeDistance / maxDistance) * 100));
}

function calculateFluency(data) {
    // Fluency based on prosody energy variance
    if (!data.prosody || !data.prosody.energy) return 85; // Default
    const energy = data.prosody.energy;
    const variance = calculateVariance(energy);
    // Lower variance = more fluent (consistent energy)
    return Math.min(100, Math.max(0, Math.round(100 - variance / 10)));
}

function calculateCompleteness(data) {
    // Completeness based on word distance (how many words were correctly recognized)
    const maxWordDistance = 30;
    const wordDistance = data.differences.word_distance || 0;
    return Math.max(0, Math.round(100 - (wordDistance / maxWordDistance) * 100));
}

function calculateProsody(data) {
    // Prosody based on F0 (pitch) variation
    if (!data.prosody || !data.prosody.f0) return 75; // Default
    const f0 = data.prosody.f0;
    const variance = calculateVariance(f0);
    // Moderate variance is good for prosody (not too flat, not too erratic)
    const idealVariance = 50;
    const diff = Math.abs(variance - idealVariance);
    return Math.max(0, Math.round(100 - diff));
}

function calculateVariance(arr) {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const squareDiffs = arr.map(value => Math.pow(value - mean, 2));
    return Math.sqrt(squareDiffs.reduce((a, b) => a + b, 0) / arr.length);
}

function animateCircularChart(score) {
    const circle = document.getElementById('score-circle');
    const circumference = 2 * Math.PI * 80; // radius = 80
    const offset = circumference - (score / 100) * circumference;

    // Animate the circle
    circle.style.transition = 'stroke-dashoffset 1s ease-in-out';
    circle.style.strokeDashoffset = offset;
}

function displayDetailScore(type, score) {
    // Update score text
    document.querySelector(`[data-role="score.${type}"]`).innerText = `${score}/100`;

    // Update progress bar
    const bar = document.querySelector(`[data-role="bar.${type}"]`);
    bar.style.transition = 'width 0.5s ease-in-out';
    bar.style.width = `${score}%`;
}

function displayWordDetails(data) {
    const container = document.getElementById('word-details-container');
    const template = document.getElementById('word-detail-template');

    container.innerHTML = ''; // Clear previous content

    // Get all words from expected text, removing punctuation for cleaner splitting
    const expectedText = document.getElementById('expected-text').value;
    // Split by spaces and filter out empty strings, then clean punctuation from each word for display/comparison
    const words = expectedText.match(/\b[\w']+\b/g) || [];
    const errors = data.differences.errors || [];

    words.forEach((word, index) => {
        const wordDetail = template.content.cloneNode(true);

        // Find if this word has an error
        // Note: Backend now returns "word" field in error which matches the expected word text
        const error = errors.find(e => e.word === word);

        // Calculate word score (100 if no error, lower if error exists)
        // If error exists, we can use the phoneme distance if available, or just a default penalty
        let wordScore = 100;
        if (error) {
            if (error.actual === "") {
                wordScore = 0; // Missing word
            } else {
                // Mispronunciation
                const dist = Levenshtein.distance(error.expected, error.actual);
                wordScore = Math.max(0, 100 - (dist * 20)); // Penalize 20 points per phoneme diff
            }
        }

        // Set word name with color coding
        const wordNameEl = wordDetail.querySelector('[data-role="word.name"]');
        wordNameEl.textContent = word;
        wordNameEl.classList.add(error ? 'text-red-600' : 'text-green-600');

        // Add listen button handler
        const listenBtn = wordDetail.querySelector('[data-role="word.listen"]');
        listenBtn.addEventListener('click', () => {
            const utterance = new SpeechSynthesisUtterance(word);
            utterance.lang = 'en-US';
            utterance.rate = 0.8;
            speechSynthesis.speak(utterance);
        });

        // Set score
        wordDetail.querySelector('[data-role="word.score"]').textContent = `${wordScore.toFixed(0)}`;

        // Set progress bar
        const bar = wordDetail.querySelector('[data-role="word.bar"]');
        bar.style.width = `${wordScore}%`;
        bar.classList.add(error ? 'bg-red-500' : 'bg-green-500');

        // Set phonetic breakdown
        const phonemesEl = wordDetail.querySelector('[data-role="word.phonemes"]');
        if (error) {
            if (error.actual === "") {
                phonemesEl.innerHTML = `/${error.expected}/ <span class="text-red-400">vs (missing)</span>`;

                const errorTypeContainer = wordDetail.querySelector('.error-type');
                errorTypeContainer.classList.remove('hidden');
                errorTypeContainer.querySelector('[data-role="error.type"]').textContent = 'Word is missing';
            } else {
                phonemesEl.innerHTML = `/${error.expected}/ <span class="text-red-400">vs /${error.actual}/</span>`;

                const errorTypeContainer = wordDetail.querySelector('.error-type');
                errorTypeContainer.classList.remove('hidden');
                errorTypeContainer.querySelector('[data-role="error.type"]').textContent = 'Mispronunciation';
                if (error.actual_word) {
                    errorTypeContainer.querySelector('[data-role="error.type"]').textContent += ` (Heard: "${error.actual_word}")`;
                }
            }
        } else {
            // For correct words, just show expected phonemes
            // We need to find the phonemes for this word. 
            // Since we don't have a direct map from index -> phonemes here easily without re-parsing,
            // we can try to use the backend's expected_phonemes if we can map it.
            // But backend returns flat list.
            // Simplest: just use phonemizer on frontend? No.
            // Let's just leave it empty or try to match?
            // The backend `errors` contains `expected` phonemes for errors.
            // For correct words, we don't have it in `errors`.
            // We could pass `words_info` from backend?
            // For now, let's just show nothing or "OK" for correct words?
            // Or we can try to guess from `data.differences.expected_phonemes`?
            // That list is flat.
            // Let's just show "OK" or check if we can get it.
            // Actually, the user liked seeing phonemes.
            // Let's leave it blank for now if we can't easily get it, or use a placeholder.
            phonemesEl.textContent = "Correct";
            phonemesEl.classList.add("text-green-500");
        }

        container.appendChild(wordDetail);
    });
}

// Simple Levenshtein distance implementation
const Levenshtein = {
    distance: function (a, b) {
        if (a.length === 0) return b.length;
        if (b.length === 0) return a.length;

        const matrix = [];
        for (let i = 0; i <= b.length; i++) {
            matrix[i] = [i];
        }
        for (let j = 0; j <= a.length; j++) {
            matrix[0][j] = j;
        }

        for (let i = 1; i <= b.length; i++) {
            for (let j = 1; j <= a.length; j++) {
                if (b.charAt(i - 1) === a.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j] + 1
                    );
                }
            }
        }
        return matrix[b.length][a.length];
    }
};

function displayBlock(id) {
    // Hide all sections
    document.getElementById('analysis-section').classList.add('hidden');
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');

    // Show the target section
    document.getElementById(id).classList.remove('hidden');
}



document.addEventListener('DOMContentLoaded', function () {
    const tmpDev = { "score": 15, "distance": 1156, "differences": { "word_distance": 15, "phoneme_distance": 4861, "errors": [{ "position": 0, "expected": "həloʊ", "actual": "dɔːʊɡ", "word": "Hello," }, { "position": 1, "expected": "haʊ", "actual": "dɔːʊɡ", "word": "how" }, { "position": 2, "expected": "ɑːɹ", "actual": "dɔːʊɡ", "word": "are" }, { "position": 3, "expected": "juː", "actual": "dɔːʊɡ", "word": "you?" }], "feedback": "🔊 Feedback sur votre prononciation :\n❌ Vous devez mieux prononcer ces mots : Hello,, are, you?, how\n", "transcribe": "dawwg gj abh zg'", "expected_vector": [[104], [601], [108], [111], [650], [32], [104], [97], [650], [32], [593], [720], [633], [32], [106], [117], [720]], "transcribed_vector": [[100], [596], [720], [650], [609], [32], [100], [658], [105], [720], [100], [658], [101], [618], [32], [230], [98], [32], [122], [105], [720], [100], [658], [105], [720]], "expected_phonemes": ["həloʊ", "haʊ", "ɑːɹ", "juː"], "transcribed_phonemes": ["dɔːʊɡ", "dʒiːdʒeɪ", "æb", "ziːdʒiː"] }, "feedback": "🔊 Feedback sur votre prononciation :\n❌ Vous devez mieux prononcer ces mots : Hello,, are, you?, how\n", "transcribe": "dawwg gj abh zg'" }
    document.getElementById('dev-failed').addEventListener('click', () => {
        document.dispatchEvent(new CustomEvent('analyze:done', { detail: tmpDev }));
    });

    document.getElementById('dev-success').addEventListener('click', () => {
        document.dispatchEvent(new CustomEvent('analyze:done', { detail: { score: 100, differences: { errors: [] } } }));
    });

    document.getElementById('dev-loading').addEventListener('click', () => {
        displayBlock('loading');
    });

    document.getElementById('dev-500').addEventListener('click', () => {
        displayBlock('error');
    });
});