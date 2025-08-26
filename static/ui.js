document.addEventListener('DOMContentLoaded', function() {
    const recorder = new AudioRecorder();

    document.getElementById('record-btn').addEventListener('click', () => {
        if(!recorder.isRecording()) {
            recorder.start();
        } else {
            recorder.stop();
        }
    });

    document.addEventListener("record:start", () => {
        document.getElementById('record-btn').classList.add('animate-ping');
        document.getElementById('analysis-section').classList.remove('visible-section');
    });

    document.addEventListener("record:stop", () => {
        document.getElementById('record-btn').classList.remove('animate-ping');

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
            // hide error after 3 sec
            setTimeout(() => {
                displayBlock('speaking');
            }, 3000);
            return;
        }


        if(data.differences.errors.length == 0) {
            // emit "part:success" event
            document.dispatchEvent(new CustomEvent('part:success', {detail: data}));
        } else {
            // emit "part:error" event
            document.dispatchEvent(new CustomEvent('part:error', {detail: data}));
        }


        document.querySelector('[data-role="result.no-errors"]').classList.toggle('hidden', data.differences.errors.length > 0);
        document.querySelector('[data-role="result.has-errors"]').classList.toggle('hidden', data.differences.errors.length == 0);

        // scroll to the analysis section
        document.getElementById('analysis-section').scrollIntoView({behavior: 'smooth'});
    });

    // on part:error
    document.addEventListener('part:error', (data) => {
        data = data.detail;

        displayBlock('analysis-section');


        displayPronunciationChart(data);
        document.querySelector('[data-role="pronunciation.transcribe"]').innerText = data.transcribe;
        document.querySelector('[data-role="pronunciation.score"]').innerText = data.score;

        if (data.prosody) {
            displayProsodyChart(data.prosody.f0, data.prosody.energy);
        }

        // Mettre à jour le tableau des erreurs
        updateErrorsCards(data.differences.errors);

        // Display pronunciation chart
        document.getElementById('pronunciationChart').scrollIntoView({behavior: 'smooth'});
    });

    // on part:success
    document.addEventListener('part:success', (data) => {
        data = data.detail;

        displayBlock('analysis-section');
    });

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

            if(error.expected == "" || error.word == "") {
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
                            title: function(tooltipItem) {
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
        displayBlock('speaking');
    });

});

function displayBlock(id) {
    // smoothy hide all sections, then display the target section
    document.getElementById('analysis-section').classList.add('hidden');
    document.getElementById('speaking').classList.add('hidden');
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');

    document.getElementById('analysis-section').classList.add('hidden-section');
    document.getElementById('speaking').classList.add('hidden-section');
    document.getElementById('loading').classList.add('hidden-section');
    document.getElementById('error').classList.add('hidden-section');

    document.getElementById(id).classList.remove('hidden');
    document.getElementById(id).classList.remove('hidden-section');
    document.getElementById(id).classList.add('visible-section');

    // for all except speaking section, show retry button
    if (id !== 'speaking') {
        document.getElementById('btn-retry').classList.remove('hidden');
    } else {
        document.getElementById('btn-retry').classList.add('hidden');
    }
}



document.addEventListener('DOMContentLoaded', function() {
    const tmpDev = {"score":15,"distance":1156,"differences":{"word_distance":15,"phoneme_distance":4861,"errors":[{"position":0,"expected":"həloʊ","actual":"dɔːʊɡ","word":"Hello,"},{"position":1,"expected":"haʊ","actual":"dɔːʊɡ","word":"how"},{"position":2,"expected":"ɑːɹ","actual":"dɔːʊɡ","word":"are"},{"position":3,"expected":"juː","actual":"dɔːʊɡ","word":"you?"}],"feedback":"🔊 Feedback sur votre prononciation :\n❌ Vous devez mieux prononcer ces mots : Hello,, are, you?, how\n","transcribe":"dawwg gj abh zg'","expected_vector":[[104],[601],[108],[111],[650],[32],[104],[97],[650],[32],[593],[720],[633],[32],[106],[117],[720]],"transcribed_vector":[[100],[596],[720],[650],[609],[32],[100],[658],[105],[720],[100],[658],[101],[618],[32],[230],[98],[32],[122],[105],[720],[100],[658],[105],[720]],"expected_phonemes":["həloʊ","haʊ","ɑːɹ","juː"],"transcribed_phonemes":["dɔːʊɡ","dʒiːdʒeɪ","æb","ziːdʒiː"]},"feedback":"🔊 Feedback sur votre prononciation :\n❌ Vous devez mieux prononcer ces mots : Hello,, are, you?, how\n","transcribe":"dawwg gj abh zg'"}
    document.getElementById('dev-failed').addEventListener('click', () => {
        document.dispatchEvent(new CustomEvent('analyze:done', {detail: tmpDev}));
    });

    document.getElementById('dev-success').addEventListener('click', () => {
        document.dispatchEvent(new CustomEvent('analyze:done', {detail: {score: 100, differences: {errors: []}}}));
    });

    document.getElementById('dev-loading').addEventListener('click', () => {
        displayBlock('loading');
    });

    document.getElementById('dev-500').addEventListener('click', () => {
        displayBlock('error');
    });
});