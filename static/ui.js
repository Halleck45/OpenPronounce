/**
 * OpenPronounce web demo.
 *
 * One page, three states: practice (record / upload), loading, result (or error).
 * All server calls live here; audio.js only records, viseme.js only animates the mouth.
 */
(function () {
    'use strict';

    const $ = (id) => document.getElementById(id);

    // BCP 47 tags for the browser SpeechSynthesis, keyed by server language code
    const SPEECH_LANGS = {
        en: 'en-US', fr: 'fr-FR', es: 'es-ES', de: 'de-DE', it: 'it-IT', pt: 'pt-BR', nl: 'nl-NL',
    };

    const EXAMPLES = {
        en: ['Hello, how are you?', 'I would like a cup of coffee.', 'The weather is lovely today.'],
        fr: ['Bonjour, comment allez-vous ?', 'Je voudrais un café, s\'il vous plaît.', 'Il fait beau aujourd\'hui.'],
        es: ['Hola, ¿cómo estás?', 'Me gustaría un café, por favor.', 'Hoy hace muy buen tiempo.'],
        de: ['Hallo, wie geht es dir?', 'Ich hätte gern einen Kaffee.', 'Das Wetter ist heute schön.'],
        it: ['Ciao, come stai?', 'Vorrei un caffè, per favore.', 'Oggi il tempo è bello.'],
        pt: ['Olá, como você está?', 'Eu gostaria de um café, por favor.', 'O tempo está bonito hoje.'],
        nl: ['Hallo, hoe gaat het?', 'Ik wil graag een kopje koffie.', 'Het weer is vandaag mooi.'],
    };

    // Same tokenization as the server (\b[\w']+\b with Unicode word characters)
    const WORD_RE = /[\p{L}\p{N}_](?:[\p{L}\p{N}_']*[\p{L}\p{N}_])?/gu;
    const RING_LENGTH = 2 * Math.PI * 62;
    const PHONE_WRONG_THRESHOLD = 0.5;
    const COLORS = { you: '#ea580c', reference: '#2f6fb3', good: '#10b981', mid: '#f59e0b', bad: '#ef4444' };
    const LANG_STORAGE_KEY = 'openpronounce.lang';

    const RECORD_ERRORS = {
        insecure: 'Recording needs a secure page (https or localhost). You can still upload a file.',
        denied: 'Microphone access was denied. Allow it in your browser, or upload a file.',
        notfound: 'No microphone found. Plug one in, or upload a file.',
        unknown: 'The microphone could not be started. Try again, or upload a file.',
    };

    const LOADING_STEPS = [
        [0, 'Listening to your recording'],
        [4000, 'Comparing your sounds with the reference'],
        [12000, 'Still working. The first run loads the models, later runs are faster'],
    ];

    const state = {
        lang: 'en',
        blob: null,
        filename: null,
        result: null,
        words: [],
        errorsByIndex: new Map(),
        selected: null,
        charts: {},
        chartsDirty: false,
        busy: false,
        loadingTimers: [],
    };

    let recorder;
    let viseme;
    let timerInterval = null;
    const silenceSound = new Audio('/static/assets/sounds/slick-notification.mp3');

    document.addEventListener('DOMContentLoaded', init);

    function init() {
        recorder = new AudioRecorder();
        viseme = new Viseme($('viseme-image'));

        loadLanguages();
        renderExamples();

        $('language-select').addEventListener('change', onLanguageChange);
        $('expected-text').addEventListener('input', hideTextHint);

        $('record-btn').addEventListener('click', onRecordClick);
        document.addEventListener('record:start', onRecordStart);
        document.addEventListener('record:stop', onRecordStop);
        document.addEventListener('record:silence', () => silenceSound.play().catch(() => { }));
        document.addEventListener('record:ready', (e) => {
            setAudio(e.detail.blob, 'recording.webm', false);
            analyze();
        });
        document.addEventListener('record:error', (e) => {
            showRecordError(RECORD_ERRORS[e.detail && e.detail.reason] || RECORD_ERRORS.unknown);
        });

        $('upload-btn').addEventListener('click', () => $('file-input').click());
        $('file-input').addEventListener('change', (e) => {
            if (e.target.files.length) {
                setAudio(e.target.files[0], e.target.files[0].name, true);
            }
        });
        setupDragAndDrop();

        $('analyze-btn').addEventListener('click', analyze);
        $('listen-btn').addEventListener('click', playReference);
        $('retry-btn').addEventListener('click', () => setView('idle'));
        $('again-btn').addEventListener('click', resetForNewTake);
        $('replay-btn').addEventListener('click', () => $('audio-player').play().catch(() => { }));
        $('detail-listen').addEventListener('click', () => {
            if (state.selected !== null) {
                speakWord(state.selected);
            }
        });
        $('details').addEventListener('toggle', () => {
            if ($('details').open && state.chartsDirty) {
                renderCharts();
            }
        });
    }

    // ---------------------------------------------------------------- languages

    async function loadLanguages() {
        try {
            const response = await fetch('/languages');
            if (!response.ok) {
                return;
            }
            const data = await response.json();
            const select = $('language-select');
            select.innerHTML = '';
            data.languages.forEach(({ code, name }) => {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = name;
                select.appendChild(option);
            });
            const codes = data.languages.map(l => l.code);
            const saved = localStorage.getItem(LANG_STORAGE_KEY);
            state.lang = codes.includes(saved) ? saved : (codes.includes(data.default) ? data.default : codes[0]);
            select.value = state.lang;
            renderExamples();
        } catch (err) {
            // Language list is a nicety; English still works without it
        }
    }

    function onLanguageChange(e) {
        const previous = state.lang;
        state.lang = e.target.value;
        localStorage.setItem(LANG_STORAGE_KEY, state.lang);

        // If the sentence was one of the previous examples, swap it for the first example of the new language
        const textarea = $('expected-text');
        const previousExamples = EXAMPLES[previous] || [];
        if (previousExamples.includes(textarea.value.trim()) && (EXAMPLES[state.lang] || []).length) {
            textarea.value = EXAMPLES[state.lang][0];
        }
        renderExamples();
    }

    function renderExamples() {
        const container = $('examples');
        container.querySelectorAll('button').forEach(b => b.remove());
        (EXAMPLES[state.lang] || EXAMPLES.en).forEach(text => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.textContent = text;
            chip.className = 'text-xs px-2.5 py-1 rounded-full border border-neutral-200 bg-white text-neutral-600 '
                + 'hover:border-accent-500 hover:text-accent-700 transition-colors';
            chip.addEventListener('click', () => {
                $('expected-text').value = text;
                hideTextHint();
            });
            container.appendChild(chip);
        });
    }

    // ---------------------------------------------------------------- text helpers

    function expectedText() {
        return $('expected-text').value.trim();
    }

    function tokenize(text) {
        return text.match(WORD_RE) || [];
    }

    function requireText() {
        if (!tokenize(expectedText()).length) {
            showTextHint('Type the sentence first, or pick one of the examples.');
            $('expected-text').focus();
            return false;
        }
        return true;
    }

    function showTextHint(message) {
        const hint = $('text-hint');
        hint.textContent = message;
        hint.classList.remove('hidden');
    }

    function hideTextHint() {
        $('text-hint').classList.add('hidden');
    }

    function speechLang() {
        return SPEECH_LANGS[state.lang] || state.lang;
    }

    // ---------------------------------------------------------------- recording

    function onRecordClick() {
        if (recorder.isRecording()) {
            recorder.stop();
            return;
        }
        if (state.busy || !requireText()) {
            return;
        }
        hideRecordError();
        recorder.start();
    }

    function onRecordStart() {
        const btn = $('record-btn');
        btn.classList.add('is-recording');
        btn.setAttribute('aria-label', 'Stop recording');
        $('mic-icon').classList.add('hidden');
        $('stop-icon').classList.remove('hidden');
        $('record-hint').textContent = 'Listening. Tap to stop, or pause for two seconds.';
        $('record-timer').classList.remove('hidden');
        $('audio-review').classList.add('hidden');
        setView('idle');

        const startedAt = Date.now();
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - startedAt) / 1000);
            const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
            const ss = String(elapsed % 60).padStart(2, '0');
            $('record-timer').textContent = `${mm}:${ss}`;
        }, 200);
    }

    function onRecordStop() {
        const btn = $('record-btn');
        btn.classList.remove('is-recording');
        btn.setAttribute('aria-label', 'Record');
        $('mic-icon').classList.remove('hidden');
        $('stop-icon').classList.add('hidden');
        $('record-hint').textContent = 'Tap to record, then say the sentence';
        $('record-timer').classList.add('hidden');
        $('record-timer').textContent = '00:00';
        clearInterval(timerInterval);
        timerInterval = null;
    }

    function showRecordError(message) {
        const el = $('record-error');
        el.textContent = message;
        el.classList.remove('hidden');
    }

    function hideRecordError() {
        $('record-error').classList.add('hidden');
    }

    // ---------------------------------------------------------------- audio input (recorded or uploaded)

    function setAudio(blob, filename, needsConfirmation) {
        state.blob = blob;
        state.filename = filename;
        const player = $('audio-player');
        if (player.dataset.url) {
            URL.revokeObjectURL(player.dataset.url);
        }
        const url = URL.createObjectURL(blob);
        player.src = url;
        player.dataset.url = url;
        $('file-name').textContent = needsConfirmation ? filename : '';
        $('analyze-btn').classList.toggle('hidden', !needsConfirmation);
        $('audio-review').classList.remove('hidden');
        hideRecordError();
    }

    function setupDragAndDrop() {
        const zone = $('practice');
        const overlay = $('drop-overlay');
        let depth = 0;
        const show = (on) => {
            overlay.classList.toggle('hidden', !on);
            overlay.classList.toggle('flex', on);
        };
        zone.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; show(true); });
        zone.addEventListener('dragover', (e) => { e.preventDefault(); });
        zone.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) show(false); });
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            depth = 0;
            show(false);
            const file = e.dataTransfer.files && e.dataTransfer.files[0];
            if (file) {
                setAudio(file, file.name, true);
            }
        });
    }

    // ---------------------------------------------------------------- analysis

    async function analyze() {
        if (state.busy || !state.blob || !requireText()) {
            return;
        }
        state.busy = true;
        $('record-btn').disabled = true;
        $('analyze-btn').disabled = true;
        setView('loading');

        const formData = new FormData();
        formData.append('file', state.blob, state.filename || 'audio.webm');
        formData.append('expected_text', expectedText());
        formData.append('lang', state.lang);

        try {
            const response = await fetch('/pronunciation', { method: 'POST', body: formData });
            if (!response.ok) {
                let detail = '';
                try {
                    detail = (await response.json()).detail || '';
                } catch (err) { /* not JSON */ }
                throw new Error(response.status === 422 && detail
                    ? detail
                    : 'The server could not analyze this recording. Try again in a moment.');
            }
            const data = await response.json();
            if (!data || !data.differences) {
                throw new Error('The server returned an unexpected answer. Try again in a moment.');
            }
            renderResult(data);
        } catch (err) {
            showError(err && err.message ? err.message : 'Could not reach the server. Check your connection and try again.');
        } finally {
            state.busy = false;
            $('record-btn').disabled = false;
            $('analyze-btn').disabled = false;
        }
    }

    function setView(name) {
        state.loadingTimers.forEach(clearTimeout);
        state.loadingTimers = [];
        $('loading').classList.toggle('hidden', name !== 'loading');
        $('error').classList.toggle('hidden', name !== 'error');
        $('result').classList.toggle('hidden', name !== 'result');

        if (name === 'loading') {
            LOADING_STEPS.forEach(([delay, label]) => {
                state.loadingTimers.push(setTimeout(() => { $('loading-step').textContent = label; }, delay));
            });
            $('loading').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else if (name === 'error') {
            $('error').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function showError(message) {
        $('error-message').textContent = message;
        setView('error');
    }

    function resetForNewTake() {
        setView('idle');
        $('audio-review').classList.add('hidden');
        state.blob = null;
        state.filename = null;
        $('practice').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ---------------------------------------------------------------- result

    function renderResult(data) {
        state.result = data;
        state.words = tokenize(expectedText());
        state.selected = null;

        const errors = (data.differences.errors || []).filter(e => e && e.word !== '');
        state.errorsByIndex = new Map();
        errors.forEach(e => {
            if (typeof e.position === 'number' && !state.errorsByIndex.has(e.position)) {
                state.errorsByIndex.set(e.position, e);
            }
        });
        // Fallback for servers that do not send positions: match by name, first unmatched occurrence
        errors.filter(e => typeof e.position !== 'number').forEach(e => {
            const index = state.words.findIndex((w, i) => !state.errorsByIndex.has(i) && w.toLowerCase() === String(e.word).toLowerCase());
            if (index >= 0) {
                state.errorsByIndex.set(index, e);
            }
        });

        renderScore(Math.round(data.score || 0), state.errorsByIndex.size, state.words.length, data.transcribe);
        renderChips();
        renderHeard(data);

        $('word-detail').classList.add('hidden');
        state.chartsDirty = true;
        if ($('details').open) {
            renderCharts();
        }

        setView('result');
        const result = $('result');
        result.classList.remove('reveal');
        void result.offsetWidth;
        result.classList.add('reveal');
        result.scrollIntoView({ behavior: 'smooth', block: 'start' });

        const firstFlagged = state.words.findIndex((_, i) => state.errorsByIndex.has(i));
        if (firstFlagged >= 0) {
            selectWord(firstFlagged, false);
        }
    }

    function renderScore(score, flagged, total, transcribe) {
        const ring = $('score-ring');
        ring.style.stroke = score >= 80 ? COLORS.good : score >= 50 ? COLORS.mid : COLORS.bad;
        ring.style.strokeDashoffset = RING_LENGTH;
        requestAnimationFrame(() => {
            ring.style.strokeDashoffset = RING_LENGTH * (1 - Math.max(0, Math.min(100, score)) / 100);
        });
        countUp($('score-value'), score, 900);

        let verdict;
        let sub;
        if (total === 0) {
            verdict = 'Nothing to compare';
            sub = 'The sentence is empty.';
        } else if (flagged === 0 && score < 70) {
            verdict = 'No single word stands out';
            sub = transcribe
                ? `But the sentence as a whole was hard to follow: we heard "${String(transcribe).toLowerCase()}". Check that the recording started before you spoke.`
                : 'But the sentence as a whole was hard to follow.';
        } else if (flagged === 0) {
            verdict = 'Every word came through clearly';
            sub = 'Tap a word to see its sounds anyway.';
        } else if (flagged === total) {
            verdict = 'None of the words came through';
            sub = 'Tap a word to see which sounds went wrong. Was it the right sentence?';
        } else {
            verdict = `${flagged} of ${total} word${total > 1 ? 's' : ''} need${flagged === 1 ? 's' : ''} work`;
            sub = 'Tap a red word to see which sounds went wrong.';
        }
        $('verdict').textContent = verdict;
        $('verdict-sub').textContent = sub;
    }

    function countUp(el, target, duration) {
        const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduced || duration <= 0) {
            el.textContent = target;
            return;
        }
        const start = performance.now();
        const step = (now) => {
            const t = Math.min(1, (now - start) / duration);
            const eased = 1 - Math.pow(1 - t, 3);
            el.textContent = Math.round(target * eased);
            if (t < 1) {
                requestAnimationFrame(step);
            }
        };
        requestAnimationFrame(step);
    }

    function renderChips() {
        const container = $('word-chips');
        container.innerHTML = '';
        const expectedPhones = state.result.differences.expected_phones || [];

        state.words.forEach((word, index) => {
            const error = state.errorsByIndex.get(index);
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.dataset.index = index;
            chip.textContent = word;
            chip.setAttribute('aria-pressed', 'false');
            chip.className = 'chip px-3 py-1.5 rounded-lg border text-base font-medium ' + (error
                ? 'bg-red-50 border-red-200 text-red-700 hover:bg-red-100'
                : 'bg-emerald-50 border-emerald-200 text-emerald-800 hover:bg-emerald-100');
            if (error) {
                const conf = typeof error.confidence === 'number' ? `${Math.round(error.confidence * 100)} % sure` : 'flagged';
                chip.title = error.actual === ''
                    ? `Not heard (${conf}). Expected /${error.expected}/`
                    : `Flagged, ${conf}. Expected /${error.expected}/, heard /${error.actual}/`;
            } else {
                const phones = expectedPhones[index];
                chip.title = phones && phones.length ? `Sounds fine: /${phones.join(' ')}/` : 'Sounds fine';
            }
            chip.addEventListener('click', () => selectWord(index, true));
            container.appendChild(chip);
        });
    }

    function renderHeard(data) {
        const transcript = (data.transcribe || '').trim();
        $('transcript').textContent = transcript ? transcript.toLowerCase() : 'nothing we could recognize';

        const heard = data.differences.heard_phones || [];
        const confidences = data.differences.heard_phones_confidence || [];
        const container = $('heard-phones');
        container.innerHTML = '';
        if (!heard.length) {
            container.textContent = 'no phones recognized';
            container.className = 'text-sm text-neutral-500';
            return;
        }
        container.className = 'font-mono text-base leading-relaxed';
        heard.forEach((phone, i) => {
            const span = document.createElement('span');
            span.textContent = phone;
            const conf = confidences[i];
            const sure = typeof conf !== 'number' || conf >= 0.5;
            span.className = 'inline-block mr-1.5 ' + (sure ? 'text-neutral-800' : 'text-neutral-400');
            if (typeof conf === 'number') {
                span.title = `${Math.round(conf * 100)} % confidence`;
            }
            container.appendChild(span);
        });
    }

    function selectWord(index, speak) {
        state.selected = index;
        $('word-chips').querySelectorAll('.chip').forEach(chip => {
            chip.setAttribute('aria-pressed', String(Number(chip.dataset.index) === index));
        });

        const word = state.words[index];
        const error = state.errorsByIndex.get(index);
        const expectedPhones = (state.result.differences.expected_phones || [])[index] || [];
        const detail = $('word-detail');
        const status = $('detail-status');
        const expectedEl = $('detail-expected');
        const heardEl = $('detail-heard');
        const note = $('detail-note');

        $('detail-word').textContent = word;
        expectedEl.innerHTML = '';
        heardEl.innerHTML = '';
        note.textContent = '';

        if (!error) {
            status.textContent = 'Sounds fine';
            status.className = 'text-sm font-medium text-emerald-700';
            expectedEl.textContent = expectedPhones.length ? `/${expectedPhones.join(' ')}/` : '/?/';
            heardEl.textContent = 'the same';
            heardEl.className = 'text-base text-neutral-500';
        } else {
            const conf = typeof error.confidence === 'number' ? Math.round(error.confidence * 100) : null;
            status.textContent = error.actual === ''
                ? (conf === null ? 'Not heard' : `Not heard, ${conf} % sure`)
                : (conf === null ? 'Flagged' : `Flagged, ${conf} % sure`);
            status.className = 'text-sm font-medium text-red-600';

            const phones = Array.isArray(error.phones) && error.phones.length ? error.phones : null;
            expectedEl.appendChild(document.createTextNode('/'));
            if (phones) {
                phones.forEach((p, i) => {
                    const span = document.createElement('span');
                    span.textContent = p.expected;
                    const wrong = typeof p.confidence === 'number' && p.confidence >= PHONE_WRONG_THRESHOLD;
                    if (wrong) {
                        span.className = 'text-red-600 font-semibold underline decoration-2 decoration-red-300 underline-offset-4';
                        span.title = `heard /${p.heard || '∅'}/ (${Math.round(p.confidence * 100)} %)`;
                    }
                    expectedEl.appendChild(span);
                    if (i < phones.length - 1) {
                        expectedEl.appendChild(document.createTextNode(' '));
                    }
                });
            } else {
                expectedEl.appendChild(document.createTextNode(error.expected || expectedPhones.join(' ')));
            }
            expectedEl.appendChild(document.createTextNode('/'));

            if (error.actual === '') {
                heardEl.textContent = 'nothing';
                heardEl.className = 'text-base text-neutral-500';
            } else {
                const heardPhones = phones ? phones.map(p => p.heard).filter(Boolean).join(' ') : error.actual;
                heardEl.textContent = `/${heardPhones}/`;
                heardEl.className = 'font-mono text-lg tracking-wide text-neutral-800';
            }
            if (error.actual_word) {
                note.textContent = `The speech recognizer understood "${String(error.actual_word).toLowerCase()}".`;
            }
        }
        expectedEl.className = 'font-mono text-lg tracking-wide text-neutral-800';

        detail.classList.remove('hidden');
        if (speak) {
            speakWord(index);
        } else {
            viseme.rest();
        }
    }

    function speakWord(index) {
        const word = state.words[index];
        const error = state.errorsByIndex.get(index);
        const expectedPhones = (state.result.differences.expected_phones || [])[index] || [];
        const phones = expectedPhones.length ? expectedPhones : (error && error.expected ? [error.expected] : []);

        if (phones.length) {
            const duration = viseme.estimateWordDuration(phones) * 1.5;
            viseme.play(phones, duration);
        }
        if ('speechSynthesis' in window) {
            speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(word);
            utterance.lang = speechLang();
            utterance.rate = 0.8;
            speechSynthesis.speak(utterance);
        }
    }

    // ---------------------------------------------------------------- reference audio (TTS)

    async function playReference() {
        if (!requireText()) {
            return;
        }
        const btn = $('listen-btn');
        const label = btn.querySelector('[data-role="listen.label"]');
        const original = label.textContent;
        btn.disabled = true;
        label.textContent = 'Loading the reference';

        try {
            const formData = new FormData();
            formData.append('text', expectedText());
            formData.append('lang', state.lang);
            const response = await fetch('/tts', { method: 'POST', body: formData });
            if (!response.ok) {
                throw new Error('tts failed');
            }
            const url = URL.createObjectURL(await response.blob());
            const audio = new Audio(url);
            label.textContent = 'Playing';
            const done = () => {
                URL.revokeObjectURL(url);
                btn.disabled = false;
                label.textContent = original;
            };
            audio.onended = done;
            audio.onerror = done;
            await audio.play();
        } catch (err) {
            label.textContent = 'Reference unavailable right now';
            setTimeout(() => {
                btn.disabled = false;
                label.textContent = original;
            }, 2500);
        }
    }

    // ---------------------------------------------------------------- charts

    function renderCharts() {
        const data = state.result;
        if (!data || typeof Chart === 'undefined') {
            return;
        }
        state.chartsDirty = false;

        const diff = data.differences;
        const expected = diff.expected_vector || [];
        const heard = diff.transcribed_vector || [];
        const n = Math.max(expected.length, heard.length);
        lineChart('phoneme-chart', Array.from({ length: n }, (_, i) => i + 1), [
            { label: 'Reference', data: expected, borderColor: COLORS.reference },
            { label: 'You', data: heard, borderColor: COLORS.you },
        ], { legend: true, yTicks: false, tooltipTitle: (items) => `Position ${items[0].label}` });

        const prosody = data.prosody || {};
        const f0 = prosody.f0 || [];
        const energy = prosody.energy || [];
        lineChart('f0-chart', f0.map((_, i) => i), [
            { label: 'Pitch (Hz)', data: f0, borderColor: COLORS.you },
        ], { legend: false, yTicks: true });
        lineChart('energy-chart', energy.map((_, i) => i), [
            { label: 'Energy', data: energy, borderColor: COLORS.you },
        ], { legend: false, yTicks: true });
    }

    function lineChart(canvasId, labels, datasets, { legend, yTicks, tooltipTitle }) {
        if (state.charts[canvasId]) {
            state.charts[canvasId].destroy();
        }
        const ctx = $(canvasId).getContext('2d');
        state.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: datasets.map(d => ({
                    ...d,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHitRadius: 8,
                    tension: 0.3,
                    fill: false,
                })),
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { display: false },
                    y: {
                        display: true,
                        grid: { color: '#f0efed' },
                        border: { display: false },
                        ticks: { display: yTicks, color: '#8a8782', font: { size: 11 }, maxTicksLimit: 4 },
                    },
                },
                plugins: {
                    legend: {
                        display: legend,
                        position: 'bottom',
                        labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, color: '#57534e', font: { size: 11 } },
                    },
                    tooltip: {
                        callbacks: tooltipTitle ? { title: tooltipTitle } : {},
                    },
                },
            },
        });
    }
})();
