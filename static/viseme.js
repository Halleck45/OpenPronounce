/**
 * Animate a mouth image from a list of IPA phones.
 *
 * Images come from the HunanBean CMU39 set (see assets/mouths/HunanBeanCMU39/README.md).
 */
class Viseme {
    constructor(mouthImage, phonemesToVisemes, imagesFolder = "/static/assets/mouths/HunanBeanCMU39") {
        this.mouthImage = mouthImage;
        this.imagesFolder = imagesFolder.replace(/\/$/, "");
        this.token = 0;

        this.phonemesToVisemes = phonemesToVisemes || {
            "b": "B.png", "p": "P.png", "m": "M.png",
            "tʃ": "CH.png", "dʒ": "JH.png",
            "d": "D.png", "t": "T.png", "ɾ": "D.png",
            "ð": "DH.png", "θ": "TH.png",
            "f": "F.png", "v": "V.png",
            "g": "G.png", "ɡ": "G.png", "k": "K.png",
            "h": "H.png", "ɦ": "HH.png",
            "j": "Y.png",
            "l": "L.png", "ɹ": "R.png", "r": "R.png", "ʁ": "R.png",
            "n": "N.png", "ŋ": "NG.png", "ɲ": "N.png",
            "s": "S.png", "z": "Z.png",
            "ʃ": "SH.png", "ʒ": "ZH.png",
            "w": "W.png",
            "ə": "AH.png", "ʌ": "AH.png", "ɐ": "AH.png",
            "a": "AE.png", "æ": "AE.png",
            "ɑ": "AA.png", "ɒ": "AA.png",
            "o": "AO.png", "ɔ": "AO.png",
            "i": "IY.png", "ɪ": "IH.png", "y": "IY.png",
            "e": "EH.png", "ɛ": "EH.png", "ø": "EH.png", "œ": "EH.png",
            "u": "UW.png", "ʊ": "UH.png",
            "ɜ": "ER.png", "ɝ": "ER.png", "ɚ": "ER.png",
            "eɪ": "EY.png", "aɪ": "AY.png", "ɔɪ": "OY.png",
            "oʊ": "OW.png", "əʊ": "OW.png", "aʊ": "AW.png",
        };

        // Compound phones split into the parts that have a mouth shape
        this.diphthongMap = {
            "eə": ["e", "ə"], "ɪə": ["ɪ", "ə"], "ʊə": ["ʊ", "ə"],
            "juː": ["j", "u"], "ju": ["j", "u"], "ɑːɹ": ["ɑ", "ɹ"], "ɑɹ": ["ɑ", "ɹ"],
        };

        this.rest();
    }

    rest() {
        this.mouthImage.src = `${this.imagesFolder}/rest.png`;
    }

    /**
     * Split a phone group (one word in IPA, or one compound phone) into displayable phones.
     * @param {string} phonemeGroup
     * @returns {string[]}
     */
    splitPhonemes(phonemeGroup) {
        const group = String(phonemeGroup).replace(/[ːˈˌ.]/g, "");
        if (this.phonemesToVisemes[group]) {
            return [group];
        }
        if (this.diphthongMap[group]) {
            return this.diphthongMap[group];
        }

        // Greedy longest-match over the known keys (2 chars first, then 1)
        const result = [];
        let i = 0;
        while (i < group.length) {
            const two = group.slice(i, i + 2);
            if (this.diphthongMap[two]) {
                result.push(...this.diphthongMap[two]);
                i += 2;
            } else if (this.phonemesToVisemes[two]) {
                result.push(two);
                i += 2;
            } else {
                result.push(group[i]);
                i += 1;
            }
        }
        return result.length ? result : [group];
    }

    /**
     * Animate the mouth through phone groups.
     * @param {string[]} phonemeGroups - words in IPA, or single phones
     * @param {number} duration - total duration per group in milliseconds
     */
    async play(phonemeGroups, duration = 300) {
        const token = ++this.token;
        for (const phonemeGroup of phonemeGroups) {
            const phonemes = this.splitPhonemes(phonemeGroup);
            const step = Math.max(150, duration / phonemes.length);
            for (const phoneme of phonemes) {
                if (token !== this.token) {
                    return;
                }
                const viseme = this.phonemesToVisemes[phoneme] || "rest.png";
                this.mouthImage.src = `${this.imagesFolder}/${viseme}`;
                await new Promise(resolve => setTimeout(() => requestAnimationFrame(resolve), step));
            }
        }
        if (token === this.token) {
            this.rest();
        }
    }

    /**
     * Rough duration in milliseconds of a list of phones, used to pace the animation.
     */
    estimateWordDuration(phonemes) {
        const phonemeDurations = {
            "b": 100, "p": 100, "m": 120, "tʃ": 150, "dʒ": 150, "d": 100, "t": 80,
            "ð": 120, "θ": 120, "f": 100, "v": 100, "g": 100, "ɡ": 100, "k": 100, "h": 80,
            "j": 120, "l": 130, "ɹ": 130, "n": 120, "ŋ": 130, "s": 90, "z": 90,
            "ʃ": 150, "ʒ": 150, "w": 120, "ə": 100, "a": 150, "o": 150, "i": 140,
            "e": 140, "ɪ": 130, "u": 160, "ʊ": 130,
            "juː": 300, "ɑː": 250, "eɪ": 250, "oʊ": 250, "aɪ": 280, "aʊ": 300,
            "ɔɪ": 280, "ɪə": 250, "ʊə": 250, "eə": 250,
        };
        let total = 0;
        phonemes.forEach(phoneme => {
            const parts = this.diphthongMap[phoneme] || [phoneme];
            parts.forEach(part => { total += phonemeDurations[part] || 200; });
        });
        return total;
    }
}
