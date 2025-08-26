class Viseme {
    constructor(mouthImage, phonemesToVisemes, imagesFolder = "/static/assets/mouths/HunanBeanCMU39/") {
        this.mouthImage = mouthImage;
        
        this.phonemesToVisemes = phonemesToVisemes || {
            "b": "B.png", "p": "P.png", "m": "M.png",
            "tʃ": "CH.png", "dʒ": "JH.png",
            "d": "D.png", "t": "T.png",
            "ð": "DH.png", "θ": "TH.png",
            "f": "F.png", "v": "V.png",
            "g": "G.png", "k": "K.png",
            "h": "H.png",
            "j": "Y.png",
            "l": "L.png", "ɹ": "R.png",
            "n": "N.png", "ŋ": "NG.png",
            "s": "S.png", "z": "Z.png",
            "ʃ": "SH.png", "ʒ": "ZH.png",
            "w": "W.png",
            "ə": "AH.png", "a": "AE.png", "o": "AO.png",
            "i": "IY.png", "e": "EH.png", "ɪ": "IH.png",
            "u": "UW.png", "ʊ": "UH.png"
        };

        this.diphthongMap = {
            "eɪ": ["e", "ɪ"],   // say
            "ɔɪ": ["ɔ", "ɪ"],   // boy
            "aɪ": ["a", "ɪ"],   // time
            "eə": ["e", "ə"],   // air
            "ɪə": ["ɪ", "ə"],   // ear
            "ʊə": ["ʊ", "ə"],   // tour
            "əʊ": ["ə", "ʊ"],   // go
            "aʊ": ["a", "ʊ"],   // house
            "juː": ["j", "u"],  // you
            "ɑːɹ": ["ɑ", "ɹ"],  // are
            "həloʊ": ["h", "ə", "l", "oʊ"], // hello
        };

        this.imagesFolder = imagesFolder;
        this.mouthImage.src = `${this.imagesFolder}/rest.png`;
    }

    /**
     * Fonction pour diviser les phonèmes complexes en phonèmes simples.
     */
    splitPhonemes(phonemeGroup) {
        // Vérifier si le phonème complet est une diphtongue
        if (this.diphthongMap[phonemeGroup]) {
            console.log(`Décomposition de ${phonemeGroup} via diphthongMap:`, this.diphthongMap[phonemeGroup]);
            return this.diphthongMap[phonemeGroup];
        }
    
        // Vérifier si le phonème contient une diphtongue connue et la diviser manuellement
        for (const diphthong in this.diphthongMap) {
            if (phonemeGroup.includes(diphthong)) {
                let splitResult = phonemeGroup.replace(diphthong, this.diphthongMap[diphthong].join(" ")).split(" ");
                console.log(`Décomposition manuelle de ${phonemeGroup}:`, splitResult);
                return splitResult;
            }
        }
    
        // Nouvelle regex pour extraire chaque phonème simple
        const phonemeRegex = /[a-zʃʒɹðθŋɑɪɔəʊʊː]+/g;
        const matchedPhonemes = phonemeGroup.match(phonemeRegex);
        
        if (matchedPhonemes) {
            let splitPhonemes = [];
            for (let phoneme of matchedPhonemes) {
                if (this.diphthongMap[phoneme]) {
                    splitPhonemes.push(...this.diphthongMap[phoneme]); // Si c'est une diphtongue connue
                } else {
                    splitPhonemes.push(phoneme);
                }
            }
            console.log(`Décomposition améliorée de ${phonemeGroup}:`, splitPhonemes);
            return splitPhonemes;
        }
    
        console.log(`Aucune décomposition trouvée pour ${phonemeGroup}, retour inchangé`);
        return [phonemeGroup]; // Si aucun découpage possible, on retourne tel quel
    }
    

    /**
     * Fonction d'animation des visèmes
     */
    async play(phonemeGroups, duration = 300) {
        const mouthImage = this.mouthImage;
        const imagesFolder = this.imagesFolder;
        const phonemesToVisemes = this.phonemesToVisemes;
        
        for (let phonemeGroup of phonemeGroups) {
            const phonemes = this.splitPhonemes(phonemeGroup);
    
            for (let phoneme of phonemes) {
                console.log("Affichage du visème pour le phonème:", phoneme);
                const viseme = phonemesToVisemes[phoneme] || "rest.png";
                console.log(viseme)
                mouthImage.src = `${imagesFolder}/${viseme}`;
    
                // Attendre avant de passer au phonème suivant
                await new Promise(resolve => setTimeout(() => requestAnimationFrame(resolve), Math.max(150, duration / phonemes.length)));
            }
        }
    
        // Remettre l’image de repos à la fin
        mouthImage.src = `${imagesFolder}/rest.png`;
    }
    

    estimateWordDuration(phonemes) {
        const phonemeDurations = {
            "b": 100, "p": 100, "m": 120,
            "tʃ": 150, "dʒ": 150,
            "d": 100, "t": 80,
            "ð": 120, "θ": 120,
            "f": 100, "v": 100,
            "g": 100, "k": 100,
            "h": 80,
            "j": 120,
            "l": 130, "ɹ": 130,
            "n": 120, "ŋ": 130,
            "s": 90, "z": 90,
            "ʃ": 150, "ʒ": 150,
            "w": 120,
            "ə": 100, "a": 150, "o": 150,
            "i": 140, "e": 140, "ɪ": 130,
            "u": 160, "ʊ": 130,
            // Diphtongues et voyelles longues
            "juː": 300, "ɑː": 250, "eɪ": 250,
            "oʊ": 250, "aɪ": 280, "aʊ": 300,
            "ɔɪ": 280, "ɪə": 250, "ʊə": 250,
            "eə": 250
        };
    
        let totalDuration = 0;
    
        phonemes.forEach(phoneme => {
            // Vérifier si le phonème est une diphtongue et le décomposer
            if (this.diphthongMap[phoneme]) {
                this.diphthongMap[phoneme].forEach(subPhoneme => {
                    totalDuration += phonemeDurations[subPhoneme] || 200; // Valeur par défaut
                });
            } else {
                totalDuration += phonemeDurations[phoneme] || 200; // Valeur par défaut
            }
        });
    
        return totalDuration;
    }
    
        
}
