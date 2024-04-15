class LanguageModel {
    constructor(n) {
        this.vocabulary = [];
        this.nGrams = {};
        this.n = n;
    }

    train(sentences) {
        for (const sentence of sentences) {
            const words = sentence.split(" ");
            const n = words.length;
            for (let i = 0; i < n - this.n + 1; i++) {
                const nGram = words.slice(i, i + this.n).join(" ");
                if (!this.nGrams[nGram]) {
                    this.nGrams[nGram] = 0;
                }
                this.nGrams[nGram]++;
            }
            for (const word of words) {
                if (!this.vocabulary.includes(word)) {
                    this.vocabulary.push(word);
                }
            }
        }
    }

    generate(startWord, numWords) {
        let generatedSentence = startWord;
        let currentWord = startWord;
        for (let i = 1; i < numWords; i++) {
            const nextWord = this.getNextWord(currentWord);
            if (nextWord === null) {
                break;
            }
            generatedSentence += " " + nextWord;
            currentWord = nextWord;
        }
        return generatedSentence;
    }

    getNextWord(currentWord) {
        const possibleNextWords = [];
        for (const nGram in this.nGrams) {
            const words = nGram.split(" ");
            const n = words.length;
            for (let i = 0; i < n; i++) {
                if (words[i] === currentWord && words[i + 1]) {
                    possibleNextWords.push(words[i + 1]);
                }
            }
        }
        if (possibleNextWords.length === 0) {
            return null;
        }
        const nextWord = possibleNextWords[Math.floor(Math.random() * possibleNextWords.length)];
        return nextWord;
    }
}
