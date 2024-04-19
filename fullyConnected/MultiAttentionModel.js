class MultiAttentionModel {
    constructor(inputSize, outputSize, numHeads) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.numHeads = numHeads;

        // Initialize the attention weights and biases
        this.attentionWeights = [];
        this.attentionBiases = [];
        for (let i = 0; i < numHeads; i++) {
            this.attentionWeights[i] = Matrix.randomize(inputSize, outputSize);
            this.attentionBiases[i] = Matrix.randomize(1, outputSize);
        }

        // Initialize the normalization layer
        this.normalizationLayer = new Matrix(1, outputSize);
    }

    forward(input) {
        // Apply attention mechanism
        this.attentionOutputs = [];
        this.attentionInput = [];

        var attentionOutput;
        for (let i = 0; i < this.numHeads; i++) {
            this.attentionInput.push(input);
            attentionOutput = Matrix.dot(input, this.attentionWeights[i]);
            attentionOutput.add(this.attentionBiases[i]);
            attentionOutput = attentionOutput.relu();
            this.attentionOutputs.push(attentionOutput);
            input = attentionOutput;
        }

        // Apply normalization layer
        const normalizedOutput = Matrix.multiplyElementWise(attentionOutput, this.normalizationLayer);

        return normalizedOutput;
    }

    backward(input, target, output, loss, learningRate) {

        // Calculate the gradients for the backward weights and biases
        var gradientsBackwardOutput = Matrix.subtract(target, output);
        gradientsBackwardOutput.normalize();

        // Calculate the gradients for the normalization layer
        const gradientsNormalizationLayer = Matrix.multiply(this.attentionOutputs[this.numHeads - 1], gradientsBackwardOutput);
        this.normalizationLayer.add(gradientsNormalizationLayer.multiplyScaler(learningRate));


        gradientsBackwardOutput = Matrix.multiply(gradientsBackwardOutput, this.normalizationLayer);
        // Calculate the gradients for the attention weights and biases
        for (let i = this.numHeads - 1; i >= 0; i--) {

            // Calculate the gradients for the attention weights
            const activationDerivative = this.attentionOutputs[i].reluDerivative();
            gradientsBackwardOutput.multiplyElementWise(activationDerivative)
            gradientsBackwardOutput.normalize();


            // Calculate the gradients for the attention weights
            const gradientsAttentionWeightsLayer = Matrix.dot(this.attentionInput[i].transpose(), gradientsBackwardOutput);

            // Calculate the gradients for the attention biases
            const gradientsAttentionBiasesLayer = gradientsBackwardOutput;

            // Calculate the gradients for the backward output
            gradientsBackwardOutput = Matrix.dot(gradientsBackwardOutput, this.attentionWeights[i].transpose());

            this.attentionWeights[i].add(gradientsAttentionWeightsLayer.multiplyScaler(learningRate));
            this.attentionBiases[i].add(gradientsAttentionBiasesLayer.multiplyScaler(learningRate));
           
        }
    }

    train(input, target, learningRate, numIterations) {
        for (let i = 0; i < numIterations; i++) {
            // Forward pass
            const output = this.forward(input);

            // Calculate the loss
            const loss = this.calculateLoss(output, target);

            // Backward pass
            this.backward(input, target, output, loss, learningRate);

            // Print the loss for monitoring
            console.log(`Iteration ${i + 1}: Loss = ${loss}`);
        }
    }

    calculateLoss(output, target) {
        // Calculate the mean squared error loss
        const error = Matrix.subtract(output, target);
        const squaredError = Matrix.multiplyElementWise(error, error);
        const sumSquaredError = squaredError.data.flat().reduce((acc, val) => acc + val, 0);
        const meanSquaredError = sumSquaredError / (output.rows * output.cols);

        return meanSquaredError;
    }
}

class Word2Vec {
    constructor(vocabularySize, embeddingSize) {
        this.vocabularySize = vocabularySize;
        this.embeddingSize = embeddingSize;

        // Initialize the word embeddings
        this.wordEmbeddings = [];
        for (let i = 0; i < vocabularySize; i++) {
            this.wordEmbeddings[i] = Matrix.randomize(1, embeddingSize);
        }
    }

    train(sentences, windowSize, learningRate, numIterations) {
        for (let iteration = 0; iteration < numIterations; iteration++) {
            for (let sentence of sentences) {
                for (let i = 0; i < sentence.length; i++) {
                    const centerWord = sentence[i];

                    // Get the context words within the window size
                    const contextWords = sentence.slice(Math.max(0, i - windowSize), i).concat(sentence.slice(i + 1, i + windowSize + 1));

                    // Update the word embeddings for the center word and context words
                    for (let contextWord of contextWords) {
                        const centerEmbedding = this.wordEmbeddings[centerWord];
                        const contextEmbedding = this.wordEmbeddings[contextWord];

                        // Calculate the dot product of the center and context embeddings
                        const dotProduct = Matrix.dot(centerEmbedding, contextEmbedding.transpose());

                        console.log(contextEmbedding, dotProduct, centerEmbedding, contextEmbedding.transpose())
                        // Calculate the gradients for the center and context embeddings
                        const gradientsCenter = contextEmbedding.multiplyElementWise(dotProduct);
                        const gradientsContext = centerEmbedding.multiplyElementWise(dotProduct.transpose());

                        // Update the center and context embeddings
                        this.wordEmbeddings[centerWord].add(gradientsCenter.multiplyScaler(learningRate));
                        this.wordEmbeddings[contextWord].add(gradientsContext.multiplyScaler(learningRate));
                    }
                }
            }
        }
    }

    getWordEmbedding(word) {
        return this.wordEmbeddings[word];
    }
}


// Define the vocabulary size and embedding size
const vocabularySize = 5;
const embeddingSize = 5;

// Create an instance of the Word2Vec model
const word2vec = new Word2Vec(vocabularySize, embeddingSize);

// Define the sentences and other parameters
const sentences = [
    [0, 1, 2, 3, 4], // Sentence 1
    [5, 6, 7, 8, 9], // Sentence 2
    [10, 11, 12, 13, 14] // Sentence 3
];
const windowSize = 2;
const learningRate = 0.01;
const numIterations = 100;

// Train the Word2Vec model
word2vec.train(sentences, windowSize, learningRate, numIterations);

// Get the word embedding for a specific word
const word = 2;
const embedding = word2vec.getWordEmbedding(word);
console.log(`Word embedding for word ${word}:`, embedding);