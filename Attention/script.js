// Create training data
const input = new Matrix(1, 3);
input.set(0, 0, 0.3);
input.set(0, 1, 0.6);
input.set(0, 2, 0.9);

const target = new Matrix(1, 3);
target.set(0, 0, 0.5);
target.set(0, 1, 0.8);
target.set(0, 2, 0.11);


// Create a multi-attention model
const model = new MultiAttentionModel(3, 3, 2);

// Train the model
const learningRate = 0.1;
const numIterations = 100;
model.train(input, target, learningRate, numIterations);