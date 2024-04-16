
// const stockData = [100, 105, 110, 115, 120, 115, 110, 105, 100];

// // Normalize the data
// const normalizedData = stockData.map(value => (value - Math.min(...stockData)) / (Math.max(...stockData) - Math.min(...stockData)));

// // Create training data
// const input = new Matrix(1, normalizedData.length - 2);
// for (let i = 0; i < normalizedData.length - 2; i++) {
//     input.set(0, i, normalizedData[i]);
// }

// const target = new Matrix(1, normalizedData.length -2);
// for (let i = 0; i < normalizedData.length -2; i++) {
//     target.set(0, i, normalizedData[i + 2]);
// }


// console.log(input, target);

// // Create a multi-attention model
// const model = new MultiAttentionModel(normalizedData.length - 2, normalizedData.length - 2, 2);

// // Train the model
// const learningRate = 0.01;
// const numIterations = 10000;
// model.train(input, target, learningRate, numIterations);

// // Test the model
// const output = model.forward(input);
// console.log(output, (Math.max(...stockData) - Math.min(...stockData)), Math.min(...stockData));


// Create training data
const inputs = [
    [0.3, 0.6, 0.9],
    [0.1, 0.4, 0.7],
    [0.2, 0.5, 0.8]
];

const targets = [
    [0.5, 0.8, 0.11],
    [0.3, 0.6, 0.9],
    [0.4, 0.7, 0.1]
];

// Create a multi-attention model
const model = new MultiAttentionModel(3, 3, 2);

// Train the model
const learningRate = 0.001;
const numIterations = 100;

for (let i = 0; i < numIterations; i++) {
    for (let j = 0; j < inputs.length; j++) {
        const input = new Matrix(1, 3);
        input.set(0, 0, inputs[j][0]);
        input.set(0, 1, inputs[j][1]);
        input.set(0, 2, inputs[j][2]);

        const target = new Matrix(1, 3);
        target.set(0, 0, targets[j][0]);
        target.set(0, 1, targets[j][1]);
        target.set(0, 2, targets[j][2]);

        model.train(input, target, learningRate, 1);
    }
}


// Test the model
for (let j = 0; j < inputs.length; j++) {
    const input = new Matrix(1, 3);
    input.set(0, 0, inputs[j][0]);
    input.set(0, 1, inputs[j][1]);
    input.set(0, 2, inputs[j][2]);

    const output = model.forward(input);
    console.log(output);
}
