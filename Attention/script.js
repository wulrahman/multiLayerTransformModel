
document.addEventListener('DOMContentLoaded', function() {

    const dModel = 6;
    const numHeads = 6;
    
    const huffmanModel = new BuildHuffmanGramModel();
    const attention = new MultiheadAttention(dModel, numHeads);
    const indexedDictionaryModel = new IndexDictionary();

    const inputStrings = [
        "What is my name?",
        "What is your name?",
        // "What is your favorite color?",
        // "What is your favorite food?",
        // "Please enter your question.",
        // "Can you help me with my homework?",
        // "What is the capital of France?",
        // "How old are you?",
        // "What is the meaning of life?",
        // "What is the weather like today?"
    ];

    const targets = [
        "Your name is Waheed Ul Rahman.",
        "My name is Ricktorious Chat Bot.",
        // "My favorite color is blue.",
        // "My favorite food is pizza.",
        // "Please enter your response.",
        // "I can help you with your homework.",
        // "The capital of France is Paris.",
        // "I am 20 years old.",
        // "The meaning of life is to be happy.",
        // "The weather is sunny today."
    ];

    inputStrings.forEach(inputString => {
        indexedDictionaryModel.addWord(inputString);
    });

    targets.forEach(outputTarget => {
        indexedDictionaryModel.addWord(outputTarget);
    });

    indexedDictionaryModel.normalizeMinMax();

    const targets_array = [], inputs_array = [];

    inputStrings.forEach(inputString => {
        const matrix = indexedDictionaryModel.stringToMatrix(inputString, dModel, dModel);
        // matrix.add(stringToMatrix(inputString, dModel, dModel))
        inputs_array.push(matrix);
    });
    
    targets.forEach(outputTarget => {
        targets_array.push(indexedDictionaryModel.stringToMatrix(outputTarget, dModel, dModel));
    });

    inputStrings.forEach(inputString => {
        huffmanModel.buildFrequencyMap(inputString.toLowerCase().replace(/[^\w\s]/gi, ''));
    });

    targets.forEach(targetString => {
        huffmanModel.buildFrequencyMap(targetString.toLowerCase().replace(/[^\w\s]/gi, ''));
    });

    huffmanModel.buildHuffmanTree();
    huffmanModel.buildHuffmanCodeMap();
    
    const inputs = inputStrings.map(inputString => {
        //return Matrix.arraytoMatrix(huffmanModel.encode(targetString.toLowerCase().replace(/[^\w\s]/gi, '')).split(''), dModel, dModel, 0)
        return huffmanModel.textToMatrix(inputString, dModel, dModel);

   });


   const outputTargets = targets.map(targetString => {
       //return Matrix.arraytoMatrix(huffmanModel.encode(targetString.toLowerCase().replace(/[^\w\s]/gi, '')).split(''), dModel, dModel, 0)
       return huffmanModel.textToMatrix(targetString, dModel, dModel);
   });


    console.log(inputs, outputTargets, inputs_array, targets_array);

    var loss;
    const losses = [];

    var learningRate = 1e-1;
    for (let i = 0; i < 10000; i++) {

        loss = 0;
        for (let j = 0; j < inputs.length; j++) {
            //const target = outputTargets[j];
            const input = inputs_array[j];
            const target = targets_array[j];

            const output = attention.forward(inputs[j], input, input);
            loss += output.meanSquaredError(target);
            // Plot the loss in a graph
            attention.backward(inputs[j], target, output, loss, learningRate);
        }
        if (losses.length > 0 && loss > losses[losses.length - 1]) {
            learningRate *= 0.995;
        }

        losses.push(loss);
        // loss /= inputs.length;
        console.log(`Epoch ${i} Loss: ${loss}`);
    }

    const canvas = document.getElementById('lossGraph');
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const scaledLosses = losses.map(loss => (loss - minLoss) / (maxLoss - minLoss) * canvas.clientHeight);

    scaledLosses.forEach((loss, i) => {
        plotLoss(i, loss);
    });

    function plotLoss(i, loss) {
        const context = canvas.getContext('2d');
        // Set the color for the point
        context.fillStyle = 'red'; // You can change the color as needed
        // Draw a point at the specified coordinates
        context.fillRect((i / losses.length) * canvas.clientWidth, loss, 1, 1); // Adjust the size of the point as needed
    }
    
    function generateResponse(input) {

        const targetArray = input.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        const query = huffmanModel.textToMatrix(input, dModel, dModel);
        const matrix = indexedDictionaryModel.stringToMatrix(input, dModel, dModel);
        // matrix.add(stringToMatrix(input, dModel, dModel))
        const output = attention.forward(query, matrix, matrix);

        const values = [];
        for (let i = 0; i < output.data[0].length; i++) {
            values.push(output.data[i].map(x => indexedDictionaryModel.findClosestMatch(x)).join(' '));
        }

        return values.join('');
    }


    const inputElement = document.getElementById('inputElement');

    inputElement.addEventListener('input', function(event) {
        const userInput = event.target.value;
        const response = generateResponse(userInput);
        console.log(response);
    });
    
    console.log(targets, inputs_array, targets_array, inputs, outputTargets)

});