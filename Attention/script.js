
document.addEventListener('DOMContentLoaded', function() {


    
    const inputStrings = [
        "What is my name?",
        'what is his name?',
        'what is your name?',
    ];

    const targetStrings = [
        "Your name is Waheed Ul Rahman.",
        "His name is David Ul Rahman.",
        'My name is Waheed Ul Rahman.'
    ];

    const uniqueWords = new Set();

    inputStrings.forEach(inputString => {
        const words = inputString.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        words.forEach(word => uniqueWords.add(word));
    });

    targetStrings.forEach(targetString => {
        const words = targetString.toLowerCase().replace(/[^\w\s]/gi, '').split(' ');
        words.forEach(word => uniqueWords.add(word));
    });

    const dModel = uniqueWords.size;
    const numHeads = uniqueWords.size;
    
    const huffmanModel = new BuildHuffmanGramModel();
    const attention = new MultiheadAttention(dModel, numHeads);
    const indexedDictionaryModel = new IndexDictionary();
    const wordEmbeddings = new WordEmbeddings(2, dModel, 1e-5, 10000);

    inputStrings.forEach(inputString => {
        indexedDictionaryModel.addWord(inputString);
    });

    targetStrings.forEach(outputTarget => {
        indexedDictionaryModel.addWord(outputTarget);
    });

    indexedDictionaryModel.normalizeMinMax();

    inputStrings.forEach(inputString => {
        inputString = inputString.toLowerCase().replace(/[^\w\s]/gi, '')
        huffmanModel.buildFrequencyMap(inputString);
        wordEmbeddings.trainWordEmbeddings(inputString);
    });
    
    targetStrings.forEach(outputTarget => {
        outputTarget = outputTarget.toLowerCase().replace(/[^\w\s]/gi, '');
        huffmanModel.buildFrequencyMap(outputTarget);
        wordEmbeddings.trainWordEmbeddings(outputTarget);
    });

    huffmanModel.buildHuffmanTree();
    huffmanModel.buildHuffmanCodeMap();
    
    
    var loss;
    const losses = [];

    var learningRate = 1e-1;
    for (let i = 0; i < 10000; i++) {

        loss = 0;
        for (let j = 0; j < inputStrings.length; j++) {
            const inputString = inputStrings[j].toLowerCase().replace(/[^\w\s]/gi, '');
            const query =  wordEmbeddings.textToMatrix(inputString, dModel, dModel);
            const key = huffmanModel.textToMatrix(inputString, dModel, dModel, 0)
            const value = indexedDictionaryModel.stringToMatrix(inputString, dModel, dModel);

            const targetString = targetStrings[j].toLowerCase().replace(/[^\w\s]/gi, '');
            const target = indexedDictionaryModel.stringToMatrix(targetString, dModel, dModel); 
            // const target  = huffmanModel.textToMatrix(targetString, dModel, dModel, 0)
            // const target  = wordEmbeddings.textToMatrix(targetString, dModel, dModel, 0)


            // console.log(target, value, query, key);
            const output = attention.forward(query, value, value);
            loss += output.meanSquaredError( target);
            // Plot the loss in a graph
            attention.backward(value, target, output, loss, learningRate);
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

        const target = input.toLowerCase().replace(/[^\w\s]/gi, '');
        const targetArray = target.split(' ');
        const query = wordEmbeddings.textToMatrix(target, dModel, dModel);
        const value = indexedDictionaryModel.stringToMatrix(target, dModel, dModel);
        const key = huffmanModel.textToMatrix(target, dModel, dModel, 0)

        // matrix.add(stringToMatrix(input, dModel, dModel))
        const output = attention.forward(query, value, value);    
        console.log(output);
        return indexedDictionaryModel.matrix2String(output);
    }


    const inputElement = document.getElementById('inputElement');

    inputElement.addEventListener('input', function(event) {
        const userInput = event.target.value;
        const response = generateResponse(userInput);
        console.log(response);
    });
});