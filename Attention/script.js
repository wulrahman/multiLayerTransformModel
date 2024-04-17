
const dModel = 6;
const numHeads = 6;

const attention = new MultiheadAttention(dModel, numHeads);

const inputStrings = [
    // Customer Service Examples
    "Hello, how can I assist you today?",
    "Thank you for contacting customer service.",
    "How may I help you?",
    "I'm here to answer any questions you have.",
    "Please let me know how I can assist you.",
    "Welcome to our customer support.",
    "Hello, how can I assist you?",
    "How can I provide excellent service to you?",
    "I'm here to help you with any issues you may have.",
    "What can I do for you today?",
    "How can I make your experience better?",
    "Thank you for reaching out to us.",
    "How can I assist you with your inquiry?",
    "I'm here to ensure your satisfaction.",
    "How can I assist you with your request?",
    "I'm here to provide you with the best service possible.",
    "How can I assist you today?",
    "Thank you for choosing our customer service.",
    "How can I assist you with your problem?",
    "I'm here to resolve any issues you may have.",
];

const targets = [
        // Customer Dialog Examples
        "How can I assist you with your question?",
        "I'm here to listen to your concerns.",
        "How can I assist you with your feedback?",
        "I'm here to address any complaints you may have.",
        "How can I assist you with your suggestion?",
        "I'm here to implement any improvements you recommend.",
        "How can I assist you with your comment?",
        "I'm here to provide you with a prompt response.",
        "How can I assist you with your inquiry?",
        "I'm here to ensure your satisfaction.",
        "How can I assist you with your request?",
        "I'm here to provide you with the best service possible.",
        "How can I assist you today?",
        "Thank you for choosing our customer service.",
        "How can I assist you with your problem?",
        "I'm here to resolve any issues you may have.",
        "How can I assist you with your question?",
        "I'm here to listen to your concerns.",
        "How can I assist you with your feedback?",
        "I'm here to address any complaints you may have."
];

const indexedDictionary = {"" : 0};
function tokenizeMessage(message) {
    const tokens = message.toLowerCase().split(" ");
    return tokens;
}

let index = 1;
inputStrings.forEach(inputString => {
    tokenizeMessage(inputString).forEach(token => {
        const lowercaseKey = token.toLowerCase();
        if (!indexedDictionary.hasOwnProperty(lowercaseKey)) {
            indexedDictionary[lowercaseKey] = index;
            index++;
        }
    });
});

targets.forEach(targetString => {
    tokenizeMessage(targetString).forEach(token => {
        const lowercaseKey = token.toLowerCase();
        if (!indexedDictionary.hasOwnProperty(lowercaseKey)) {
            indexedDictionary[lowercaseKey] = index;
            index+=2;
        }
    });
});

// Normalize the values in indexedDictionary
const values = Object.values(indexedDictionary);
const min = Math.min(...values);
const max = Math.max(...values);

for (const key in indexedDictionary) {
    const normalizedValue = (indexedDictionary[key] - min) / (max - min);
    indexedDictionary[key] = normalizedValue;
}

const inputs = inputStrings.map(inputString => {
    const query = Matrix.fromArray(inputString.split(' ').map(char => indexedDictionary[char.toLowerCase()]), dModel, numHeads);
    return query;
});

const outputTargets = targets.map(targetString => {
    const target = Matrix.fromArray(targetString.split('').map(char => indexedDictionary[char.toLowerCase()]), dModel, numHeads);
    return target;
});


var loss;
for (let i = 0; i < 10000; i++) {
    loss = 0;
    for (let j = 0; j < inputs.length; j++) {
        const output = attention.forward(inputs[j], inputs[j], inputs[j]);
        const target = outputTargets[j];

        loss += output.meanSquaredError(target);
        attention.backward(inputs[j], target, output, loss, 1e-5);
    }
    console.log(loss);
}

function generateResponse(input) {

    const target = Matrix.fromArray(input.split('').map(char => indexedDictionary[char.toLowerCase()]), dModel, numHeads);

    const output = attention.forward(target, target, target);
    console.log(output);
    const outputArray = output.data;
    const outputTokens = [];
    for (let i = 0; i < outputArray.length; i++) {
        outputTokens.push((outputArray[i].map(token => {
            token = (token * (max - min)) + min;
            for (const key in indexedDictionary) {
                if ( (indexedDictionary[key] * (max - min)) + min === Math.ceil(token)) {
                    return key;
                }
            }
        }).join(" "))
    );
    }
    
    return outputTokens.join(" ");
}


document.addEventListener('DOMContentLoaded', function() {
    const inputElement = document.getElementById('inputElement');

    inputElement.addEventListener('input', function(event) {
        const userInput = event.target.value;
        const response = generateResponse(userInput);
        console.log(response);
    });
});

