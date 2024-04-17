
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


// // Create training data
// const inputs = [
//     [3, 6, 9],
//     [1, 4, 7],
//     [2, 5, 8]
// ];

// const targets = [
//     [5, 8, 11],
//     [3, 6, 9],
//     [4, 7, 1]
// ];

const inputSize = 40;
const outputSize = 40;
// Create a multi-attention model
const model = new MultiAttentionModel(inputSize, outputSize, 1);

// Train the model
const learningRate = 1e-2;
const numIterations = 1000;

// Chatbot implementation using a dictionary
const messages = {
    "hello": "Hi there!",
    "how are you?": "I'm good, thanks for asking.",
    "what's your name?": "My name is Chatbot.",
    "bye": "Goodbye!",
    "what's up?": "Not much, just here to assist you.",
    "how's the weather?": "I'm not sure, I don't have access to real-time data.",
    "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "what's your favorite color?": "I don't have a favorite color, I'm just a program.",
    "what's the meaning of life?": "That's a philosophical question, I'm afraid I can't answer that.",
    "what's your favorite food?": "I don't eat, I'm just a program.",
    "how old are you?": "I don't have an age, I'm just a program.",
    "where are you from?": "I don't have a physical location, I exist in the digital world.",
    "what programming languages do you know?": "I have knowledge of various programming languages including JavaScript, Python, Java, C++, and more.",
    "can you help me with my code?": "Of course! I'm here to assist you with your programming needs.",
    "what's the best IDE for web development?": "There are many great IDEs for web development, some popular ones include Visual Studio Code, Sublime Text, and Atom.",
    "how can I improve my coding skills?": "Practice regularly, work on projects, and continuously learn new concepts and technologies.",
    "what's the difference between var, let, and const in JavaScript?": "var is function-scoped, let and const are block-scoped. var can be redeclared and reassigned, let can be reassigned but not redeclared, and const cannot be redeclared or reassigned.",
    "what's the difference between == and === in JavaScript?": "== performs type coercion, while === performs strict equality comparison without type coercion.",
    "how can I debug my JavaScript code?": "You can use the browser's built-in developer tools or console.log statements to debug your JavaScript code.",
    "what's the difference between null and undefined in JavaScript?": "null represents the intentional absence of any object value, while undefined represents the absence of a value or uninitialized variable.",
    "how can I handle errors in JavaScript?": "You can use try-catch blocks to handle errors and prevent your program from crashing.",
    "what's the difference between synchronous and asynchronous programming?": "Synchronous programming executes tasks sequentially, while asynchronous programming allows tasks to run concurrently and handles the results once they are available.",
    "what's the difference between a function declaration and a function expression in JavaScript?": "A function declaration is hoisted and can be called before it is defined, while a function expression is not hoisted and can only be called after it is defined.",
    "how can I make an HTTP request in JavaScript?": "You can use the built-in Fetch API or XMLHttpRequest object to make HTTP requests in JavaScript.",
    "what's the difference between a map and a set in JavaScript?": "A map is an ordered collection of key-value pairs, while a set is an unordered collection of unique values.",
    "how can I manipulate the DOM in JavaScript?": "You can use the Document Object Model (DOM) API to manipulate HTML elements and their attributes in JavaScript.",
    "what's the difference between a callback function and a promise in JavaScript?": "A callback function is a traditional way of handling asynchronous operations, while a promise provides a more structured and readable way to handle asynchronous code.",
    "how can I handle form submissions in JavaScript?": "You can listen for the 'submit' event on a form element and use event.preventDefault() to prevent the default form submission behavior.",
    "what's the difference between a class and an object in JavaScript?": "A class is a blueprint for creating objects, while an object is an instance of a class.",
    "how can I sort an array in JavaScript?": "You can use the Array.prototype.sort() method to sort an array in JavaScript.",
    "what's the difference between local storage and session storage in JavaScript?": "Local storage persists data even after the browser is closed, while session storage only persists data for the duration of the browser session.",
    "how can I handle user input in JavaScript?": "You can use event listeners to capture user input from various HTML elements such as input fields and buttons.",
    "what's the difference between a for loop and a while loop in JavaScript?": "A for loop is used when you know the number of iterations in advance, while a while loop is used when the number of iterations is not known in advance.",
    "how can I convert a string to a number in JavaScript?": "You can use the parseInt() or parseFloat() functions to convert a string to a number in JavaScript.",
    "what's the difference between a shallow copy and a deep copy in JavaScript?": "A shallow copy creates a new object that references the original object's properties, while a deep copy creates a new object with its own copies of the original object's properties.",
    "how can I check if an element exists in an array in JavaScript?": "You can use the Array.prototype.includes() method or the Array.prototype.indexOf() method to check if an element exists in an array in JavaScript.",
    "what's the difference between a function and a method in JavaScript?": "A function is a standalone block of code that can be called with or without an object, while a method is a function that is associated with an object and can only be called on that object.",
    "how can I convert an object to a JSON string in JavaScript?": "You can use the JSON.stringify() function to convert an object to a JSON string in JavaScript.",
    "what's the difference between a synchronous and an asynchronous function in JavaScript?": "A synchronous function blocks the execution until it completes, while an asynchronous function allows the execution to continue while it performs its task in the background.",
    "how can I remove duplicates from an array in JavaScript?": "You can use the Set object or the Array.prototype.filter() method to remove duplicates from an array in JavaScript.",
    "what's the difference between a named function and an anonymous function in JavaScript?": "A named function has a name that can be used to call it or refer to it, while an anonymous function does not have a name and is typically used as a callback or immediately invoked.",
    "how can I check if a variable is an array in JavaScript?": "You can use the Array.isArray() method to check if a variable is an array in JavaScript.",
    "what's the difference between a closure and a scope in JavaScript?": "A scope is a context in which variables are declared and accessed, while a closure is a function that has access to variables from its outer scope even after the outer function has returned.",
    "how can I convert a number to a string in JavaScript?": "You can use the String() function or the Number.prototype.toString() method to convert a number to a string in JavaScript.",
    "what's the difference between a module and a script in JavaScript?": "A module is a reusable piece of code that exports and imports functionality, while a script is a standalone piece of code that is executed in a specific order.",
    "how can I check if an object has a property in JavaScript?": "You can use the 'in' operator or the Object.prototype.hasOwnProperty() method to check if an object has a property in JavaScript.",
    "what's the difference between a let and a const in JavaScript?": "let allows reassignment of the variable's value, while const does not allow reassignment and requires an initial value to be assigned.",
    "how can I convert a string to uppercase or lowercase in JavaScript?": "You can use the String.prototype.toUpperCase() method to convert a string to uppercase or the String.prototype.toLowerCase() method to convert a string to lowercase in JavaScript.",
    "what's the difference between a promise and an async/await in JavaScript?": "A promise is a way to handle asynchronous operations and chain them together, while async/await provides a more synchronous-like way to write asynchronous code using promises.",
    "how can I check if a string contains a substring in JavaScript?": "You can use the String.prototype.includes() method or the String.prototype.indexOf() method to check if a string contains a substring in JavaScript.",
    "what's the difference between a function and an arrow function in JavaScript?": "An arrow function is a shorthand syntax for writing functions and does not bind its own 'this' value, while a regular function has its own 'this' value.",
    "how can I convert an array to a string in JavaScript?": "You can use the Array.prototype.join() method to convert an array to a string in JavaScript.",
    "what's the difference between a class and a prototype in JavaScript?": "A class is a syntactical sugar for creating objects using constructor functions and prototype inheritance, while a prototype is an object that is used as a blueprint for creating other objects through delegation.",
    "how can I check if a variable is undefined or null in JavaScript?": "You can use the '===' operator to check if a variable is strictly equal to undefined or null in JavaScript.",
    "what's the difference between a callback and a promise in JavaScript?": "A callback is a function that is passed as an argument to another function and is called when a certain event occurs, while a promise is an object that represents the eventual completion or failure of an asynchronous operation.",
    "how can I convert a string to an array in JavaScript?": "You can use the String.prototype.split() method to convert a string to an array in JavaScript.",
    "what's the difference between a function declaration and a function expression in JavaScript?": "A function declaration is hoisted and can be called before it is defined, while a function expression is not hoisted and can only be called after it is defined.",
    "how can I make an HTTP request in JavaScript?": "You can use the built-in Fetch API or XMLHttpRequest object to make HTTP requests in JavaScript.",
    "what's the difference between a map and a set in JavaScript?": "A map is an ordered collection of key-value pairs, while a set is an unordered collection of unique values.",
    "how can I manipulate the DOM in JavaScript?": "You can use the Document Object Model (DOM) API to manipulate HTML elements and their attributes in JavaScript.",
    "what's the difference between a callback function and a promise in JavaScript?": "A callback function is a traditional way of handling asynchronous operations, while a promise provides a more structured and readable way to handle asynchronous code.",
    "how can I handle form submissions in JavaScript?": "You can listen for the 'submit' event on a form element and use event.preventDefault() to prevent the default form submission behavior.",
    "what's the difference between a class and an object in JavaScript?": "A class is a blueprint for creating objects, while an object is an instance of a class.",
    "how can I sort an array in JavaScript?": "You can use the Array.prototype.sort() method to sort an array in JavaScript.",
    "what's the difference between local storage and session storage in JavaScript?": "Local storage persists data even after the browser is closed, while session storage only persists data for the duration of the browser session.",
    "how can I handle user input in JavaScript?": "You can use event listeners to capture user input from various HTML elements such as input fields and buttons.",
    "what's the difference between a for loop and a while loop in JavaScript?": "A for loop is used when you know the number of iterations in advance, while a while loop is used when the number of iterations is not known in advance.",
    "how can I convert a string to a number in JavaScript?": "You can use the parseInt() or parseFloat() functions to convert a string to a number in JavaScript.",
    "what's the difference between a shallow copy and a deep copy in JavaScript?": "A shallow copy creates a new object that references the original object's properties, while a deep copy creates a new object with its own copies of the original object's properties.",
    "how can I check if an element exists in an array in JavaScript?": "You can use the Array.prototype.includes() method or the Array.prototype.indexOf() method to check if an element exists in an array in JavaScript.",
    "what's the difference between a function and a method in JavaScript?": "A function is a standalone block of code that can be called with or without an object, while a method is a function that is associated with an object and can only be called on that object.",
    "how can I convert an object to a JSON string in JavaScript?": "You can use the JSON.stringify() function to convert an object to a JSON string in JavaScript.",
    "what's the difference between a synchronous and an asynchronous function in JavaScript?": "A synchronous function blocks the execution until it completes, while an asynchronous function allows the execution to continue while it performs its task in the background.",
    "how can I remove duplicates from an array in JavaScript?": "You can use the Set object or the Array.prototype.filter() method to remove duplicates from an array in JavaScript.",
    "what's the difference between a named function and an anonymous function in JavaScript?": "A named function has a name that can be used to call it or refer to it, while an anonymous function does not have a name and is typically used as a callback or immediately invoked.",
    "how can I check if a variable is an array in JavaScript?": "You can use the Array.isArray() method to check if a variable is an array in JavaScript.",
    "what's the difference between a closure and a scope in JavaScript?": "A scope is a context in which variables are declared and accessed, while a closure is a function that has access to variables from its outer scope even after the outer function has returned.",
    "how can I convert a number to a string in JavaScript?": "You can use the String() function or the Number.prototype.toString() method to convert a number to a string in JavaScript.",
    "what's the difference between a module and a script in JavaScript?": "A module is a reusable piece of code that exports and imports functionality, while a script is a standalone piece of code that is executed in a specific order.",
    "how can I check if an object has a property in JavaScript?": "You can use the 'in' operator or the Object.prototype.hasOwnProperty() method to check if an object has a property in JavaScript.",
    "what's the difference between a let and a const in JavaScript?": "let allows reassignment of the variable's value, while const does not allow reassignment and requires an initial value to be assigned.",
    "how can I convert a string to uppercase or lowercase in JavaScript?": "You can use the String.prototype.toUpperCase() method to convert a string to uppercase or the String.prototype.toLowerCase() method to convert a string to lowercase in JavaScript.",
    "what's the difference between a promise and an async/await in JavaScript?": "A promise is a way to handle asynchronous operations and chain them together, while async/await provides a more synchronous-like way to write asynchronous code using promises.",
    "how can I check if a string contains a substring in JavaScript?": "You can use the String.prototype.includes() method or the String.prototype.indexOf() method to check if a string contains a substring in JavaScript.",
    "what's the difference between a function and an arrow function in JavaScript?": "An arrow function is a shorthand syntax for writing functions and does not bind its own 'this' value, while a regular function has its own 'this' value.",
    "how can I convert an array to a string in JavaScript?": "You can use the Array.prototype.join() method to convert an array to a string in JavaScript.",
    "what's the difference between a class and a prototype in JavaScript?": "A class is a syntactical sugar for creating objects using constructor functions and prototype inheritance, while a prototype is an object that is used as a blueprint for creating other objects through delegation.",
    "how can I check if a variable is undefined or null in JavaScript?": "You can use the '===' operator to check if a variable is strictly equal to undefined or null in JavaScript.",
    "what's the difference between a callback and a promise in JavaScript?": "A callback is a function that is passed as an argument to another function and is called when a certain event occurs, while a promise is an object that represents the eventual completion or failure of an asynchronous operation.",
    "how can I convert a string to an array in JavaScript?": "You can use the String.prototype.split() method to convert a string to an array in JavaScript."
};

const indexedDictionary = {"" : 0};

// Function to tokenize a message
function tokenizeMessage(message) {
    const tokens = message.toLowerCase().split(" ");
    return tokens;
}

let index = 1;
for (const key in messages) {
    tokenizeMessage(messages[key]).forEach(token => {
        const lowercaseKey = token.toLowerCase();
        if (!indexedDictionary.hasOwnProperty(lowercaseKey)) {
            indexedDictionary[lowercaseKey] = index;
            index++;
        }
    });
    
    tokenizeMessage(key).forEach(token => {
        const lowercaseKey = token.toLowerCase();
        if (!indexedDictionary.hasOwnProperty(lowercaseKey)) {
            indexedDictionary[lowercaseKey] = index;
            index++;
        }
    });
}

function limitArraySize(arr, size, paddingValue) {
    if (arr.length == size) {
        return arr;
    } else {
        return arr.slice(0, size).concat(Array(size - arr.length).fill(paddingValue));
    }
}


// Normalize the values in indexedDictionary
const values = Object.values(indexedDictionary);
const min = Math.min(...values);
const max = Math.max(...values);

for (const key in indexedDictionary) {
    const normalizedValue = (indexedDictionary[key] - min) / (max - min);
    indexedDictionary[key] = normalizedValue;
}

const inputs = [];
const targets = [];

for (const key in messages) {
    const input = [];
    const target = [];

    tokenizeMessage(messages[key]).forEach(token => {
        input.push(indexedDictionary[token.toLowerCase()]);
    });

    tokenizeMessage(key).forEach(token => {
        target.push(indexedDictionary[token.toLowerCase()]);
    });

    inputs.push(limitArraySize(input, inputSize, 0));
    targets.push(limitArraySize(target, outputSize, 0));
}


for (let i = 0; i < numIterations; i++) {
    for (let j = 0; j < inputs.length; j++) {
        const input = new Matrix(1, inputSize);
        inputs[j].forEach((token, index) => {
            input.set(0, index, token);
        });
       
        const target = new Matrix(1, outputSize);
        targets[j].forEach((token, index) => {
            target.set(0, index, token);
        });
       

        model.train(input, target, learningRate, 1);
    }
}

function generateResponse(input) {
    const tokens = tokenizeMessage(input);
    var inputArray = [];
    tokens.forEach(token => {
        inputArray.push(indexedDictionary[token.toLowerCase()]);
    });

    inputArray = limitArraySize(inputArray, inputSize, 0)

    const inputMatrix = new Matrix(1, inputSize);
    inputArray.forEach((token, index) => {
        inputMatrix.set(0, index, token);
    });

    const outputMatrix = model.forward(inputMatrix);
    console.log(outputMatrix);
    const outputArray = outputMatrix.data;
        const outputTokens = outputArray[0].map(token => {
            token = (token * (max - min)) + min;
            for (const key in indexedDictionary) {
                if ( (indexedDictionary[key] * (max - min)) + min === Math.ceil(token)) {
                    return key;
                }
            }
        }
    );
    return outputTokens.join(" ");
}

console.log(indexedDictionary);

document.addEventListener('DOMContentLoaded', function() {
    const inputElement = document.getElementById('inputElement');

    inputElement.addEventListener('input', function(event) {
        const userInput = event.target.value;
        const response = generateResponse(userInput);
        console.log(response);
    });
});