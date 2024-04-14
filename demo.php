<?php
require_once __DIR__ . '/Math.php';
require_once __DIR__ . '/Transformer.php';
require_once __DIR__ . '/SelfAttention.php';
require_once __DIR__ . '/EncoderDecoder.php';
require_once __DIR__ . '/BuildHuffmanGramModel.php';
require_once __DIR__ . '/MultiLayerTransformer.php';
require_once __DIR__ . '/MultiLayerEncoderDecoder.php';
require_once __DIR__ . '/Chatbot.php';

// Initialize word embeddings

// Example input
$input_array = [
    "Hello, how are you?",
    "Good morning! How's your day going?",
    "Hey there! What's up?",
    "Hi! How have you been lately?",
    "Hello! How's everything with you?"
];

// Corresponding target responses
$target_array = [
    "I'm doing great, thanks!",
    "Good morning! My day is off to a good start, thank you for asking!",
    "Hey! Not much, just enjoying the day.",
    "Hi! I've been doing well, thank you for asking.",
    "Hello! Everything's going well, thanks for checking in!"
];

// Set context window size
$contextWindowSize = 2;
$inputSize = 10;
$outputSize = 10;
$hiddenSize = 10;
$numLayers = 10;
$learningRate = 0.001;
$epochs = 1000;

// Create a chatbot

$model_filename = 'model.dat';


if (file_exists($model_filename)) {
    // Load the existing model from the file
    $chatbot = unserialize(file_get_contents($model_filename));
} else {
    // Create a new model instance
    $chatbot = new Chatbot($inputSize, $outputSize, $hiddenSize, $numLayers, $contextWindowSize);
    // Serialize and save the model to a file
    file_put_contents($model_filename, serialize($chatbot));
}


// Train word embeddings
foreach ($input_array as $key => $input) {
    $target = $target_array[$key];

    $words_input = explode(" ", $input);
    $words_target = explode(" ", $target);

    $chatbot->trainEmbeddings($words_input, $epochs, $learningRate);
    $chatbot->trainEmbeddings($words_target, $epochs, $learningRate);
}

// Train word embeddings
foreach ($input_array as $key => $input) {
    $target = $target_array[$key];
    $chatbot->train($input, $target, $learningRate, $epochs);
}

// $chatbot->trainEmbeddings($words, $epochs, $learningRate);
// $chatbot->train($input_array[0], $target_array[0], $learningRate, $epochs);

// Example interaction
$userInput = "Hello, how are you?";
$response = $chatbot->respond($userInput);
echo "Chatbot: ";

file_put_contents($model_filename, serialize($chatbot));

print_r( $response);





?>