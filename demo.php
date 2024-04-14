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
    "Hey! How's your day treating you?",
    "Good day! What's new with you?",
    "Hi there! How's life been treating you lately?",
    "Hello! How are things on your end?",
    "Hey! How's the weather over there?",
    "Good afternoon! How's your week been so far?",
    "Hi! What have you been up to recently?",
    "Hello! How's your family doing?",
    "Hey! How was your weekend?",
    "Good evening! How's your day winding down?",
    "Hi there! How's work going for you?",
    "Hello! How's your health these days?",
    "Hey! Any exciting plans for the weekend?",
    "Good day! How's your mood today?",
    "Hi! How's the new project coming along?",
    "Hello! How's your pet doing?",
    "Hey! Have you tried any new restaurants lately?",
    "Good afternoon! How's your energy level?",
    "Hi! How's your commute been?",
    "Hello! How's your hobby going?"
];

$target_array = [
    "Hey! My day is going well, thanks for asking!",
    "Good day! Not much, just the usual.",
    "Hi there! Life's been treating me pretty well, thank you.",
    "Hello! Things are going smoothly on my end.",
    "Hey! The weather here is pretty nice.",
    "Good afternoon! My week has been busy but good.",
    "Hi! Just been catching up on some work.",
    "Hello! Family is doing great, thanks for asking!",
    "Hey! The weekend was relaxing, thanks for asking.",
    "Good evening! My day is winding down nicely.",
    "Hi there! Work has been challenging but fulfilling.",
    "Hello! Health-wise, I'm feeling good.",
    "Hey! Just planning to relax over the weekend.",
    "Good day! My mood is pretty positive today.",
    "Hi! The new project is coming along nicely.",
    "Hello! My pet is doing great, thanks for asking!",
    "Hey! Yes, tried a new place and it was amazing.",
    "Good afternoon! Energy level is decent.",
    "Hi! Commute has been smooth, thanks.",
    "Hello! My hobby is keeping me busy and happy."
];

// Set context window size
$contextWindowSize = 2;
$inputSize = 20;
$outputSize = 20;
$hiddenSize = 20;
$numLayers = 1;
$learningRate = 0.01;
$epochs = 100;

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