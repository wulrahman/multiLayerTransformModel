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
    "Hi there! How's everything?",
    "Hello! How are you doing today?",
    "Hey! What have you been up to?",
    "Hi! How's your day going?",
    "Hello there! How's life treating you?",
    "Hey! How are things on your end?",
    "Hi! How's your week been so far?",
    "Hello! How have you been lately?",
    "Hey! What's new with you?",
    "Good morning! How are you today?",
    "Hey! What's up?",
    "Hi! How's your day been so far?",
    "Hello! How's the weather where you are?",
    "Hey there! How was your weekend?",
    "Hi! Did you have a good night's sleep?",
    "Hello! How's work going?",
    "Hey! How's your family doing?",
    "Hi there! How's your pet doing?",
    "Hello! Have you watched any good movies lately?"
];

$target_array = [
    "Hey! My day is going well, thanks for asking!",
    "Hi! Everything's going fine, thanks for checking in!",
    "Hello! I'm doing good, thanks for asking.",
    "Hey! I've been keeping busy, thanks for asking!",
    "Hi there! My day is going great, thanks!",
    "Hello! Life's been treating me well, thank you!",
    "Hey! Things are good on my end, how about you?",
    "Hi! My week's been pretty good, thanks for asking!",
    "Hello! I've been doing well lately, thanks!",
    "Hey! Not much, just catching up on things.",
    "Good morning! I'm doing well, thank you!",
    "Hey! Not much, just chilling.",
    "Hi! My day's been busy, but good!",
    "Hello! It's sunny here, how about there?",
    "Hey there! My weekend was relaxing, thanks!",
    "Hi! Yes, I slept well, thanks for asking!",
    "Hello! Work's been hectic but manageable.",
    "Hey! Family's doing great, thanks!",
    "Hi there! My pet's doing fine, just napping.",
    "Hello! Yeah, I watched a great movie recently!"
];


// Set context window size
$contextWindowSize = 2;
$inputSize = 20;
$outputSize = 20;
$hiddenSize = 20;
$numLayers = 2;

// Create a chatbot

$model_filename = 'model1.dat';


if (file_exists($model_filename)) {
    // Load the existing model from the file
    $chatbot = unserialize(file_get_contents($model_filename));
} else {
    // Create a new model instance
    $chatbot = new Chatbot($inputSize, $outputSize, $hiddenSize, $numLayers, $contextWindowSize);
    // Serialize and save the model to a file
    file_put_contents($model_filename, serialize($chatbot));
}


$train_model = true;
if(!$train_model ) {
    $epochs = 100;
    $learningRate = 1e-2;


    // Train word embeddings
    for($i = 0; $i < $epochs; $i++) {

        foreach ($input_array as $key => $input) {
            $target = $target_array[$key];

            $words_input = explode(" ", $input);
            $words_target = explode(" ", $target);

            $chatbot->trainEmbeddings($words_input, 1, $learningRate);
            $chatbot->trainEmbeddings($words_target, 1, $learningRate);
        }
    }
}
else {
    $epochs = 100;
    $learningRate = 1e-8;


    // Train the chatbot
    for($i = 0; $i < $epochs; $i++) {
        foreach ($input_array as $key => $input) {
            $target = $target_array[$key];
            $chatbot->train($input, $target, $learningRate, 1);
        }
    }
}

// $chatbot->trainEmbeddings($words, $epochs, $learningRate);
// $chatbot->train($input_array[0], $target_array[0], $learningRate, $epochs);

// Example interaction
echo $target_array[10];
$userInput = $input_array[10];

$response = $chatbot->respond($userInput);
echo "Chatbot: ";

file_put_contents($model_filename, serialize($chatbot));

echo "<pre>";
print_r( $response);
echo "</pre>";

$url1=$_SERVER['REQUEST_URI'];
header("Refresh: 1; URL=$url1");
