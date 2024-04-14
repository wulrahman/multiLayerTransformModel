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
];

$target_array = [
    "Hey! My day is going well, thanks for asking!",
];

// Set context window size
$contextWindowSize = 2;
$inputSize = 11;
$outputSize = 11;
$hiddenSize = 11;
$numLayers = 1;
$learningRate = 0.01;
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
// for($i = 0; $i < $epochs; $i++) {

//     foreach ($input_array as $key => $input) {
//         $target = $target_array[$key];

//         $words_input = explode(" ", $input);
//         $words_target = explode(" ", $target);

//         $chatbot->trainEmbeddings($words_input, 1, $learningRate);
//         $chatbot->trainEmbeddings($words_target, 1, $learningRate);
//     }
// }

for($i = 0; $i < $epochs; $i++) {
    foreach ($input_array as $key => $input) {
        $target = $target_array[$key];
        $chatbot->train($input, $target, $learningRate, 1);
    }
}
// Train word embeddings
foreach ($input_array as $key => $input) {
    $target = $target_array[$key];
    $chatbot->train($input, $target, $learningRate, $epochs);
}

// $chatbot->trainEmbeddings($words, $epochs, $learningRate);
// $chatbot->train($input_array[0], $target_array[0], $learningRate, $epochs);

// Example interaction
$userInput = "Hey! How's your day treating you?";
$response = $chatbot->respond($userInput);
echo "Chatbot: ";

file_put_contents($model_filename, serialize($chatbot));

print_r( $response);

$url1=$_SERVER['REQUEST_URI'];
header("Refresh: 1; URL=$url1");


?>