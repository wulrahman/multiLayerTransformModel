<?php
require_once __DIR__ . '/Math.php';
require_once __DIR__ . '/Transformer.php';
require_once __DIR__ . '/SelfAttention.php';
require_once __DIR__ . '/EncoderDecoder.php';
require_once __DIR__ . '/BuildHuffmanGramModel.php';
require_once __DIR__ . '/MultiLayerTransformer.php';
require_once __DIR__ . '/MultiLayerEncoderDecoder.php';
require_once __DIR__ . '/Chatbot.php';
require_once __DIR__ . '/DocumentProcessor.php';


// Example usage
$inputSize = 2;
$outputSize = 2;
$hiddenSize = 2;
$numLayers = 3;

$documentProcessor = new DocumentProcessor($inputSize, $outputSize, $hiddenSize, $numLayers);

function splitIntoChunks($string, $wordSize) {
    $words = explode(" ", $string); // Split the string into words
    $chunks = array_chunk($words, $wordSize); // Split the words into chunks of specified size
    $chunkedStrings = array_map(function($chunk) {
        return implode(" ", $chunk); // Join the words in each chunk back into a string
    }, $chunks);
    return $chunkedStrings;
}

$question1 = "Question: What is the capital of France?";
$answer1 = "Answer: The capital of France is Paris.";

$question2 = "Question: How many continents are there in the world?";
$answer2 = "Answer: There are seven continents in the world: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.";

$question3 = "Question: What is the boiling point of water in Celsius?";
$answer3 = "Answer: The boiling point of water in Celsius is 100 degrees.";

$question4 = "Question: Who wrote the play 'Romeo and Juliet'?";
$answer4 = "Answer: 'Romeo and Juliet' was written by William Shakespeare.";

// Example usage:
$question1Chunks = splitIntoChunks($question1, $inputSize);
$answer1Chunks = splitIntoChunks($answer1, $outputSize);

$learningRate = 0.001;
$epochs = 10000;

$documentProcessor->train($question1Chunks, $answer1Chunks, $learningRate, $epochs);

$encodeInput = $question1Chunks;
$encodedOutput = $documentProcessor->encode($encodeInput);
$decodedOutput = $documentProcessor->decode($encodedOutput);

foreach ($decodedOutput as $output) {
    echo $output . "\n";
}


?>