<?php

$document = ['question' => "What is the capital of France?", 'answer' => "The capital of France is Paris."];

// Function to tokenize the text
function tokenizeText($text) {
    // Split the text into tokens (words) by whitespace
    $tokens = explode(" ", $text);
    return $tokens;
}

// Sample document

// Tokenize question and answer
function createInputOutputPairs($document) {
    $questionTokens = tokenizeText($document['question']);
    $answerTokens = tokenizeText($document['answer']);

    $inputOutputPairs = [];

    // Create input-output pairs for each segment of the answer
    for ($i = 0; $i <= count($answerTokens) + count($answerTokens); $i++) {
        // Input sequence is the question tokens
        $inputSequence = $questionTokens;
        $inputSequence = array_slice($inputSequence, $i, count($inputSequence));

        // Output sequence is the first $i tokens of the answer
        $outputSequence = array_slice($answerTokens, 0, $i);
        // Store the input-output pair
        $inputOutputPairs[] = ['input' => $inputSequence, 'output' => $outputSequence];
    }

    return $inputOutputPairs;
}

$inputOutputPairs = createInputOutputPairs($document);

echo "<pre>";
print_r($inputOutputPairs);
echo "</pre>";

echo "<pre>";
print_r($inputOutputPairs);
echo "</pre>";

// Print input and output sequences (for demonstration)
// foreach ($inputOutputPairs as $index => $pair) {
//     echo "Input Sequence ($index): " . implode(" ", $pair['input']) . "\n";
//     echo "Output Sequence ($index): " . implode(" ", $pair['output']) . "\n";
//     echo "\n";
// }