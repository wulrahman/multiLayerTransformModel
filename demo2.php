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
$inputSize = 4;
$outputSize = 4;
$hiddenSize = 4;
$numLayers = 1;

function splitIntoChunks($string, $wordSize) {
    $words = explode(" ", $string); // Split the string into words
    $chunks = array_chunk($words, $wordSize); // Split the words into chunks of specified size
    $chunkedStrings = array_map(function($chunk) {
        return implode(" ", $chunk); // Join the words in each chunk back into a string
    }, $chunks);
    return $chunkedStrings;
}

$documents[] = ['question' => "What is the capital of France?", 'answer' => "The capital of France is Paris."];
$documents[] = ['question' => "How many continents are there in the world?", 'answer' => "There are seven continents in the world: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."];
$documents[] = ['question' => "What is the boiling point of water in Celsius?", 'answer' => "The boiling point of water in Celsius is 100 degrees."];
$documents[] = ['question' => "Who wrote the play 'Romeo and Juliet'?.", 'answer' => "Romeo and Juliet' was written by William Shakespeare."];
$documents[] = ['question' => "What is the largest ocean on Earth?", 'answer' => "The largest ocean on Earth is the Pacific Ocean."];
$documents[] = ['question' => "What is the capital of Japan?", 'answer' => "The capital of Japan is Tokyo."];
$documents[] = ['question' => "What is the chemical symbol for water?", 'answer' => "The chemical symbol for water is H2O."];
$documents[] = ['question' => "Who invented the telephone?", 'answer' => "The telephone was invented by Alexander Graham Bell."];
$documents[] = ['question' => "What is the currency of the United Kingdom?", 'answer' => "The currency of the United Kingdom is the British Pound Sterling."];
$documents[] = ['question' => "Who painted the Mona Lisa?", 'answer' => "The Mona Lisa was painted by Leonardo da Vinci."];
$documents[] = ['question' => "What is the chemical symbol for gold?", 'answer' => "The chemical symbol for gold is Au."];
$documents[] = ['question' => "What is the largest planet in our solar system?", 'answer' => "The largest planet in our solar system is Jupiter."];
$documents[] = ['question' => "Who wrote 'To Kill a Mockingbird'?", 'answer' => "To Kill a Mockingbird' was written by Harper Lee."];
$documents[] = ['question' => "What is the tallest mountain in the world?", 'answer' => "The tallest mountain in the world is Mount Everest."];
$documents[] = ['question' => "What is the chemical symbol for oxygen?", 'answer' => "The chemical symbol for oxygen is O."];
$documents[] = ['question' => "Who discovered penicillin?", 'answer' => "Penicillin was discovered by Alexander Fleming."];
$documents[] = ['question' => "What is the speed of light in a vacuum?", 'answer' => "The speed of light in a vacuum is approximately 299,792 kilometers per second."];
$documents[] = ['question' => "What is the national animal of China?", 'answer' => "he national animal of China is the giant panda."];
$documents[] = ['question' => "Who wrote 'The Great Gatsby'?", 'answer' => "The Great Gatsby' was written by F. Scott Fitzgerald."];
$documents[] = ['question' => "What is the chemical symbol for carbon?", 'answer' => "The chemical symbol for carbon is C."];
$documents[] = ['question' => "What is the longest river in the world?", 'answer' => "The longest river in the world is the Nile River."];
$documents[] = ['question' => "Who discovered gravity?", 'answer' => "Gravity was discovered by Sir Isaac Newton."];
$documents[] = ['question' => "What is the melting point of ice in Celsius?", 'answer' => "The melting point of ice in Celsius is 0 degrees."];
$documents[] = ['question' => "Who painted 'Starry Night'?", 'answer' => "'Starry Night' was painted by Vincent van Gogh."];
$documents[] = ['question' => "What is the chemical symbol for sodium?", 'answer' => "The chemical symbol for sodium is Na."];
$documents[] = ['question' => "What is the largest desert in the world?", 'answer' => "The largest desert in the world is the Sahara Desert."];
$documents[] = ['question' => "Who discovered the theory of relativity?", 'answer' => "The theory of relativity was discovered by Albert Einstein."];
$documents[] = ['question' => "What is the boiling point of mercury in Celsius?", 'answer' => "The boiling point of mercury in Celsius is 356.7 degrees."];
$documents[] = ['question' => "What is the capital of Russia?", 'answer' => "The capital of Russia is Moscow."];
$documents[] = ['question' => "How many bones are there in the human body?", 'answer' => "There are 206 bones in the human body."];
$documents[] = ['question' => "Who is the author of 'Pride and Prejudice'?", 'answer' => "Pride and Prejudice was written by Jane Austen."];
$documents[] = ['question' => "What is the chemical symbol for iron?", 'answer' => "The chemical symbol for iron is Fe."];
$documents[] = ['question' => "What is the highest mountain in North America?", 'answer' => "The highest mountain in North America is Denali (formerly known as Mount McKinley)."];
$documents[] = ['question' => "Who discovered electricity?", 'answer' => "Electricity was discovered by many scientists over time, but Benjamin Franklin is often credited with its discovery."];
$documents[] = ['question' => "What is the most populous country in the world?", 'answer' => "The most populous country in the world is China."];
$documents[] = ['question' => "Who wrote 'The Catcher in the Rye'?", 'answer' => "The Catcher in the Rye was written by J.D. Salinger."];
$documents[] = ['question' => "What is the chemical symbol for silver?", 'answer' => "The chemical symbol for silver is Ag."];
$documents[] = ['question' => "What is the longest bone in the human body?", 'answer' => "The longest bone in the human body is the femur."];
$documents[] = ['question' => "Who was the first person to walk on the moon?", 'answer' => "The first person to walk on the moon was Neil Armstrong."];
$documents[] = ['question' => "What is the chemical symbol for lead?", 'answer' => "The chemical symbol for lead is Pb."];
$documents[] = ['question' => "What is the largest mammal in the world?", 'answer' => "The largest mammal in the world is the blue whale."];
$documents[] = ['question' => "Who is the Greek god of the sea?", 'answer' => "The Greek god of the sea is Poseidon."];
$documents[] = ['question' => "What is the chemical symbol for helium?", 'answer' => "The chemical symbol for helium is He."];
$documents[] = ['question' => "What is the largest bird in the world?", 'answer' => "The largest bird in the world is the ostrich."];
$documents[] = ['question' => "Who painted 'The Scream'?", 'answer' => "'The Scream' was painted by Edvard Munch."];
$documents[] = ['question' => "What is the chemical symbol for nitrogen?", 'answer' => "The chemical symbol for nitrogen is N."];
$documents[] = ['question' => "What is the hottest planet in our solar system?", 'answer' => "The hottest planet in our solar system is Venus."];
$documents[] = ['question' => "Who discovered penicillin?", 'answer' => "Penicillin was discovered by Alexander Fleming."]; // This one is a duplicate, you may remove it if needed.
$documents[] = ['question' => "What is the chemical symbol for potassium?", 'answer' => "The chemical symbol for potassium is K."];
$documents[] = ['question' => "What is the largest oceanic mammal?", 'answer' => "The largest oceanic mammal is the blue whale."];
$documents[] = ['question' => "Who was the first woman to win a Nobel Prize?", 'answer' => "Marie Curie was the first woman to win a Nobel Prize."];
$documents[] = ['question' => "What is the chemical symbol for silver?", 'answer' => "The chemical symbol for silver is Ag."]; // This one is a duplicate, you may remove it if needed.
$documents[] = ['question' => "What is the fastest land animal?", 'answer' => "The fastest land animal is the cheetah."];
$documents[] = ['question' => "Who wrote '1984'?", 'answer' => "'1984' was written by George Orwell."];
$documents[] = ['question' => "What is the chemical symbol for silicon?", 'answer' => "The chemical symbol for silicon is Si."];
$documents[] = ['question' => "What is the largest moon in the solar system?", 'answer' => "The largest moon in the solar system is Ganymede, a moon of Jupiter."];
$documents[] = ['question' => "Who was the first president of the United States?", 'answer' => "The first president of the United States was George Washington."];


$model_filename = 'model.dat';
if (file_exists($model_filename)) {
    // Load the existing model from the file
    $documentProcessor = unserialize(file_get_contents($model_filename));
} else {
    // Create a new model instance
    $documentProcessor = new DocumentProcessor($inputSize, $outputSize, $hiddenSize, $numLayers);
    // Serialize and save the model to a file
    file_put_contents($model_filename, serialize($documentProcessor));
}

$train = false;
$epochs = 10;
$learningRate = 1e-8;

if($train) {
    for($i = 0; $i < $epochs; $i++) {
        foreach ($documents as $document) {
            $question = $document['question'];
            $answer = $document['answer'];
            $questionChunks = splitIntoChunks($question, $inputSize);
            $answerChunks = splitIntoChunks($answer, $outputSize);
            $documentProcessor->train($questionChunks, $answerChunks, $learningRate, 1);
        }    
    }
}


$questionChunks = splitIntoChunks($documents[2]['question'], $inputSize);

$encodedOutput = $documentProcessor->encode($questionChunks);
$decodedOutput = $documentProcessor->decode($encodedOutput);

foreach ($decodedOutput as $output) {
    echo $output . "\n";
}

file_put_contents($model_filename, serialize($documentProcessor));


$url1=$_SERVER['REQUEST_URI'];
header("Refresh: 1; URL=$url1");
?>