<?php
class BuildHuffmanGramModel {
    public $wordVectors = []; // Associative array to store word vectors
    private $vocabSize;
    private $embeddingSize;

    public $wordToIndex = [];

    private $negativeSamples; // Define the number of negative samples

    private $wordFrequencies = [];

    private $huffmanParent = [];
    private $subsamplingThreshold;



    public function __construct($embeddingSize) {
        $this->embeddingSize = $embeddingSize;
        $this->subsamplingThreshold = 1e-3;
        $this->wordToIndex[' '] = 0;
        $this->wordVectors[0]  = $this->zeros(1, $this->embeddingSize)[0];
        $this->wordFrequencies[0] = 1;
    }

    public function zeros($rows, $cols) {
        $matrix = [];
        for ($i = 0; $i < $rows; $i++) {
            $row = [];
            for ($j = 0; $j < $cols; $j++) {
                $row[] = 0;
            }
            $matrix[] = $row;
        }
        return $matrix;
    }
    public function getBinaryPath($contextWordIndex) {
        $path = [];
    
        while ($contextWordIndex !== null) {
            $path[] = $contextWordIndex;
            $contextWordIndex = $this->huffmanParent[$contextWordIndex] ?? null; // Navigate to parent node
        }
    
        return $path;
    }
    public function forwardPropagation($centerWordIndex, $contextWordIndex) {
        $path = $this->getBinaryPath($contextWordIndex); // Get the binary path to the context word
        
        $prob = 1.0;
        foreach ($path as $nodeIndex) {
            $prob *= $this->sigmoid($this->dotProduct($this->wordVectors[$centerWordIndex], $this->wordVectors[$nodeIndex]));
        }
    
        return $prob;
    }

    
    public function findClosestWord($sumVector) {
        $closestWord = null;
        $maxSimilarity = -1; // Initialize to a low value

        foreach ($this->wordVectors as $word => $wordVector) {
            $similarity = $this->cosineSimilarity($sumVector, $wordVector);
            
            if ($similarity > $maxSimilarity) {
                $maxSimilarity = $similarity;
                $closestWord = $word;
            }
        }

        return $key = array_search($closestWord, $this->wordToIndex);
    }


    // Define the dotProduct function
    function dotProduct($vector1, $vector2) {
        $result = 0;
        $length = count($vector1); // Assuming both vectors have the same length

        for ($i = 0; $i < $length; $i++) {
            $result += $vector1[$i] * $vector2[$i];
        }

        return $result;
    }

    private function buildHuffmanTree() {
        $this->vocabSize =  count($this->wordVectors);

        $wordFrequencies = $this->wordFrequencies; // Replace this with your actual word frequencies
    
        // Create initial nodes for each word
        $nodes = [];
        for ($i = 0; $i < $this->vocabSize; $i++) {
            $nodes[] = ['index' => $i, 'frequency' => $wordFrequencies[$i] ?? 0];
        }
    
        // Build the Huffman tree
        while (count($nodes) > 1) {
            usort($nodes, function ($a, $b) {
                return $a['frequency'] - $b['frequency'];
            });
    
            $min1 = array_shift($nodes);
            $min2 = array_shift($nodes);
    
            // Create a new node representing the merged nodes
            $mergedNode = ['index' => null, 'frequency' => $min1['frequency'] + $min2['frequency'], 'children' => [$min1, $min2]];
            
            // Add the merged node back to the list of nodes
            $nodes[] = $mergedNode;
        }
    
        // Populate $huffmanParent based on the tree structure
        $this->populateHuffmanParent($nodes[0], null);
    }
    
    private function populateHuffmanParent($node, $parentIndex) {
        if ($node['index'] !== null) {
            // This is a leaf node
            $this->huffmanParent[$node['index']] = $parentIndex;
        } else {
            // This is an internal node with children
            foreach ($node['children'] as $childNode) {
                $this->populateHuffmanParent($childNode, $node['index']);
            }
        }
    }

    private function sigmoid($x) {
        return 1 / (1 + exp(-$x));
    }

    public function backwardPropagation($centerWordIndex, $contextWordIndex, $learningRate) {

        $this->vocabSize =  count($this->wordVectors);

        $predictedProbability = $this->forwardPropagation($centerWordIndex, $contextWordIndex);
        $gradient = $predictedProbability - 1;

        $path = $this->getBinaryPath($contextWordIndex);
        
        foreach ($path as $nodeIndex) {
            for ($i = 0; $i < $this->embeddingSize; $i++) {
                $this->wordVectors[$centerWordIndex][$i] -= $learningRate * $gradient * $this->wordVectors[$nodeIndex][$i];
                $this->wordVectors[$nodeIndex][$i] -= $learningRate * $gradient * $this->wordVectors[$centerWordIndex][$i];
            }
        }
    }
    

    public function train($trainingPairs, $epochs, $initialLearningRate) {
        $learningRate = $initialLearningRate;
    
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            foreach ($trainingPairs as $pair) {
                $centerWordIndex = $pair[0];
                $contextWordIndex = $pair[1];
                
                // Adjust learning rate dynamically
                $learningRate *= 0.95; // Reduce the learning rate by a factor
                
                // Apply subsampling
                if (mt_rand() / mt_getrandmax() < $this->getSubsamplingProbability($contextWordIndex)) {
                    continue; // Skip this context word
                }
    
                // Use the adjusted learning rate in backwardPropagation
                $this->backwardPropagation($centerWordIndex, $contextWordIndex, $learningRate);
            }
        }
    }
    

    public function evaluateSimilarity($word1, $word2) {
        $vector1 = $this->wordVectors[$word1];
        $vector2 = $this->wordVectors[$word2];
        return $this->cosineSimilarity($vector1, $vector2);
    }

    private function initializeRandomVector() {
        $vector = [];
        for ($i = 0; $i < $this->embeddingSize; $i++) {
            $vector[] = (mt_rand() / mt_getrandmax() - 0.5) / $this->embeddingSize;
        }
        return $vector;
    }

    private function cosineSimilarity($vectorA, $vectorB) {
        $dotProduct = 0;
        $normA = 0;
        $normB = 0;

        for ($i = 0; $i < $this->embeddingSize; $i++) {
            $dotProduct += $vectorA[$i] * $vectorB[$i];
            $normA += $vectorA[$i] * $vectorA[$i];
            $normB += $vectorB[$i] * $vectorB[$i];
        }

        if ($normA == 0 || $normB == 0) {
            return 0; // Handle division by zero
        }

        return $dotProduct / (sqrt($normA) * sqrt($normB));
    }
    public function generateTrainingPairs($vocabulary, $contextWindowSize) {
        $trainingPairs = [];

        foreach($vocabulary as $key => $value) {
            $value = strtolower($value);
            $value = preg_replace('/[^a-z0-9]+/i', '', $value);

            if(!array_key_exists($value, $this->wordToIndex)) {
                // Initialize word vectors randomly
                $this->wordVectors[count($this->wordToIndex)] = $this->initializeRandomVector();
                $this->wordToIndex[$value] = count($this->wordToIndex);
                $this->wordFrequencies[$this->wordToIndex[$value]] = 1;
            }
            else {
                $this->wordFrequencies[$this->wordToIndex[$value]]++;
            }
            
        }

        for ($i = $contextWindowSize; $i < count($vocabulary) - $contextWindowSize; $i++) {
            $centerWordIndex = $i;

            for ($j = $i - $contextWindowSize; $j <= $i + $contextWindowSize; $j++) {
                if ($j !== $i) {
                    $contextWordIndex = $j;
                    $trainingPairs[] = [$centerWordIndex, $contextWordIndex];
                }
            }
        }

        $this->buildHuffmanTree();

        return $trainingPairs;
    }

    public function displayTrainingPairs($vocabulary, $trainingPairs) {
        foreach ($trainingPairs as $pair) {
            $centerWord = $vocabulary[$pair[0]];
            $contextWord = $vocabulary[$pair[1]];
            echo "Center word: $centerWord, Context word: $contextWord\n";
        }
    }

    
    function getSubsamplingProbability($wordIndex) {
        $totalWords =  count($this->wordVectors);
        $wordFrequency = $this->wordFrequencies[$wordIndex] ?? 0;
        return 1 - sqrt($this->subsamplingThreshold / ($wordFrequency / $totalWords));
    }

    public function getWordVector($wordIndex) {
        return $this->wordVectors[$wordIndex];
    }
    
    public function getWordIndex($word) {
        $word = strtolower($word);
        $word = preg_replace('/[^a-z0-9]+/i', '', $word);
        return $this->wordToIndex[$word];
    }
}
