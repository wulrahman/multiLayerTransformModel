<?php

class TransformerInputPreprocessor {
    private $tokenizer;
    private $embeddingModel;
    private $maxSequenceLength;
    
    public function __construct($tokenizer, $embeddingModel, $maxSequenceLength) {
        $this->tokenizer = $tokenizer;
        $this->embeddingModel = $embeddingModel;
        $this->maxSequenceLength = $maxSequenceLength;
    }

    public function preprocess($inputText) {
        // Tokenization
        $tokens = $this->tokenizer->tokenize($inputText);

        // Truncate tokens if exceeding max sequence length
        $tokens = array_slice($tokens, 0, $this->maxSequenceLength);

        // Convert tokens to numerical indices
        $tokenIds = [];
        foreach ($tokens as $token) {
            $tokenId = $this->tokenizer->getTokenId($token);
            if ($tokenId !== null) {
                $tokenIds[] = $tokenId;
            } else {
                // Handle unknown tokens
                // For example, you could use a special token for unknown tokens
                // Here, we'll skip unknown tokens
            }
        }

        // Pad sequence if shorter than max sequence length
        while (count($tokenIds) < $this->maxSequenceLength) {
            $tokenIds[] = $this->tokenizer->getPaddingTokenId();
        }

        // Lookup embeddings for token IDs
        $embeddings = [];
        foreach ($tokenIds as $tokenId) {
            $embedding = $this->embeddingModel->getEmbedding($tokenId);
            $embeddings[] = $embedding;
        }

        // Add positional encoding
        $positionEncoding = $this->getPositionalEncoding(count($tokenIds));
        foreach ($embeddings as &$embedding) {
            $embedding = array_map(function($x, $y) { return $x + $y; }, $embedding, $positionEncoding);
        }

        // Return preprocessed input
        return $embeddings;
    }

    private function getPositionalEncoding($sequenceLength) {
        // Generate sinusoidal positional encoding
        // Implementation of positional encoding depends on your requirements
        // Here's a simple example for demonstration purposes
        $positionalEncoding = [];
        for ($pos = 0; $pos < $sequenceLength; $pos++) {
            $positionalEncoding[$pos] = [];
            for ($i = 0; $i < $this->embeddingModel->getEmbeddingDimension(); $i++) {
                $positionalEncoding[$pos][$i] = sin($pos / (10000 ** ($i / $this->embeddingModel->getEmbeddingDimension())));
            }
        }
        return $positionalEncoding;
    }
}



// // Define vocabulary
// $vocab = ["[PAD]", "[UNK]", "How", "are", "you", "?"];

// // Instantiate tokenizer
// $tokenizer = new Tokenizer($vocab, 1, 0); // Assuming "[UNK]" has ID 1 and "[PAD]" has ID 0

// // Dummy embedding model (just for demonstration)
// class DummyEmbeddingModel {
//     public function getEmbedding($tokenId) {
//         // Dummy embedding lookup
//         return array_fill(0, 10, 0.1); // Dummy embedding vector of dimension 10
//     }
    
//     public function getEmbeddingDimension() {
//         return 10; // Dummy embedding dimension
//     }
// }

// $embeddingModel = new DummyEmbeddingModel();

// // Instantiate TransformerInputPreprocessor
// $maxSequenceLength = 10; // Maximum sequence length
// $inputPreprocessor = new TransformerInputPreprocessor($tokenizer, $embeddingModel, $maxSequenceLength);
// // Example input text
// $inputText = "How are you?";

// // Preprocess input
// $inputData = $inputPreprocessor->preprocess($inputText);

// // Print preprocessed input
// print_r($inputData);