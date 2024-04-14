<?php

class Transformer {
    private $weights;
    private $bias;

    public function __construct($inputSize, $outputSize) {
        // Initialize weights and bias based on input and output sizes
        $this->initializeWeights($inputSize, $outputSize);
        $this->initializeBias($outputSize);
    }

    private function initializeWeights($inputSize, $outputSize) {
        // Initialize weights matrix with random values
        $this->weights = [];
        for ($i = 0; $i < $outputSize; $i++) {
            $this->weights[] = array_fill(0, $inputSize, 0.0); // Initialize with zeros, you can change this if needed
        }
    }

    private function initializeBias($outputSize) {
        // Initialize bias vector with zeros
        $this->bias = array_fill(0, $outputSize, 0.0); // Initialize with zeros, you can change this if needed
    }
    
    public function forward($input) {
        // Matrix multiplication
        $output = Math::matmul($this->weights, $input);

        // Add bias
        $output = Math::addMatVector($output, $this->bias);

        // Apply activation function (e.g., ReLU)
        $output = Math::relu($output);

        return $output;
    }

    public function train($input, $target, $learningRate, $epochs) {
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            
            // Forward pass
            $output = $this->forward($input);

            // Calculate loss
            $loss = Math::sub($output, $target);

            // Backward pass
            $gradient = Math::mul($loss, $learningRate);

            // Update weights
            $this->weights = Math::sub($this->weights, $gradient);

            // echo array_sum(array_map('array_sum', $loss)) . "\n\r";

            // Update bias
            $this->bias = Math::subVectorValue($this->bias, $learningRate * array_sum(array_map('array_sum', $loss)));
        }
    }

    public function backward($gradient) {
        // Calculate loss gradient
        $lossGradient = Math::matmul(Math::transpose($this->weights), $gradient);

        // Update weights
        $this->weights = Math::sub($this->weights, $gradient);

        // Update bias
        $this->bias = Math::subVectorValue($this->bias, array_sum(array_map('array_sum', $gradient)));

        return $lossGradient;
    }

    public function getWeights() {
        return $this->weights;
    }

    public function getBias() {
        return $this->bias;
    }

    public function setWeights($weights) {
        $this->weights = $weights;
    }

    public function setBias($bias) {
        $this->bias = $bias;
    }

    public function getParams() {
        return [$this->weights, $this->bias];
    }

    public function setParams($params) {
        $this->weights = $params[0];
        $this->bias = $params[1];
    }

    public function __toString() {
        return "Weights: " . json_encode($this->weights) . "\nBias: " . json_encode($this->bias);
    }

    public function __serialize(): array {
        return [
            'weights' => $this->weights,
            'bias' => $this->bias
        ];
    }
}

// // Create a transformer
// $inputSize = 2; // Example input size
// $outputSize = 2; // Example output size

// // Create a transformer
// $transformer = new Transformer($inputSize, $outputSize);

// // Define input and target
// $inputs[] = [[5, 6], [7, 8]];
// $targets[] = [[7, 8], [9, 10]];
// // Define input and target
// $inputs[] = [[1, 2], [3, 4]];
// $targets[] = [[3, 4], [5, 6]];

// // Train the transformer
// $learningRate = 0.01;
// $epochs = 10;
// for($i = 0; $i < $epochs ; $i++) {
//     foreach ($inputs as $index => $input) {
//         $target = $targets[$index];
//         $transformer->train($input, $target, $learningRate, $epochs);
//     }
// }

// // Print the trained transformer
// echo $transformer;

// // Serialize the transformer
// $serializedTransformer = serialize($transformer);

// // Unserialize the transformer
// $unserializedTransformer = unserialize($serializedTransformer);

// // Print the unserialized transformer
// echo $unserializedTransformer;
?>