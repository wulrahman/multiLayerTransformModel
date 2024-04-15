<?php
require_once __DIR__ . '/Math.php';

class Transformer {
    private $weights;
    private $bias;

    private $activations;

    public function __construct($inputSize, $outputSize) {
        // Initialize weights and bias based on input and output sizes
        $this->initializeWeights($inputSize, $outputSize);
        $this->initializeBias($outputSize);
    }

    private function initializeWeights($inputSize, $outputSize) {
        $scale = sqrt(6 / ($inputSize + $outputSize));
        $this->weights = [];
        for ($i = 0; $i < $outputSize; $i++) {
            $this->weights[] = array_map(function() use ($scale) {
                return 1 * $scale;
            }, range(0, $inputSize - 1));
        }
    }

    private function initializeBias($outputSize) {
        $this->bias = array_fill(0, $outputSize, 0.0);
    }
    
    
    public function forward($input) {
        // Matrix multiplication
        $output = Math::matmul($this->weights, $input);

        // Add bias
        $output = Math::addMatVector($output, $this->bias);

        // Apply activation function (e.g., ReLU)
        $this->activations = Math::relu($output);

        return $this->activations;
    }

    public function train($inputs, $targets, $learningRate, $epochs) {
        $numInputs = count($inputs);
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            for ($i = 0; $i < $numInputs; $i++) {
                $input = $inputs[$i];
                $target = $targets[$i];
                // Forward pass
                $output = $this->forward($input);
        
                // Calculate loss
                $loss = Math::sub($output, $target);
        
                // Backward pass
                $gradient = Math::mul($loss, $learningRate);
                $this->backward($gradient, $learningRate);
            }
        }
    }
    
    public function backward($gradient, $learningRate) {
        // Calculate loss gradient
        $lossGradient = Math::matmul(Math::transpose($this->weights), $gradient);

        // If your activation function is ReLU, compute its derivative
        $reluDerivative = Math::reluDerivative($this->activations); // You need to provide activations here

        // Element-wise multiplication with ReLU derivative
        $lossGradient = Math::mulMatrix($lossGradient, $reluDerivative);

        // Update weights
        $this->weights = Math::sub($this->weights, Math::mul($lossGradient, $learningRate));

        // Update bias
        $this->bias = Math::subVectorValue($this->bias, Math::sumVector(Math::sum($gradient)) * $learningRate);

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

    
    public function initializeWeightsFromParams($params) {
        $this->weights = $params[0];
    }

    public function initializeBiasFromParams($params) {
        $this->bias = $params[1];
    }

    public function initializeWeightsFromSerialized($weights) {
        $this->weights = $weights;
    }

    public function initializeBiasFromSerialized($bias) {
        $this->bias = $bias;
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

// Example usage
$inputSize = 2; // Example input size
$outputSize = 2; // Example output size

$transformer = new Transformer($inputSize, $outputSize);

$inputs = [[[5, 6], [7, 8]], [[1, 2], [3, 4]]];
$targets = [[[7, 8], [9, 10]], [[3, 4], [5, 6]]];

$learningRate = 0.01;
$epochs = 1000;

$transformer->train($inputs, $targets, $learningRate, $epochs);

print_r($transformer->forward([[5, 6], [7, 8]]));

echo $transformer;

?>
