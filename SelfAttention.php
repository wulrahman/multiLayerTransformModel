<?php
// Self-Attention
class SelfAttention {
    private $weights;
    private $bias;

    private $input;

    public function __construct($inputSize) {
        $this->weights = Math::randomMatrix($inputSize, $inputSize);
        $this->bias = Math::randomVector($inputSize);
    }

    public function forward($input) {

        $this->input = $input;

        // Calculate attention scores
        $attentionScores = Math::matmul($input, $this->weights);

        // Add bias
        $attentionScores = Math::addVector($attentionScores, $this->bias);

        // Apply softmax activation
        $attentionScores = Math::softmax($attentionScores);

        // Apply attention weights to input
        $output = Math::matmul($attentionScores, $input);

        return $output;
    }

    public function backward($gradient, $learningRate) {
        // Calculate gradient of weights
        $weightsGradient = Math::matmul(Math::transpose($this->input), $gradient);
    
        // Calculate gradient of bias
        $biasGradient = Math::sumVector(Math::transpose($gradient)); // Assuming bias is a vector

        // Calculate gradient of input
        $inputGradient = Math::matmul($gradient, Math::transpose($this->weights));
    
        // Update weights and bias
        $this->weights = Math::sub($this->weights, Math::mul($weightsGradient, $learningRate));
        $this->bias = Math::subVectorValue($this->bias, $biasGradient * $learningRate);
    
        return $inputGradient;
    }
    

    public function getWeights() {
        return $this->weights;
    }

    public function setWeights($weights) {
        $this->weights = $weights;
    }

    public function getBias() {
        return $this->bias;
    }

    public function setBias($bias) {
        $this->bias = $bias;
    }
}