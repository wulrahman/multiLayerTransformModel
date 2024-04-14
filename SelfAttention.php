<?php
// Self-Attention
class SelfAttention {
    private $weights;
    private $bias;

    public function __construct($inputSize) {
        $this->weights = Math::randomMatrix($inputSize, $inputSize);
        $this->bias = Math::randomVector($inputSize);
    }

    public function forward($input) {
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

    public function backward($gradient) {
        // Calculate gradient of weights
        $weightsGradient = Math::matmul($gradient, Math::transpose($this->weights));

        // Calculate gradient of bias
        $biasGradient = $gradient;

        // Update weights and bias
        $this->weights = Math::sub($this->weights, $weightsGradient);
        $this->bias = Math::subVectorValue($this->bias, array_sum(array_map('array_sum', $gradient)));
        return $weightsGradient;
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