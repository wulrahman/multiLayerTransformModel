<?php
// Encoder
// EncoderDecoder class
class EncoderDecoder {
    private $encoder;
    private $decoder;
    public $selfAttention;

    public function __construct($inputSize, $outputSize, $hiddenSize) {
        $this->encoder = new Transformer($inputSize, $hiddenSize);
        $this->decoder = new Transformer($hiddenSize, $outputSize);
        $this->selfAttention = new SelfAttention($hiddenSize);
    }

    public function train($input, $target, $learningRate, $epochs) {
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            // Forward pass through encoder
            $encodedOutput = $this->encoder->forward($input);

            // Apply self-attention
            $attentionOutput = $this->selfAttention->forward($encodedOutput);

            // Decoder input is the attention output
            $decoderInput = $attentionOutput;

            // Forward pass through decoder
            $decodedOutput = $this->decoder->forward($decoderInput);

            // Calculate loss
            $loss = $this->calculateLoss($decodedOutput, $target);

            // Backpropagation
            $this->backward($loss, $learningRate);
        }
    }

    public function forward($input) {
        // Forward pass through encoder
        $encodedOutput = $this->encoder->forward($input);

        // Apply self-attention
        $attentionOutput = $this->selfAttention->forward($encodedOutput);

        // Decoder input is the attention output
        $decoderInput = $attentionOutput;

        // Forward pass through decoder
        $decodedOutput = $this->decoder->forward($decoderInput);

        return $decodedOutput;
    }

    public function calculateLoss($output, $target) {
        return Math::sub($output, $target);
    }

    public function backward($loss, $learningRate) {
        // Backward pass through decoder
        $gradient = Math::mul($loss, $learningRate);

        $this->decoder->backward($gradient);

        // Backward pass through self-attention
        $gradient = Math::mul($gradient, $learningRate);
        $this->selfAttention->backward($gradient);

        // Backward pass through encoder
        $gradient = Math::mul($gradient, $learningRate);
        $this->encoder->backward($gradient);

        return $gradient;
    }


    public function encode($input) {
        return $this->encoder->forward($input);
    }

    public function decode($input) {
        return $this->decoder->forward($input);
    }
}

// // Example usage
// $inputSize = 2;
// $outputSize = 2;
// $hiddenSize = 2;

// // Create an instance of EncoderDecoder
// $encoderDecoder = new EncoderDecoder($inputSize, $outputSize, $hiddenSize);

// // Define input and target sequences for training
// $input = [[5, 6], [7, 8]]; // Placeholder for input sequence
// $target = [[9, 10], [11, 12]]; // Placeholder for target sequence

// // Train the encoder-decoder
// $learningRate = 0.01;
// $epochs = 10000;
// $encoderDecoder->train($input, $target, $learningRate, $epochs);

// // Define input for encoding
// $encodeInput = [[1, 2], [3, 4]];

// // Encode the input
// $encodedOutput = $encoderDecoder->encode($encodeInput);

// // Decode the encoded output
// $decodedOutput = $encoderDecoder->decode($encodedOutput);

// // Print the encoded and decoded outputs
// echo "Encoded Output: " . json_encode($encodedOutput) . "\n";
// echo "Decoded Output: " . json_encode($decodedOutput) . "\n";
