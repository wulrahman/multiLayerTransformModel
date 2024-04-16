<?php
class BackPropagation {
    private $learningRate;
    private $weights;
    private $bias;

    public function __construct($learningRate) {
        $this->learningRate = $learningRate;
        $this->weights = [];
        $this->bias = 0;
    }

    public function train($inputs, $targets, $epochs) {
        // Initialize weights randomly
        $this->weights = array_fill(0, count($inputs[0]), 0);

        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            $predictions = $this->predict($inputs);
            $errors = [];

            // Calculate errors
            for ($i = 0; $i < count($targets); $i++) {
                $errors[$i] = $targets[$i] - $predictions[$i];
            }

            // Update weights and bias
            for ($i = 0; $i < count($this->weights); $i++) {
                $gradient = 0;

                for ($j = 0; $j < count($inputs); $j++) {
                    $gradient += $inputs[$j][$i] * $errors[$j];
                }

                $this->weights[$i] += $this->learningRate * $gradient;
            }

            $this->bias += $this->learningRate * array_sum($errors);
        }
    }

    public function predict($inputs) {
        $predictions = [];

        foreach ($inputs as $input) {
            $prediction = 0;

            for ($i = 0; $i < count($input); $i++) {
                $prediction += $input[$i] * $this->weights[$i];
            }

            $prediction += $this->bias;
            $predictions[] = $prediction;
        }

        return $predictions;
    }

    public function calculateLoss($inputs, $targets) {
        $predictions = $this->predict($inputs);
        $loss = 0;

        for ($i = 0; $i < count($targets); $i++) {
            $loss += pow($targets[$i] - $predictions[$i], 2);
        }

        return $loss / count($targets);
    }
}

// Example usage
$backPropagation = new BackPropagation(0.01);
$inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
$targets = [0, 1, 1, 0];
$epochs = 1000;

$backPropagation->train($inputs, $targets, $epochs);
$loss = $backPropagation->calculateLoss($inputs, $targets);

echo "Final loss: " . $loss; // Output: Final loss: 0.000000