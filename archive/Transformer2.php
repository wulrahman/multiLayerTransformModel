<?php
class Transformer {
    private $inputSize;
    private $outputSize;

    public function __construct($inputSize, $outputSize) {
        $this->inputSize = $inputSize;
        $this->outputSize = $outputSize;
    }

    public function transform($input) {
        $output = array();
        $inputSize = count($input);
        $outputSize = $this->outputSize;
        $inputIndex = 0;
        $outputIndex = 0;
        $inputValue = 0;
        $outputValue = 0;
        $outputBit = 0;
        $outputMask = 0;
        $outputMaskSize = 0;
        $outputMaskValue = 0;

        while ($inputIndex < $inputSize) {
            $inputValue = $input[$inputIndex];
            $outputMask = 0;
            $outputMaskSize = 0;
            $outputMaskValue = 0;

            while ($outputMaskSize < $outputSize) {
                $outputMaskValue = $outputMaskValue << 1;
                $outputMaskValue = $outputMaskValue | ($inputValue & 1);
                $outputMask = $outputMask << 1;
                $outputMask = $outputMask | 1;
                $inputValue = $inputValue >> 1;
                $outputMaskSize++;
            }

            $outputValue = $outputMaskValue;
            $outputBit = 0;

            while ($outputBit < $outputSize) {
                $output[$outputIndex] = $outputValue & 1;
                $outputValue = $outputValue >> 1;
                $outputBit++;
                $outputIndex++;
            }

            $inputIndex++;
        }

        return $output;
    }

    public function inverseTransform($output) {
        $input = array();
        $inputSize = $this->inputSize;
        $outputSize = count($output);
        $inputIndex = 0;
        $outputIndex = 0;
        $inputValue = 0;
        $outputValue = 0;
        $inputBit = 0;
        $inputMask = 0;
        $inputMaskSize = 0;
        $inputMaskValue = 0;

        while ($outputIndex < $outputSize) {
            $outputValue = 0;
            $inputMask = 0;
            $inputMaskSize = 0;
            $inputMaskValue = 0;

            while ($inputMaskSize < $inputSize) {
                $inputMaskValue = $inputMaskValue << 1;
                $inputMaskValue = $inputMaskValue | ($output[$outputIndex] & 1);
                $inputMask = $inputMask << 1;
                $inputMask = $inputMask | 1;
                $output[$outputIndex] = $output[$outputIndex] >> 1;
                $inputMaskSize++;
            }

            $inputValue = $inputMaskValue;
            $inputBit = 0;

            while ($inputBit < $inputSize) {
                $input[$inputIndex] = $inputValue & 1;
                $inputValue = $inputValue >> 1;
                $inputBit++;
                $inputIndex++;
            }

            $outputIndex++;
        }

        return $input;
    }

    public function test() {
        $input = array(1, 0, 1, 0, 1, 0, 1, 0);
        $output = $this->transform($input);
        $input = $this->inverseTransform($output);
        $result = implode("", $input);
        return $result;
    }
}

$transformer = new Transformer(4, 8);
echo $transformer->test();