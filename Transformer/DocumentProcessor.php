<?php

class DocumentProcessor {
    private $multiLayerEncoderDecoder;
    private $vocab;
    private $unkTokenId;
    private $inputSize;
    private $outputSize;
    
    public function __construct($inputSize, $outputSize, $hiddenSize, $numLayers) {
        $this->multiLayerEncoderDecoder = new MultiLayerEncoderDecoder($inputSize, $outputSize, $hiddenSize, $numLayers);
        $this->vocab = [0 => '<PAD>', 1 => '<UNK>'];
        $this->unkTokenId = null;
        $this->inputSize = $inputSize;
        $this->outputSize = $outputSize;
    }
    
    public function train($inputSentences, $targetSentences, $learningRate, $epochs) {
        $input = [];
        $target = [];
        
        $inputSentences = array_slice($inputSentences, 0, $this->inputSize);
        foreach ($inputSentences as $sentence) {
            $ids = $this->sentenceToIds(explode(' ', $sentence));
            $input[] = $this->padSequence($ids, 0, $this->inputSize);
        }
        
        $input = $this->padSequence($input, array_fill(0, $this->inputSize, 0), $this->inputSize);

        $targetSentences = array_slice($targetSentences, 0, $this->outputSize);
        foreach ($targetSentences as $sentence) {
            $ids = $this->sentenceToIds(explode(' ', $sentence));
            $target[] = $this->padSequence($ids, 0, $this->inputSize);
        }

        $target = $this->padSequence($target, array_fill(0, $this->inputSize, 0), $this->inputSize);
        
        $this->multiLayerEncoderDecoder->train($input, $target, $learningRate, $epochs);
    }
    
    public function encode($inputSentences) {
        $input = [];
        

        $inputSentences = array_slice($inputSentences, 0, $this->inputSize);
        foreach ($inputSentences as $sentence) {
            $ids = $this->sentenceToIds(explode(' ', $sentence));
            $input[] = $this->padSequence($ids, 0, $this->inputSize);
        }
        $input = $this->padSequence($input, array_fill(0, $this->inputSize, 0), $this->inputSize);

        $encodedOutput = $this->multiLayerEncoderDecoder->encode($input);
        
        return $encodedOutput;
    }
    
    public function decode($encodedOutput) {
        $decodedOutput = $this->multiLayerEncoderDecoder->decode($encodedOutput);
        $outputSentences = [];
        foreach ($decodedOutput as $output) {
            $sentence = [];
            foreach ($output as $id) {
                $sentence[] = $this->vocab[$id];
            }
            $outputSentences[] = implode(' ', $sentence);
        }

        
        return $outputSentences;
    }
    
    private function sentenceToIds($sentence) {
        $ids = [];
        foreach ($sentence as $word) {
            $id = array_search($word, $this->vocab);
            if ($id === false) {
                $id = count($this->vocab);
                $this->vocab[] = $word;
                if ($this->unkTokenId === null) {
                    $this->unkTokenId = $id;
                }
            }
            $ids[] = $id;
        }
        return $ids;
    }
    
    private function padSequence($sequence, $padTokenId, $maxSequenceLength) {
        $length = count($sequence);
        if ($length < $maxSequenceLength) {
            $padding = array_fill(0, $maxSequenceLength - $length, $padTokenId);
            return array_merge($sequence, $padding);
        } else {
            return array_slice($sequence, 0, $maxSequenceLength);
        }
    }
}
