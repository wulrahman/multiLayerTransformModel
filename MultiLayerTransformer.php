<?php
// MultiLayerTransformer class
class MultiLayerTransformer {
    private $layers;

    public function __construct($inputSize, $outputSize, $numLayers) {
        $this->layers = [];
        for ($i = 0; $i < $numLayers; $i++) {
            $this->layers[] = new Transformer($inputSize, $outputSize);
        }
    }

    public function forward($input) {
        $output = $input;
        foreach ($this->layers as $layer) {
            $output = $layer->forward($output);
        }
        return $output;
    }

    public function backward($gradient) {
        foreach (array_reverse($this->layers) as $layer) {
            $gradient = $layer->backward($gradient);
        }
        return $gradient;
    }
}
