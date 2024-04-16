<?php

class Tokenizer {
    private $vocab;
    private $unkTokenId;
    private $padTokenId;

    public function __construct($vocab, $unkTokenId, $padTokenId) {
        $this->vocab = $vocab;
        $this->unkTokenId = $unkTokenId;
        $this->padTokenId = $padTokenId;
    }

    public function tokenize($text) {
        // Simple tokenization by splitting on spaces
        return explode(' ', $text);
    }

    public function getTokenId($token) {
        // Lookup token in vocabulary
        $tokenId = array_search($token, $this->vocab);
        if ($tokenId === false) {
            // Token not found, return unknown token ID
            return $this->unkTokenId;
        } else {
            return $tokenId;
        }
    }

    public function getPaddingTokenId() {
        return $this->padTokenId;
    }
}
