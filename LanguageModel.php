
<?php

class LanguageModel {
    private $vocabulary;
    private $nGrams;
    private $n;

    public function __construct($n) {
        $this->vocabulary = [];
        $this->nGrams = [];
        $this->n = $n;
    }

    public function train($sentences) {
        foreach ($sentences as $sentence) {
            $words = explode(" ", $sentence);
            $n = count($words);
            for ($i = 0; $i < $n - $this->n + 1; $i++) {
                $nGram = implode(" ", array_slice($words, $i, $this->n));
                if (!isset($this->nGrams[$nGram])) {
                    $this->nGrams[$nGram] = 0;
                }

                $this->nGrams[$nGram]++;
            }
            foreach ($words as $word) {
                if (!in_array($word, $this->vocabulary)) {
                    $this->vocabulary[] = $word;
                }
            }
        }
    }

    public function generate($startWord, $numWords) {
        $generatedSentence = $startWord;
        $currentWord = $startWord;
        for ($i = 1; $i < $numWords; $i++) {
            $nextWord = $this->getNextWord($currentWord);
            if ($nextWord === null) {
                break;
            }
            $generatedSentence .= " " . $nextWord;
            $currentWord = $nextWord;
        }
        return $generatedSentence;
    }

    private function getNextWord($currentWord) {
        $possibleNextWords = [];
        foreach ($this->nGrams as $nGram => $count) {
            $words = explode(" ", $nGram);
            $n = count($words);

            for($i = 0; $i < $n; $i++) {
                if ($words[$i] === $currentWord) {
                    if(isset($words[$i+1])) {
                        $possibleNextWords[] = $words[$i+1];
                    }
                }
            }
        }
        if (count($possibleNextWords) === 0) {
            return null;
        }
        $nextWord = $possibleNextWords[array_rand($possibleNextWords)];
        return $nextWord;
    }
}

// Example usage
$n = 2;
$languageModel = new LanguageModel($n);

$sentences = [
    "message" => "Hello! Thank you for reaching out to us regarding your account access issue. We apologize for any inconvenience you're experiencing. Our team is here to assist you.",
    "steps" => "To better assist you, could you please provide us with your account username or email address? Once we have this information, we'll investigate further to resolve the issue.",
    "contact_info" => "Feel free to reply to this message or contact our support team at [support email/phone number]. We're committed to resolving this matter promptly and ensuring your satisfaction.",
    "closing" => "Thank you for your patience and cooperation. We look forward to assisting you further.",
    "message_1" => "Hello! Thank you for reaching out to us. How may I assist you today?",
    "steps_1" => "To better assist you, could you please provide more details about the issue you're facing?",
    "contact_info_1" => "You can reach our support team via email at support@example.com or by phone at 1-800-123-4567.",
    "closing_1" => "We appreciate your patience and understanding. Please don't hesitate to reach out if you need further assistance.",
    "message_2" => "Greetings! Thank you for contacting us. How can I help you with your inquiry?",
    "steps_2" => "In order to assist you effectively, could you please provide any error messages or specific details related to the issue?",
    "contact_info_2" => "For immediate assistance, please contact our support team at support@example.com or call us at 1-800-555-1234.",
    "closing_2" => "Thank you for reaching out to us. We're dedicated to resolving your issue promptly and ensuring your satisfaction.",
    "message_3" => "Hi there! Thanks for getting in touch. What can I do to assist you today?",
    "steps_3" => "To better understand your concern, could you please provide information such as your account number or the time of the incident?",
    "contact_info_3" => "If you require further assistance, please don't hesitate to contact our support team at support@example.com or by phone at 1-800-987-6543.",
    "closing_3" => "Thank you for allowing us the opportunity to assist you. We're committed to resolving your issue and ensuring a positive experience.",
    "message_refund" => "Hello! We're sorry to hear that you're not satisfied with your purchase. How can we assist you with a refund or return?",
    "steps_refund" => "To initiate a refund or return, please provide us with your order number and the reason for the return. Once we receive this information, we'll guide you through the process.",
    "contact_info_refund" => "For further assistance with your refund or return, please contact our support team at refunds@example.com or call us at 1-800-555-6789.",
    "closing_refund" => "We understand the importance of resolving this matter promptly. Thank you for reaching out to us, and we'll do our best to assist you.",
    "message_return" => "Hello! We're here to help with your return request. How can we assist you in returning the item?",
    "steps_return" => "To proceed with the return, please provide us with your order number and details about the item you wish to return. We'll then provide you with instructions on how to return it.",
    "contact_info_return" => "If you have any questions or need assistance with your return, please contact our support team at returns@example.com or by phone at 1-800-123-4567.",
    "closing_return" => "Thank you for contacting us about your return. We appreciate your patience and understanding as we work to resolve this matter to your satisfaction."
];
    



$languageModel->train($sentences);

$startWord = "refund";
$numWords = 10;
$generatedSentence = $languageModel->generate($startWord, $numWords);

echo "Generated Sentence: " . $generatedSentence;