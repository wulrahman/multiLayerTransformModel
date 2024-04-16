

document.addEventListener("DOMContentLoaded", function() {
    const startWordInput = document.getElementById("startWord");
    const suggestionsContainer = document.getElementById("suggestions");
    const generatedSentenceOutput = document.getElementById("generatedSentence");

    const languageModel = new LanguageModel(10);

    const sentences = [
        "Hello! Thank you for reaching out to us regarding your account access issue. We apologize for any inconvenience you're experiencing. Our team is here to assist you.",
        "To better assist you, could you please provide us with your account username or email address? Once we have this information, we'll investigate further to resolve the issue.",
        "Feel free to reply to this message or contact our support team at [support email/phone number]. We're committed to resolving this matter promptly and ensuring your satisfaction.",
        "Thank you for your patience and cooperation. We look forward to assisting you further.",
        "Hello! Thank you for reaching out to us. How may I assist you today?",
        "To better assist you, could you please provide more details about the issue you're facing?",
        "You can reach our support team via email at {{support_email}} or by phone at {{mobile_number}}.",
        "We appreciate your patience and understanding. Please don't hesitate to reach out if you need further assistance.",
        "Greetings! Thank you for contacting us. How can I help you with your inquiry?",
        "In order to assist you effectively, could you please provide any error messages or specific details related to the issue?",
        "For immediate assistance, please contact our support team at {{support_email}} or call us at {{mobile_number}}.",
        "Thank you for reaching out to us. We're dedicated to resolving your issue promptly and ensuring your satisfaction.",
        "Hi there! Thanks for getting in touch. What can I do to assist you today?",
        "To better understand your concern, could you please provide information such as your account number or the time of the incident?",
        "If you require further assistance, please don't hesitate to contact our support team at {{support_email}} or by phone at {{mobile_number}}.",
        "Thank you for allowing us the opportunity to assist you. We're committed to resolving your issue and ensuring a positive experience.",
        "Hello! We're sorry to hear that you're not satisfied with your purchase. How can we assist you with a refund or return?",
        "To initiate a refund or return, please provide us with your order number and the reason for the return. Once we receive this information, we'll guide you through the process.",
        "For further assistance with your refund or return, please contact our support team at {{refund_email}} or call us at {{mobile_number}}.",
        "We understand the importance of resolving this matter promptly. Thank you for reaching out to us, and we'll do our best to assist you.",
        "Hello! We're here to help with your return request. How can we assist you in returning the item?",
        "To proceed with the return, please provide us with your order number and details about the item you wish to return. We'll then provide you with instructions on how to return it.",
        "If you have any questions or need assistance with your return, please contact our support team at {{return_email}}  or by phone at {{mobile_number}}.",
        "Thank you for contacting us about your return. We appreciate your patience and understanding as we work to resolve this matter to your satisfaction."
    ];
    

    languageModel.train(sentences);

    startWordInput.addEventListener("input", function(event) {
        event.preventDefault();
        const startWord = startWordInput.value.trim();
        if (startWord !== "") {
            const generatedSentence = languageModel.generate(startWord, 50); // Change 50 to desired number of words
            generatedSentenceOutput.textContent = "Generated Sentence: " + generatedSentence;
        } else {
            generatedSentenceOutput.textContent = "Please enter a start word.";
        }
    });

    startWordInput.addEventListener("blur", function() {
        setTimeout(clearSuggestions, 200);
    });

    suggestionsContainer.addEventListener("click", function(event) {
        if (event.target.classList.contains("suggestion")) {
            startWordInput.value = event.target.textContent;
            clearSuggestions();
        }
    });

    function displaySuggestions(suggestions) {
        suggestionsContainer.innerHTML = "";
        suggestions.forEach(function(suggestion) {
            const suggestionElement = document.createElement("div");
            suggestionElement.classList.add("suggestion");
            suggestionElement.textContent = suggestion;
            suggestionsContainer.appendChild(suggestionElement);
        });
    }

    function clearSuggestions() {
        suggestionsContainer.innerHTML = "";
    }
});
