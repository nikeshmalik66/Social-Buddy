// scripts.js
function populateInput(text) {
    document.getElementById('userInput').value = text;
}

function updateWordCount() {
    let userInput = document.getElementById('userInput').value;
    let words = userInput.split(/\s+/).filter(function(word) { return word.length > 0 });
    document.getElementById('wordCount').innerText = words.length + " words";
}

document.getElementById('sendButton').addEventListener('click', function() {
    let userInput = document.getElementById('userInput').value;
    if(userInput.trim() !== "") {
        let chatHistory = document.getElementById('chatHistory');
        
        // Add user's message to chat
        let userMessage = document.createElement('div');
        userMessage.innerHTML = `
            <div style="text-align:right; margin:10px; background-color: #e9f5ff; padding:10px; border-radius:10px;">
                ${userInput} <span class="user-icon"></span> 
            </div>`;
        chatHistory.appendChild(userMessage);

        // Determine the current page (Emotion, Hate Speech, or Sarcasm)
        const currentPage = window.location.pathname;
        let endpoint;
        if (currentPage === "/hatespeech") {
            endpoint = '/predict_hatespeech';
        } else if (currentPage === "/sarcasm") {
            endpoint = '/predict_sarcasm';
        } else if (currentPage === "/slang") {
            endpoint = '/detect_slang';
        }else if (currentPage === "/slang_new") {
            endpoint = '/slang_detect';
        }else {
            endpoint = '/predict';
        }

        // Make an AJAX request to the Flask server
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: userInput
            })
        })
        .then(response => response.json())
        .then(data => {
            let aiMessage = document.createElement('div');
            if (currentPage === "/hatespeech") {
                aiMessage.innerHTML = `
                    <div style="text-align:left; margin:10px; background-color: #d4e8ff; padding:10px; border-radius:10px;">
                        <span class="ai-icon"></span> ${data.label}
                    </div>`;
            } else if (currentPage === "/sarcasm") {
                aiMessage.innerHTML = `
                    <div style="text-align:left; margin:10px; background-color: #d4e8ff; padding:10px; border-radius:10px;">
                        <span class="ai-icon"></span> ${data.sarcasm_label}
                    </div>`;
            } else if (currentPage === "/slang") {
                aiMessage.innerHTML = `
                    <div style="text-align:left; margin:10px; background-color: #d4e8ff; padding:10px; border-radius:10px;">
                        <span class="ai-icon"></span> ${data.slang_label}
                    </div>`;
            }else if (currentPage === "/slang_new") {  
                console.log(data.is_slang_sentence)
                let slangMessage = data.is_slang_sentence;
                let slangWordsHtml = "";
                if (slangMessage === "Slang") {
                    slangWordsHtml += "Slang <br><p><strong>Detected Slang Words and Their Meanings:</strong></p>";
                    for (let [word, meaning] of Object.entries(data.slang_words)) {
                        slangWordsHtml += `<p><strong>${word}</strong>: ${meaning}</p>`;
                    }
                } else {
                    slangWordsHtml = "No Slang Detected";
                }
                aiMessage.innerHTML = `
                    <div style="text-align:left; margin:10px; background-color: #d4e8ff; padding:10px; border-radius:10px;">
                        <span class="ai-icon"></span> ${slangWordsHtml}
                    </div>`;
            }else{
                aiMessage.innerHTML = `
                    <div style="text-align:left; margin:10px; background-color: #d4e8ff; padding:10px; border-radius:10px;">
                        <span class="ai-icon"></span> ${data.emotion}
                    </div>`;
            }
            chatHistory.appendChild(aiMessage);
        });

        // Clear the input
        document.getElementById('userInput').value = "";

        // Hide examples and instructions
        document.querySelector('.examples').style.display = 'none';
        document.querySelector('.instructions').style.display = 'none';
        document.querySelector('.heading').style.display = 'none';

        // Auto-scroll to bottom
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});

// Function to type the text
function typeText() {
    const textElement = document.getElementById('typed-text');
    const text = textElement.innerText;
    textElement.innerText = '';

    let charIndex = 0;
    const typingSpeed = 40; // Adjust the typing speed as needed

    function type() {
        if (charIndex < text.length) {
            textElement.innerText += text.charAt(charIndex);
            charIndex++;
            setTimeout(type, typingSpeed);
        }
    }

    type();
}

// Start the animation and repeat every 6 seconds
typeText();
setInterval(typeText, 8701);

const modelTile = document.querySelector('.model-tile');
const ctaSection = modelTile.querySelector('.cta');

modelTile.addEventListener('mouseenter', function() {
    ctaSection.style.display = 'block';
});

modelTile.addEventListener('mouseleave', function() {
    ctaSection.style.display = 'none';
});
