// Your JavaScript code here

function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    if (userInput === "") {
        return;
    }

    document.getElementById("chatbox").innerHTML += `<div class="message user">${userInput}</div>`;
    document.getElementById("user-input").value = "";

    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        var chatbotResponse = data.response;
        var sentimentScore = data.sentiment_score;
        var recommendedSongs = data.recommended_songs;

        document.getElementById("chatbox").innerHTML += `<div class="message chatbot">${chatbotResponse}</div>`;
        for (var i = 0; i < recommendedSongs.length; i++) {
            var song = recommendedSongs[i];
            var songLink = `<a href="${song.url}" target="_blank">${song.name} - ${song.artist}</a>`;
            document.getElementById("chatbox").innerHTML += `<div class="message chatbot">${songLink}</div>`;
        }

        document.getElementById("chatbox").scrollTop = document.getElementById("chatbox").scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
