
from flask import Flask, render_template, request, jsonify
import pickle
import json
from Music_Recommendation_based_on_user_emotion_using_AI_chatbot import predict_class, getResponse, get_sentiment, recommend_songs

app = Flask(__name__)
app.static_folder = 'static'

# Load the trained model using pickle
with open('trained_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.get_json()
    user_input = data['user_input']

    # Load the intents data from the file
    with open('intents.json', encoding='utf-8') as intents_file:
        intents = json.load(intents_file)

    # Use the loaded model to predict
    ints = predict_class(user_input)
    chatbot_response = getResponse(ints, intents)

    # Perform sentiment analysis
    # combined_text = user_input + ' ' + chatbot_response
    # sentiment_score = get_sentiment(combined_text)

    # Recommend songs based on sentiment score
    # recommended_songs = recommend_songs(sentiment_score)

    combined_text = user_input+' '+chatbot_response  # You can also include chatbot_response
    sentiment_score = get_sentiment(combined_text)
    emotion = "happy" if sentiment_score > 0 else "neutral" if sentiment_score == 0 else "sad"
    recommended_songs = recommend_songs(emotion)
    
    # combined_text = user_input + ' ' + chatbot_response
    # sentiment_score = get_sentiment(combined_text)

    # # Recommend songs based on sentiment score
    # recommended_songs = recommend_songs(sentiment_score)

    # # Determine the user's emotion
    # emotion = "happy" if sentiment_score > 0 else "neutral" if sentiment_score == 0 else "sad"

    
    return jsonify({
        "response": chatbot_response,
        "sentiment_score": sentiment_score,
        "recommended_songs": recommended_songs,
        "emotion":emotion
    })

@app.route('/getRecommendedSongs', methods=['POST'])

def get_recommended_songs():
    data = request.get_json()
    user_input = data['user_input']

    sentiment_score = get_sentiment(user_input)
    recommended_songs = recommend_songs(sentiment_score)

    return jsonify({'recommended_songs': recommended_songs})

def generate_recommended_songs(user_input, sentiment_score):
    if user_input.lower() == 'bye':
        recommended_songs = recommend_songs(sentiment_score)  # Replace this with your actual recommendation logic
        return recommended_songs
    else:
        return []

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
