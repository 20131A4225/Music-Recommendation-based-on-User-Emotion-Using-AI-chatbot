
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
import random
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
#loading the intents file
intents_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['input']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
#print(documents)
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
#print (len(documents), "documents")
# classes = intents
#print (len(classes), "classes", classes)
# words = all words, vocabulary
#print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

import nltk
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
import requests  # Don't forget to import the requests module
intents_file = open('intents.json', encoding='utf-8').read()
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

# Your clean_up_sentence, bag_of_words, and predict_class functions remain the same
def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score
1

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['response'])
            break
    return result

msg = list()
text = str()

def responsed(msg1):
    msg.append(msg1)
    ints = predict_class(msg1)
    res = getResponse(ints, intents)
    return res

import speech_recognition as sr
import requests


def recommend_songs(emotion):
    api_key = "6e55fb4a2653b88414684bc79b5c3691"  # Replace with your actual Last.fm API key
    url = f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={emotion}&api_key={api_key}&format=json&limit=10"
    response = requests.get(url)
    payload = response.json()
    recommended_songs = []

    try:
        tracks = payload['tracks']['track']
        if isinstance(tracks, dict):  # Check if tracks is a dictionary instead of a list
            tracks = [tracks]  # Convert dictionary to a list
        for track in tracks:
            song_name = track['name']
            artist_name = track['artist']['name']
            song_url = track['url']
            recommended_songs.append(f"{song_name} - {artist_name} ({song_url})")
    except (KeyError, IndexError):
        print("Error: Unable to extract tracks from the payload.")

    return recommended_songs


recognizer = sr.Recognizer()

print("Welcome to the Song Recommendation Chatbot!")

# Get the user's choice of input method at the beginning
choice = input("Enter 'text' for text input or 'voice' for voice input: ")
if choice.lower() not in ['text', 'voice']:
    print("Invalid choice. Exiting.")
else:
    user_inputs = []

    while True:
        if choice.lower() == 'text':
            user_input = input("User : ")
            if user_input.lower() == 'bye':
                print("Goodbye!")
                break
            user_inputs.append(user_input)
            response = responsed(user_input)
            print("Chatbot :", response)
        elif choice.lower() == 'voice':
            print("Speak something:")
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                try:
                    user_input = recognizer.recognize_google(audio)
                    print("You said:", user_input)
                    if user_input.lower() == 'exit':
                        print("Goodbye!")
                        break
                    user_inputs.append(user_input)
                    response = responsed(user_input)
                    print("Chatbot :", response)
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results; check your network connection")

    combined_text = ' '.join(user_inputs)
    sentiment_score = get_sentiment(combined_text)
    emotion = "happy" if sentiment_score > 0 else "neutral" if sentiment_score == 0 else "sad"

    print("\nConversation:")
    for idx, user_input in enumerate(user_inputs, start=1):
        print(f"User {idx}: {user_input}")

    print("\nEmotion:", emotion)
    recommended_songs = recommend_songs(emotion)

    print("\nRecommended Songs:")
    for idx, song in enumerate(recommended_songs, start=1):
        print(f"Song {idx}: {song}")


with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
