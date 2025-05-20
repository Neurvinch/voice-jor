import speech_recognition as sr
from transformers import pipeline
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime
import os 

# to start the speech recognition
recognizer = sr.Recognizer();

# to load the model that fine-tuned 
emotion_classifier = pipeline("text-classification",
                              model="bhadresh-savani/distilbert-base-uncased-emotion");

# the file where all journals are stored
JOURNAL_FILE = "mood_journal.txt";


# now lets cook a recording voice procedure

def record_voice():
    # opens microphone and starts listening
    with sr.Microphone() as source:
        print("Listening... Speak about your day.")
    
        # used to make ambident noise to listen
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout= 5, phrase_time_limit= 10);
        
        #this does a try ..catch thing
        # where it uses a company to transcribe audio to text
        try:
            text = recognizer.recognize_amazon(audio);
            print(f"Transcribed: {text}")
            return text;
    # if it does not go well we use unknow vlaue error and connectio timeout error
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.");
            return None;
        except sr.RequestError as e:
            print(f"Error: {e}")
            return None;

# now lets cook a procedure for emotion detection;

def detect_emotion(text):
    if not text:
        return "neutral", 0.0;

# this classfiy the emotion based on that text and returns the emotion by labelled
    result = emotion_classifier(text)[0];
    emotion = result['label']
    
    # this is a text blob which is ued to get the sentiment of the text like integers value
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return emotion, sentiment;