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
    with sr.Microphone() as source:
