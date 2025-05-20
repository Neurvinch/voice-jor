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
            text = recognizer.recognize_google(audio);
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

# now lets cook a procedure to save the journal and a  file like log entry;

def log_entry(text, emotion , sentiment):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S");
    entry = f"[{timestamp}] Emotion: {emotion}, Sentiment: {sentiment:.2f}, Text: {text}\n";

    with open(JOURNAL_FILE, "a") as f:
        f.write(entry);
    print(f"Logged: {entry.strip()}");

def provide_suggestions(emotion,sentiment):

    suggestions = {
         "sadness": "Try listening to uplifting music or talking to a friend.",
        "anger": "Take deep breaths or try meditation.",
        "fear": "Practice grounding techniques or write down your worries.",
        "joy": "Keep spreading positivity or try something creative!",
        "love": "Share your positivity or write a gratitude list.",
        "surprise": "Reflect on what's exciting in your life!"
    }

    suggestion = suggestions.get(emotion,"Take a moment to reflect on your feelings.")

    if sentiment < -0.2:
        suggestion += " Consider talking to someone about how you feel."

    elif sentiment > 0.2:
        suggestion += " Keep up the positive vibes!"

    print(f"Suggestion: {suggestion}");    
            

def plot_emotions():
    entries = [] 
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE,"r") as f:
            entries = [line.strip() for line in f if line.strip()]           

    if not entries:
        print("No journal entries found.");
        return;  

    timestamps , emotions, sentiments = [], [], [];

    emotion_counts = {
        "sadness": 0,
        "anger": 0,
        "fear": 0,
        "joy": 0,
        "love": 0,
        "surprise": 0
    }

    for entry in entries:
        try:
            timestamp = entry.split("]")[0][1:]
            emotion = entry.split("Emotion: ")[1].split(",")[0]
            sentiment = float(entry.split("Sentiment: ")[1].split(",")[0])
            emotions.append(emotion)
            sentiments.append(sentiment)
            timestamps.append(timestamp)

            emotion_counts[emotion] += 1
        except:
            continue

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.bar(emotion_counts.key(), emotion_counts.values(), color=[...])

        plt.title("Emotion Distribution")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        plt.plot(timestamps, sentiments, marker='o', color='#06D6A0')

        plt.title("Sentiment Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Sentiment Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    while True:
        print("\nAI Voice Mood Journal\n1. Record entry\n2. View entries\n3. Plot emotions\n4. Exit")

        choice = input("choose an option(1-4): ")

        if choice == "1":
            text = record_voice();

            if text :
                emotion , sentimnet = detect_emotion(text);

                log_entry(text, emotion, sentimnet);

                provide_suggestions(emotion, sentimnet);

        elif choice == "2":

            if os.path.exists(JOURNAL_FILE):

                with open(JOURNAL_FILE, "r") as f:
                    print("\nEntries:")
                    print(f.read());
            else:
                print("No journal entries found.");
            
        elif choice == "3":
            plot_emotions();

        elif choice == "4":
            print("Exiting...");
            break;

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main();
