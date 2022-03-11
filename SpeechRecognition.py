from gtts import gTTS
import speech_recognition as sr
import time
from pygame import mixer  # Load the popular external library
import pocketsphinx

mixer.init()
def speek(text, lang, filename="voice1.mp3"):
    # tts = gTTS(text=text, lang=lang)
    # tts.save(filename)
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():  # wait for music to finish playing
        time.sleep(1)

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak")
        audio = r.listen(source, phrase_time_limit=4)
        said = ""
        try:
            said = r.recognize_sphinx(audio)
        except Exception as e:
            print("Error: ", e)
    return said


speek("berlin ground CN397 request take off", lang="en", filename="data/Audio/TakeOffRequest.mp3")
text = get_audio()
print("Said: ", text)
if "ok" in text:
    speek("Got It Take Off", lang="en", filename="data/Audio/ApprovedTakeOff.mp3")


speek("Request tracking mode", lang="en", filename="data/Audio/TrackingRequest.mp3")

text = get_audio()
print("Said: ", text)
if "ok" in text:
    speek("Tracking Mode Start be careful", lang="en", filename="data/Audio/StartTracking.mp3")

text = get_audio()
print("Said: ", text)
if "lan" in text or "sto" in text:
    speek("Stop Tracking", lang="en", filename="data/Audio/StopTracking.mp3")

text = get_audio()
print("Said: ", text)
if text is not "":
    speek("Stop Tracking", lang="en", filename="data/Audio/StopTracking.mp3")

# speek("Hello ")