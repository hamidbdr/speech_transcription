import librosa    
from transformers import pipeline

data, _ = librosa.load('sound_sample.wav', 
                       sr=16000)

speech_recognizer = pipeline("automatic-speech-recognition",
                      model="facebook/wav2vec2-base-960h")

#speech_recognizer = pipeline(
#                "automatic-speech-recognition",
#                "modeldir")

#speech_recognizer = pipeline("automatic-speech-recognition",
#                              "modeldir")

result = speech_recognizer(data)


print("")
print("")
print("-"*40)
print(result)
print("-"*40)
print("")
print("")