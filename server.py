from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
from pydub import AudioSegment
import os

app = Flask(__name__)
CORS(app)

# Load the Whisper model
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio = AudioSegment.from_file(audio_file, format='wav')

    audio.export("audio.wav", format="wav")
    
    # Use Whisper to transcribe the audio
    result = model.transcribe("audio.wav")

    # Remove the temporary audio file
    os.remove("audio.wav")

    return jsonify({'transcription': result['text']})

if __name__ == '__main__':
    app.run(debug=True)
