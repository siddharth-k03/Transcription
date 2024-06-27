import time
import os
from google.cloud import speech_v1p1beta1 as speech
import boto3
import whisper
from jiwer import wer, cer
from dotenv import load_dotenv
import requests
from pydub import AudioSegment

# Set the path to your Google Cloud credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials_file"

# Load variables from .env file
load_dotenv()

# Load your audio file
AUDIO_FILE_PATH = "path/to/audio_file"

# Function to split audio into chunks
def split_audio(audio_file_path, chunk_length_ms=60000):
    audio = AudioSegment.from_file(audio_file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

# Google Cloud Speech-to-Text
def transcribe_google(audio_chunks):
    client = speech.SpeechClient()
    transcripts = []

    for chunk in audio_chunks:
        audio_data = chunk.raw_data
        sample_rate_hertz = chunk.frame_rate

        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hertz,
            language_code="en-US",
        )

        response = client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        transcripts.append(transcript)

    return " ".join(transcripts)

# AWS Transcribe
def transcribe_aws(audio_file_path):
    transcribe = boto3.client('transcribe')
    job_name = "job_name"
    job_uri = f"s3://{os.getenv('AWS_BUCKET')}/{os.path.basename(audio_file_path)}"


    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode='en-US'
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        response = requests.get(transcript_uri)
        transcript = response.json()['results']['transcripts'][0]['transcript']
        return transcript

#Whisper
def transcribe_whisper_base(audio_file_path):
    model = whisper.load_model("base.en")
    transcript = model.transcribe(audio_file_path)
    return transcript["text"]

def transcribe_whisper_large(audio_file_path):
    model = whisper.load_model("large")
    transcript = model.transcribe(audio_file_path)
    return transcript["text"]

def transcribe_whisper_large_v3(audio_file_path):
    model = whisper.load_model("large-v3")
    transcript = model.transcribe(audio_file_path)
    return transcript["text"]

# Whisper Large V3 with GROQ API
def transcribe_api(audio_file_path):
    url = "http://127.0.0.1:8000/transcribe/"
    with open(audio_file_path, "rb") as audio_file:
        response = requests.post(url, files={"file": audio_file})
    
    if response.status_code == 200:
        return response.json()["transcription"]
    else:
        raise Exception(f"API Error: {response.json()}")

# Benchmarking function
def benchmark_transcription(method, transcribe_func, audio_file_path, reference_text, **kwargs):
    start_time = time.time()
    audio_chunks = split_audio(audio_file_path)
    transcript = transcribe_func(audio_chunks, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    word_error_rate = wer(reference_text, transcript)
    character_error_rate = cer(reference_text, transcript)
    return {
        "method": method,
        "duration": duration,
        "word_error_rate": word_error_rate,
        "character_error_rate": character_error_rate
    }

# Main function to run benchmarks
def run_benchmarks(audio_file_path, reference_text):
    results = []
    #Whisper
    results.append(benchmark_transcription("Whisper Base", transcribe_whisper_base, audio_file_path, reference_text))
    results.append(benchmark_transcription("Whisper Large V1", transcribe_whisper_large, audio_file_path, reference_text))
    results.append(benchmark_transcription("Whisper Large V3", transcribe_whisper_large_v3, audio_file_path, reference_text))

    #Google Cloud
    results.append(benchmark_transcription("Google Cloud Speech-to-Text", transcribe_google, audio_file_path, reference_text))

    '''#AWS
    results.append(benchmark_transcription("AWS Transcribe", transcribe_aws, audio_file_path, reference_text))

    #GROQ API
    results.append(benchmark_transcription("Whisper Large V3 with GROQ API", transcribe_api, audio_file_path, reference_text))'''

    return results

# Example usage
reference_text = "reference text here."
results = run_benchmarks(AUDIO_FILE_PATH, reference_text)
for result in results:
    print(f"Method: {result['method']}")
    print(f"Duration: {result['duration']} seconds")
    print(f"Word Error Rate: {result['word_error_rate']}")
    print(f"Character Error Rate: {result['character_error_rate']}")
