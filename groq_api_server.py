import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import aiohttp
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Function to convert the sample rate to 16000 Hz if needed
def convert_sample_rate(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    return audio

# Asynchronous function to transcribe audio using Groq API
async def transcribe_audio(audio_bytes):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    api_key = os.getenv('GROQ_API_KEY')

    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    form_data = aiohttp.FormData()
    form_data.add_field('model', 'whisper-large-v3')
    form_data.add_field('file', audio_bytes, filename='audio.mp4', content_type='audio/mp4')
    form_data.add_field('response_format', 'verbose_json')

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=form_data) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=await response.text())
            return await response.json()


# Endpoint to handle audio file uploads and transcription
@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        converted_audio = convert_sample_rate(audio_bytes)

        # Save converted audio to a supported format (e.g., 'mp4')
        buffer = io.BytesIO()
        converted_audio.export(buffer, format='mp4')
        buffer.seek(0)

        transcription = await transcribe_audio(buffer)
        transcribed_text = transcription.get('text', 'Transcription failed')
        segments = transcription.get('segments', [])

        return JSONResponse(content={
            "transcription": transcribed_text,
            "segments": segments
        },
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
