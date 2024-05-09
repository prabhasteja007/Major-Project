import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import assemblyai as aai
from assemblyai import Transcriber
from groq import Groq
from api import grog_api
from api import api_speech, MONGO_URI, DB_NAME, COLLECTION_NAME
import os
import shutil
import uuid
from typing import Any, List
from pydantic import BaseModel
from pydub import AudioSegment
from pymongo import MongoClient



client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

app = FastAPI()

aai.settings.api_key = api_speech


# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

def store_question_answer(question, answer):
    try:
        # Insert question-answer pair into MongoDB collection
        collection.insert_one({"question": question, "answer": answer})
        print("Question-answer pair inserted into MongoDB successfully.")
    except Exception as e:
        print(f"Error storing question-answer pair in MongoDB: {e}")

def retrieve_from_mongodb() -> List[dict]:
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    documents = list(collection.find({}, {"_id": 0}))
    client.close()
    return documents

class TextRequest(BaseModel):
    text: str

@app.post("/upload/")
async def upload_file(audio_file: UploadFile = File(...)):
    try:

        filename = str(uuid.uuid4()) + ".wav"
        file_path = os.path.join("uploads", filename)
        with open(file_path, "wb") as file_object:
            shutil.copyfileobj(audio_file.file, file_object)
        if os.path.getsize(file_path) == 0:
            raise ValueError("Uploaded audio file is empty")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)
        if not transcript.text:
            raise ValueError("Transcript text is null or empty")
        transcript_text1 = transcript.text
        transcript_text = transcript_text1 + "make it sound like a conversation."
        client = Groq(api_key=grog_api)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": transcript_text,
                }
            ],
            model="llama3-70b-8192",
        )
        generated_text = chat_completion.choices[0].message.content
        collection.insert_one({"transcript": transcript_text, "generated_text": generated_text})
        print("audio")
        return {"transcript": transcript_text1, "generated_text": generated_text}
    except Exception as e:
        error_message = f"Internal server error: {str(e)}"
        print(error_message)
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/send-text")
async def send_text(text_request: TextRequest):
    transcript_text = text_request.text
    client = Groq(api_key=grog_api)
    chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": transcript_text,
                }
            ],
            model="llama3-70b-8192",
        )
    generated_text = chat_completion.choices[0].message.content
    collection.insert_one({"transcript": transcript_text, "generated_text": generated_text})
    print(" text ")
    return {"transcript": transcript_text, "generated_text": generated_text}

@app.get("/retrieve-data", response_class=JSONResponse)
async def retrieve_data() -> List[dict]:
    try:
        data = retrieve_from_mongodb()
        return data
    except Exception as e:
        error_message = f"Internal server error: {str(e)}"
        print(error_message)
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=error_message)