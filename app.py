from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pypdf import PdfReader
import tempfile
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load .env file
load_dotenv()


# Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Request model
class AskRequest(BaseModel):
    notes: str
    question: str

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "AI Study Assistant Running"
    }

# Ask endpoint
@app.post("/ask")
def ask_question(data: AskRequest):

    prompt = f"""
Use ONLY the notes below to answer.

Notes:
{data.notes}

Question:
{data.question}

If the answer is not in the notes, reply:
Answer not found in provided notes.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer
    }
class NotesRequest(BaseModel):
    notes: str

@app.post("/quiz")
def generate_quiz(data: NotesRequest):

    prompt = f"""
Generate 5 quiz questions from these notes.

Notes:
{data.notes}

Return only the questions, one per line.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response.choices[0].message.content

    questions = [
        q.strip()
        for q in result.split("\n")
        if q.strip()
    ]

    return {
        "questions": questions
    }
@app.post("/flashcards")
def generate_flashcards(data: NotesRequest):

    prompt = f"""
Create 5 flashcards from these notes.

Format:
Question | Answer

Notes:
{data.notes}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response.choices[0].message.content

    flashcards = []

    for line in result.split("\n"):

        if "|" in line:

            front, back = line.split("|", 1)

            flashcards.append({
                "front": front.strip(),
                "back": back.strip()
            })

    return {
        "flashcards": flashcards
    }
class StudyPlanRequest(BaseModel):
    subject: str
    days: int
@app.post("/study-plan")
def generate_study_plan(data: StudyPlanRequest):

    prompt = f"""
Create a {data.days}-day study plan for {data.subject}.

Requirements:
- Day-wise plan
- Short tasks
- Easy to follow
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response.choices[0].message.content

    plan = []

    for line in result.split("\n"):
        if line.strip():
            plan.append({
                "task": line.strip()
            })

    return {
        "plan": plan
    }
@app.post("/summary")
def generate_summary(data: NotesRequest):

    prompt = f"""
Create a concise study summary from the notes below.

Notes:
{data.notes}

Requirements:
- Important points only
- Easy to understand
- Bullet points preferred
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    summary = response.choices[0].message.content

    return {
        "summary": summary
    }   
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    content = await file.read()

    temp_file.write(content)

    temp_file.close()

    reader = PdfReader(temp_file.name)

    pdf_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pdf_text += text + "\n"

    prompt = f"""
Create a study summary from this PDF content.

Content:
{pdf_text[:10000]}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    summary = response.choices[0].message.content

    return {
        "filename": file.filename,
        "summary": summary
    }