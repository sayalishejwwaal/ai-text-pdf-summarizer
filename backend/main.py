# Import FastAPI framework to create API endpoints
from fastapi import FastAPI, UploadFile, File  

from fastapi import FastAPI

# Import BaseModel to define request body structure
from pydantic import BaseModel  

# Import PdfReader to extract text from PDF files
from PyPDF2 import PdfReader  

# Import CORS middleware to allow frontend-backend communication
from fastapi.middleware.cors import CORSMiddleware  

# Import OpenAI client for AI summarization
from openai import OpenAI  

# Import os to read environment variables (API key)
import os  


# Create FastAPI app instance
app = FastAPI()  


# Enable CORS so frontend (React) can call backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (for development)
    allow_credentials=True,     # Allow cookies
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)


# Initialize OpenAI client using API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  


# Define request body structure for text summarization
class TextRequest(BaseModel):  
    text: str                  # Input text
    length: str = "medium"     # Summary length (default = medium)


# Root endpoint to check if backend is running
@app.get("/")  
def home():  
    return {"message": "Backend is working 🚀"}  


# ===============================
# TEXT SUMMARIZATION API
# ===============================
@app.post("/summarize")  
def summarize(data: TextRequest):  

    try:
        # Extract user input text
        user_text = data.text  

        # Extract selected summary length
        summary_length = data.length  

        # Decide instruction based on selected length
        if summary_length == "short":
            instruction = "Summarize in 2 bullet points."
        elif summary_length == "long":
            instruction = "Summarize in detailed 8-10 bullet points."
        else:
            instruction = "Summarize in 4-5 bullet points."

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # Fast and cost-efficient model
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert summarizer. Provide clear bullet-point summaries."
                },
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{user_text}"
                }
            ],
        )

        # Extract summary text from response
        summary = response.choices[0].message.content  

        # Return summary to frontend
        return {"summary": summary}  

    except Exception as e:
        # Return error message if something fails
        return {"error": str(e)}  


# ===============================
# PDF SUMMARIZATION API
# ===============================
@app.post("/summarize-pdf")  
def summarize_pdf(file: UploadFile = File(...)):  

    try:
        # Read uploaded PDF file
        reader = PdfReader(file.file)  

        # Initialize empty text variable
        text = ""  

        # Extract text from each page
        for page in reader.pages:
            text += page.extract_text() or ""  

        # Call OpenAI API for summarization
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the PDF content into clear bullet points."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
        )

        # Extract summary
        summary = response.choices[0].message.content  

        # Return summary
        return {"summary": summary}  

    except Exception as e:
        # Return error if something fails
        return {"error": str(e)}  

 