# uvicorn app:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


# FastAPI app instance
app = FastAPI()

# Allow CORS for all origins (React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your React frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.7, max_tokens=100)  # You can adjust the temperature for creativity

# Define the prompt template
template = """
You are a helpful assistant. Answer the following question based on your knowledge:
Provide informative and relevant responses to asked questions.
You must Avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
Be patient and considerate when responding to user queries, and provide clear explanations.
If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
Do Not generate the long paragarphs in response. Maximum Words should be 100.
Question: {user_input}

Answer:
"""
prompt = PromptTemplate(input_variables=["user_input"], template=template)
chatbot_chain = LLMChain(llm=llm, prompt=prompt)

# Define the request body model
class ChatRequest(BaseModel):
    user_input: str


@app.get("/")
async def home():
    return {"message": "success"}


# API to handle chatbot responses
@app.post("/chat/")
async def chat(request: ChatRequest):
    response = chatbot_chain.invoke({"user_input": request.user_input})
    return {"response": response["text"]}

