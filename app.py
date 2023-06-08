from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
import openai
import pickle
import os
import gradio

# Load environment variables
# Find the .env file
load_dotenv(find_dotenv())

# Load the necessary data and initialize your chat bot components
with open("doc_search.pkl", "rb") as f:
    doc_search = pickle.load(f)
embeddings = OpenAIEmbeddings()
aasb_search = Chroma.from_documents(doc_search, embeddings)
chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type='stuff', vectorstore=aasb_search)

query = "What is the cost of capital?"
print(chain.run(query))

def AASB_Bot(User_Question):
    bot_reply = chain.run(User_Question)
 
    return f"Answer: {bot_reply}"


demo = gradio.Interface(fn=AASB_Bot, inputs = "text", outputs = "text", title = "Cost of Capital Bot")

demo.launch()