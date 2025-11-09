import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load env vars
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Models
openai_model = ChatOpenAI()
ollama_model = Ollama(model="llama3.2:latest")

# Prompts
essay_prompt = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words"
)
poem_prompt = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5 year old child with 100 words"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/essay", methods=["POST"])
def generate_essay():
    data = request.json
    topic = data.get("topic", "")
    chain = essay_prompt | openai_model
    response = chain.invoke({"topic": topic})
    return jsonify({"response": response.content})

@app.route("/api/poem", methods=["POST"])
def generate_poem():
    data = request.json
    topic = data.get("topic", "")
    chain = poem_prompt | ollama_model
    response = chain.invoke({"topic": topic})
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
