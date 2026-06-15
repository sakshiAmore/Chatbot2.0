from flask import Flask, request, jsonify, send_from_directory
from langchain_core.messages import HumanMessage
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.agents import create_agent
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import math
import os

# ---------------- LOAD ENV ----------------
load_dotenv()

MONGO_URI = "mongodb://localhost:27017/"

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------- RAG DOCUMENTS ----------------
documents = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="FAISS enables efficient similarity search."),
    Document(page_content="Flask is used to build backend APIs."),
]

# ---------------- MONGODB ----------------
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
collection = db["conversations"]

# ---------------- SAVE MESSAGE ----------------
def save_message(conversation_id, role, content):
    collection.insert_one({
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    })

# ---------------- LOAD HISTORY ----------------
def load_history(conversation_id, limit=6):
    messages = list(
        collection.find({"conversation_id": conversation_id})
        .sort("timestamp", -1)
        .limit(limit)
    )

    messages.reverse()

    history = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    return history

# ---------------- GET RECENT CHATS ----------------
def get_recent_chats(limit=10):
    pipeline = [
        {"$sort": {"timestamp": -1}},
        {
            "$group": {
                "_id": "$conversation_id",
                "last_time": {"$first": "$timestamp"}
            }
        },
        {"$sort": {"last_time": -1}},
        {"$limit": limit}
    ]
    return list(collection.aggregate(pipeline))

# ---------------- RAG TOOL ----------------
def rag_tool(query):
    """Answer questions using local knowledge documents."""
    lower_query = query.lower()
    docs = [
        doc.page_content
        for doc in documents
        if lower_query in doc.page_content.lower()
    ]
    if not docs:
        docs = [doc.page_content for doc in documents]

    context = "\n".join(docs)

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ---------------- CALCULATOR TOOL ----------------
def calculator_tool(query):
    """Evaluate simple math expressions safely."""
    try:
        allowed = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "pow": pow,
            "abs": abs,
            "round": round
        }
        return str(eval(query, allowed))
    except Exception:
        return "Invalid calculation"

# ---------------- TOOLS ----------------
# Pass plain callables to create_agent; it will wrap them internally.
tools = [
    rag_tool,
    calculator_tool,
]

# ---------------- AGENT ----------------
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,
)

# ---------------- BOT RESPONSE ----------------
def get_bot_response(user_message, conversation_id):
    history = load_history(conversation_id)

    prompt = f"""
Conversation History:
{history}

User:
{user_message}
"""

    try:
        # The compiled agent graph expects inputs as a dict with `messages`.
        inputs = {"messages": [{"role": "user", "content": prompt}]}

        outputs = agent.run(inputs) if hasattr(agent, "run") else None

        if outputs is None:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content

        # If outputs is a dict with messages, extract the assistant content
        if isinstance(outputs, dict) and "messages" in outputs:
            msgs = outputs["messages"]
            if msgs:
                last = msgs[-1]
                return last.get("content") if isinstance(last, dict) else getattr(last, "content", str(last))

        # If outputs is a list of messages or strings
        if isinstance(outputs, list) and outputs:
            last = outputs[-1]
            if isinstance(last, dict) and "content" in last:
                return last["content"]
            return str(last)

        return str(outputs)

    except Exception:
        response = llm.invoke([HumanMessage(content=user_message)])
        return response.content

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return send_from_directory(
        str(Path(__file__).resolve().parent),
        "premium_chatbot_ui.html"
    )

# ---------------- CLEAR CHAT ----------------
@app.route("/clear/<conversation_id>", methods=["POST"])
def clear_conversation(conversation_id):
    collection.delete_many({"conversation_id": conversation_id})
    return jsonify({"status": "cleared"})

# ---------------- CHAT API ----------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    conversation_id = data.get("conversation_id", "default")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    save_message(conversation_id, "user", user_message)

    bot_response = get_bot_response(user_message, conversation_id)

    save_message(conversation_id, "assistant", bot_response)

    return jsonify({"response": bot_response})

# ---------------- HISTORY LIST ----------------
@app.route("/history", methods=["GET"])
def history_list():
    chats = get_recent_chats()

    return jsonify([
        {
            "conversation_id": c["_id"],
            "last_time": c["last_time"].isoformat()
        }
        for c in chats
    ])

# ---------------- LOAD FULL CHAT ----------------
@app.route("/history/<conversation_id>", methods=["GET"])
def load_chat(conversation_id):
    msgs = list(
        collection.find({"conversation_id": conversation_id})
        .sort("timestamp", 1)
    )

    return jsonify([
        {
            "role": m["role"],
            "content": m["content"],
            "time": m["timestamp"].isoformat()
        }
        for m in msgs
    ])

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)