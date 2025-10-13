"""
CPA Study Assistant - RAG-based Q&A System
This code creates a study assistant that answers questions based on your CPA materials.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import sqlite3
from datetime import datetime
import hashlib

app = Flask(__name__)
CORS(app)  # Enable CORS for website integration

# Configuration
UPLOAD_FOLDER = 'study_materials'
VECTOR_DB_PATH = 'vectorstore'
USER_DB = 'users.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize embeddings model (runs locally, no API costs)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize vector store
vectorstore = None

# Custom prompt for CPA-focused responses
CPA_PROMPT = PromptTemplate(
    template="""You are an expert CPA exam tutor. Use the following context from study materials to answer the question.
    
Context: {context}

Question: {question}

Provide a clear, detailed answer that helps the student understand the concept. Include relevant examples if applicable.
If you're not sure about something, say so rather than guessing.

Answer:""",
    input_variables=["context", "question"]
)


def init_user_database():
    """Initialize SQLite database for user management and usage tracking"""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password_hash TEXT,
                  subscription_tier TEXT,
                  questions_remaining INTEGER,
                  created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS usage_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  question TEXT,
                  timestamp TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()


def load_and_process_documents():
    """Load all study materials and create vector database"""
    global vectorstore
    
    documents = []
    
    # Load PDF files
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    print(f"Loaded {len(documents)} documents and created {len(splits)} chunks")


def verify_user(username, password):
    """Verify user credentials and check subscription status"""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    c.execute("SELECT id, questions_remaining FROM users WHERE username=? AND password_hash=?",
              (username, password_hash))
    result = c.fetchone()
    conn.close()
    
    if result and result[1] > 0:
        return result[0]
    return None


def log_usage(user_id, question):
    """Log question usage and decrement counter"""
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    
    c.execute("INSERT INTO usage_logs (user_id, question, timestamp) VALUES (?, ?, ?)",
              (user_id, question, datetime.now()))
    c.execute("UPDATE users SET questions_remaining = questions_remaining - 1 WHERE id=?",
              (user_id,))
    
    conn.commit()
    conn.close()


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Main endpoint for asking questions"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    question = data.get('question')
    
    if not all([username, password, question]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Verify user
    user_id = verify_user(username, password)
    if not user_id:
        return jsonify({'error': 'Invalid credentials or no questions remaining'}), 401
    
    if vectorstore is None:
        return jsonify({'error': 'Study materials not loaded'}), 500
    
    try:
        # Initialize LLM (using Ollama - free, runs locally)
        llm = Ollama(model="llama2")  # You can use llama2, mistral, or other models
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": CPA_PROMPT}
        )
        
        # Get answer
        result = qa_chain({"query": question})
        answer = result['result']
        
        # Log usage
        log_usage(user_id, question)
        
        return jsonify({
            'answer': answer,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/register', methods=['POST'])
def register_user():
    """Register new user with subscription"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    tier = data.get('tier', 'basic')  # basic, premium, etc.
    
    # Set question limits based on tier
    question_limits = {
        'basic': 50,
        'premium': 200,
        'unlimited': 999999
    }
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        conn = sqlite3.connect(USER_DB)
        c = conn.cursor()
        c.execute("""INSERT INTO users (username, password_hash, subscription_tier, 
                     questions_remaining, created_at) VALUES (?, ?, ?, ?, ?)""",
                  (username, password_hash, tier, question_limits[tier], datetime.now()))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'User registered successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 400


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'running', 'vectorstore_loaded': vectorstore is not None})


if __name__ == '__main__':
    # Initialize database
    init_user_database()
    
    # Load study materials on startup
    print("Loading study materials...")
    load_and_process_documents()
    print("Ready to answer questions!")
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True)