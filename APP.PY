"""
CPA Study Assistant - RAG-based Q&A System
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'Study_Materials'
VECTOR_DB_PATH = 'vectorstore'

# Global variable for vectorstore
vectorstore = None
embeddings = None

def initialize_embeddings():
    """Initialize embeddings model (only once)"""
    global embeddings
    if embeddings is None:
        print("Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings

def load_and_process_documents():
    """Load study materials and create vector database"""
    global vectorstore
    
    if vectorstore is not None:
        print("Vector store already loaded")
        return vectorstore
    
    print("Loading study materials...")
    documents = []
    
    # Check if folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"Warning: {UPLOAD_FOLDER} folder not found!")
        return None
    
    # Load all PDF files
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")
    
    for filename in pdf_files:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        try:
            print(f"Loading {filename}...")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    if not documents:
        print("No documents loaded!")
        return None
    
    print(f"Loaded {len(documents)} pages total")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} text chunks")
    
    # Create vector store
    print("Creating vector database...")
    emb = initialize_embeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=emb,
        persist_directory=VECTOR_DB_PATH
    )
    
    print("Vector database created successfully!")
    return vectorstore

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handle question requests"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Load documents if not already loaded (lazy loading)
        store = load_and_process_documents()
        
        if store is None:
            return jsonify({
                'answer': 'Sorry, the study materials are not available yet. Please make sure PDF files are uploaded to the Study_Materials folder.',
                'sources': []
            }), 200
        
        # Search for relevant documents
        docs = store.similarity_search(question, k=3)
        
        if not docs:
            return jsonify({
                'answer': 'I could not find relevant information in the study materials to answer your question.',
                'sources': []
            }), 200
        
        # Format the answer from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a simple answer based on the context
        answer = f"Based on your CPA study materials:\n\n{context[:500]}..."
        
        # Get source information
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            sources.append(f"{os.path.basename(source)} (Page {page})")
        
        return jsonify({
            'answer': answer,
            'sources': sources
        }), 200
        
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
