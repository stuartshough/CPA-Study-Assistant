python"""
CPA Study Assistant - RAG-based Q&A System
This code creates a study assistant that answers questions based on your CPA materials.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for website integration

# Configuration
UPLOAD_FOLDER = 'Study_Materials'
VECTOR_DB_PATH = 'vectorstore'

# Initialize embeddings model (runs locally, no API costs)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize vector store
vectorstore = None


def load_and_process_documents():
    """Load all study materials and create vector database"""
    global vectorstore
    
    print("Loading study materials...")
    documents = []
    
    # Check if folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"Warning: {UPLOAD_FOLDER} folder not found!")
        return
    
    # Load PDF files
    pdf_count = 0
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.pdf'):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                pdf_count += 1
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not documents:
        print("No documents loaded!")
        return
    
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
    
    print(f"✓ Loaded {pdf_count} PDF files")
    print(f"✓ Created {len(splits)} searchable chunks")
    print("✓ Ready to answer questions!")


@app.route('/')
def serve_frontend():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Main endpoint for asking questions"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if vectorstore is None:
            return jsonify({'error': 'Study materials not loaded yet. Please wait...'}), 500
        
        # Search for relevant content
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return jsonify({
                'answer': 'I could not find relevant information in the study materials to answer this question.'
            })
        
        # Combine the top results into an answer
        answer_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            # Clean up the content
            if len(content) > 500:
                content = content[:500] + "..."
            answer_parts.append(f"{content}")
        
        # Create a comprehensive answer
        answer = "\n\n".join(answer_parts)
        
        # Add a header based on the question topic
        if len(answer) > 100:
            answer = f"Based on your CPA study materials:\n\n{answer}"
        
        return jsonify({
            'answer': answer,
            'success': True
        })
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running', 
        'vectorstore_loaded': vectorstore is not None,
        'materials_folder_exists': os.path.exists(UPLOAD_FOLDER)
    })


if __name__ == '__main__':
    # Load study materials on startup
    load_and_process_documents()
    
    # Run server
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 Starting CPA Study Assistant on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 Starting CPA Study Assistant on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

