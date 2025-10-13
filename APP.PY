from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- NEW IMPORTS FOR LLM AND RAG CHAIN ---
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub 

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for website integration

# Configuration
UPLOAD_FOLDER = 'Study_Materials'
VECTOR_DB_PATH = 'vectorstore'

# --- LLM CONFIGURATION (REQUIRES HUGGINGFACEHUB_API_TOKEN ENV VAR) ---
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
LLM_REPO_ID = "google/flan-t5-large" # Placeholder LLM

# Initialize embeddings model (runs locally)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize LLM (used for generating the final answer)
llm = None
if HUGGINGFACE_TOKEN:
    print("Initializing LLM...")
    try:
        llm = HuggingFaceHub(
            repo_id=LLM_REPO_ID,
            huggingfacehub_api_token=HUGGINGFACE_TOKEN,
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )
        print(f"✓ LLM Initialized ({LLM_REPO_ID})")
    except Exception as e:
        print(f"Error initializing LLM: {e}. Running in context-only mode.")
else:
    print("Warning: HUGGINGFACEHUB_API_TOKEN not set. Running in context-only mode.")

# Initialize vector store (Global Variable)
vectorstore = None

# -------------------------------------------------------------------
# --- VECTOR STORE LOADING LOGIC (CRITICAL FOR RAILWAY DEPLOYMENT) ---
# -------------------------------------------------------------------

def try_load_vectorstore():
    """Attempts to load an existing, persistent vector store."""
    global vectorstore
    if os.path.exists(VECTOR_DB_PATH):
        print(f"Loading existing vector store from {VECTOR_DB_PATH}...")
        try:
            vectorstore = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            if vectorstore._collection.count() > 0:
                print(f"✓ Pre-built vector store loaded with {vectorstore._collection.count()} entries.")
                return True
        except Exception as e:
            print(f"Error loading vector store: {e}. Attempting to rebuild.")
    return False


def load_and_process_documents():
    """Load all study materials and create vector database"""
    global vectorstore
    
    print("Loading study materials...")
    documents = []
    
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"Warning: {UPLOAD_FOLDER} folder not found! Cannot build vector store.")
        return
    
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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
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
    """Main endpoint for asking questions, now using LLM for synthesis"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if vectorstore is None:
            return jsonify({'error': 'Study materials not loaded yet. Please wait...'}), 500
        
        # Search for relevant content using the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        if llm is None:
            # Fallback: return raw context chunks if LLM is not available
            docs = retriever.get_relevant_documents(question)
            if not docs:
                 return jsonify({
                     'answer': 'I could not find relevant information in the study materials to answer this question. (LLM Disabled)'
                 })

            answer_parts = [doc.page_content.strip() for doc in docs]
            answer = "⚠️ **LLM Disabled!** Retrieved Context:\n\n---\n\n" + "\n\n---\n\n".join(answer_parts)
            
        else:
            # ⭐ New Logic: Use RetrievalQA Chain to retrieve context AND generate an answer ⭐
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )
            result = qa_chain({"query": question})
            answer = result['result']

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
        'materials_folder_exists': os.path.exists(UPLOAD_FOLDER),
        'llm_initialized': llm is not None
    })


if __name__ == '__main__':
    # 1. Attempt to load the pre-built vector store first (FAST startup)
    if not try_load_vectorstore():
        # 2. Only run the expensive rebuild if it fails (SLOW startup/first run)
        load_and_process_documents()
    
    # Run server
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 Starting CPA Study Assistant on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)


