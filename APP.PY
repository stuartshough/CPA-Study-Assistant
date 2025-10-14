"""
CPA Study Assistant - Simple PDF Search
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import PyPDF2

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'Study_Materials'
pdf_contents = {}


def load_pdfs():
    """Load all PDF files into memory"""
    global pdf_contents
    
    print("Loading PDF files...")
    
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"ERROR: {UPLOAD_FOLDER} folder not found!")
        return
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.pdf'):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                with open(filepath, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    pdf_contents[filename] = text
                    print(f"✓ Loaded: {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
    
    print(f"\n✓ Total PDFs loaded: {len(pdf_contents)}")


def search_pdfs(question):
    """Search PDFs for relevant content"""
    question_lower = question.lower()
    results = []
    
    for filename, content in pdf_contents.items():
        content_lower = content.lower()
        
        # Find relevant sections
        sentences = content.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_lower.split()):
                relevant_sentences.append(sentence.strip())
                if len(relevant_sentences) >= 3:
                    break
        
        if relevant_sentences:
            results.append({
                'file': filename,
                'content': '. '.join(relevant_sentences)
            })
    
    return results


@app.route('/')
def serve_frontend():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Answer questions based on PDF content"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if not pdf_contents:
            return jsonify({'error': 'No study materials loaded'}), 500
        
        # Search for relevant content
        results = search_pdfs(question)
        
        if not results:
            return jsonify({
                'answer': 'I could not find relevant information in your study materials for this question. Try rephrasing or asking about a different topic.'
            })
        
        # Build answer from results
        answer = "Based on your CPA study materials:\n\n"
        for i, result in enumerate(results[:3], 1):
            answer += f"{result['content']}\n\n"
        
        return jsonify({
            'answer': answer.strip(),
            'success': True
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'running',
        'pdfs_loaded': len(pdf_contents)
    })


if __name__ == '__main__':
    # Load PDFs on startup
    load_pdfs()
    
    # Start server
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 Server starting on port {port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
