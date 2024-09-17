import os
import shutil
from typing import Dict, List
import unicodedata
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import requests
from supabase import create_client
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging
import uuid
from datetime import datetime
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv()
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # Set a secret key for sessions
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

app.config['UPLOAD_FOLDER'] ='uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'txt'}
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192" # "mixtral-8x7b-32768"
TEMPERATURE = 0.2
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


# Supabase configuration
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def allowed_file(filename):
    print("Filename is ", filename)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_loader(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'pdf':
        return PyMuPDFLoader(file_path)
    elif file_extension == 'docx':
        return Docx2txtLoader(file_path)
    elif file_extension == 'xlsx':
        return UnstructuredExcelLoader(file_path)
    elif file_extension == 'txt':
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def count_paragraphs(documents):
    # Simple paragraph counting logic
    total_paragraphs = 0
    for doc in documents:
        total_paragraphs += doc.page_content.count('\n\n') + 1
    return total_paragraphs


@app.before_request
def set_client_session():
    # Get the ID from the query parameter
    if 'id' in request.args:
        # Set the user_id in the session
        session['user_id'] = request.args.get('id')

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return handle_options_request()

    if 'file' not in request.files and 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('file') if 'file' in request.files else request.files.getlist('files')
    uploaded_files = []
    errors = []

    for file in files:
        if file.filename == '':
            continue
        if file and allowed_file(file.filename):
            filename = sanitize_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)

            file_size = os.path.getsize(file_path)
            file_metadata = {
                "filename": filename,
                "user_id": session.get('user_id', 'Not set'),
                "created_date": datetime.now().strftime("%m/%d/%Y %I:%M %p"),
                "file_size": file_size,
                "page_count": 0,  # Initialize page count
                "paragraph_count": 0  # Initialize paragraph count
            }

            loader = get_loader(file_path)
            documents = loader.load()

            # Extract additional metadata
            file_metadata["page_count"] = len(documents)
            file_metadata["paragraph_count"] = count_paragraphs(documents)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20, length_function=len)
            texts = text_splitter.split_documents(documents)

            for text in texts:
                text.metadata.update(file_metadata)

            vector_store = SupabaseVectorStore.from_documents(
                documents=texts,
                embedding=embeddings,
                client=supabase,
                table_name="documents",
                query_name="match_documents"
            )

            os.remove(file_path)
            uploaded_files.append(file_metadata)
        else:
            errors.append(f"File type not allowed: {file.filename}")

    if uploaded_files:
        uploaded_files_str = ', '.join([file['filename'] for file in uploaded_files])
        return jsonify({"message": f"Files uploaded and processed successfully: {uploaded_files_str}"}), 200
    elif errors:
        return jsonify({"error": "; ".join(errors)}), 400
    else:
        return jsonify({"error": "No valid files were uploaded"}), 400

# Currently not used-
@app.route('/query', methods=['POST'])
def query():
    try:
        API_URL = "https://flowiseai-railway-production-cd23.up.railway.app/api/v1/prediction/729a2fee-cf8a-4ee5-9749-22c66972a9eb"
        question = {"question": request.json['query']}
        response = requests.post(API_URL, json=question)
        resp = response.json()

        print(resp['text'])
        return jsonify({"response": response.json()['text']}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/documents', methods=['GET', 'OPTIONS'])
def get_documents():
    print("user id session", session.get('user_id', 'Not set'))
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    # If files is not in the session, query supabase for the user's documents and save as session variable
    #if 'files' not in session:
    response = supabase.table("documents").select("*").eq("metadata->>user_id", session.get('user_id', 'Not set')).execute()
    if response.data:
        session['files'] = get_unique_documents(response.data) 
    else:
        session['files'] = []

    return jsonify({"documents": session.get('files', [])}), 200


def get_unique_documents(objects: List[Dict]) -> List[Dict]:
  unique_objects = []
  seen_objects = set()

  for obj in objects:
      metadata = obj.get('metadata')
      if metadata:
          # Create a new dictionary without the 'page' property
          filtered_metadata = {k: v for k, v in metadata.items() if k != 'page'}
          obj_hash = hash(tuple(sorted(filtered_metadata.items())))
          if obj_hash not in seen_objects:
              unique_objects.append(metadata)  # Append the original metadata
              seen_objects.add(obj_hash)

  return unique_objects


@app.route('/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    if request.method == 'OPTIONS':
        return handle_options_request()

    user_id =session.get('user_id', 'Not set')  # This should be determined based on your authentication system

    # Delete the document from Supabase
    response = supabase.table("documents").delete().eq("metadata->>filename", filename).eq("metadata->>user_id", user_id).execute()

    if response.data:
        return jsonify({"message": f"Document {filename} deleted successfully"}), 200
    else:
        return jsonify({"error": "Document not found or couldn't be deleted"}), 404

def handle_options_request():
    response = app.make_default_options_response()
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, DELETE',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Credentials': 'true'
    }
    response.headers.extend(headers)
    return response

@app.route('/')
def home():
    return render_template('index3.html')

def sanitize_filename(filename):
    filename = unicodedata.normalize('NFKD', filename)
    filename = filename.encode('ASCII', 'ignore').decode('ASCII')
    return secure_filename(filename)

if __name__ == '__main__':
    logger.info("Starting server")
    port = int(os.environ.get('PORT', 8000))
    app.run(port=8000, debug=True)