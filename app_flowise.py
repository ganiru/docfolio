import os
import shutil
import unicodedata
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import requests
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging
import uuid
import datetime

load_dotenv()
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # Set a secret key for sessions
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'txt'}
FAISS_INDEX_FOLDER = 'faiss_indexes'
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192" # "mixtral-8x7b-32768"
TEMPERATURE = 0.2
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FAISS_INDEX_FOLDER'] = FAISS_INDEX_FOLDER

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name=LLM_MODEL,
    temperature=TEMPERATURE,
    streaming=False,
)

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

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    print("Uploading...")
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('file')
    uploaded_files = []
    errors = []

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FAISS_INDEX_FOLDER'], exist_ok=True)

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    if 'files' not in session:
        session['files'] = []

    for file in files:
        if file.filename == '':
            continue
        
        if not isinstance(file.filename, str):
            logger.error(f"Unexpected filename type: {type(file.filename)}")

        print("** THE Filename is ", file.filename)
        status = ''
        if file and allowed_file(file.filename):
            try:
                filename = sanitize_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                print(">> Sanitized filename ", filename)
                API_URL = "https://flowiseai-railway-production-cd23.up.railway.app/api/v1/vector/upsert/729a2fee-cf8a-4ee5-9749-22c66972a9eb"

                print("opening the file...")
                # use form data to upload files
                form_data = {
                    "files": (file_path, open(file_path, 'rb'),'application/pdf')
                }
                print("file opened")

                print(">> Sanitized filename ", filename)
                body_data = {
                    "returnSourceDocuments": False,
               #     "supabaseApiKey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNld3djcWtxb2F2cXJ1Z3BnY295Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjQzNjM1OTYsImV4cCI6MjAzOTkzOTU5Nn0.SgfiCTqxfwdPx8P-scV4QpAINQLEJgXOFXpgCzh4NNw",
              #      "supabaseProjUrl": "https://cewwcqkqoavqrugpgcoy.supabase.co",
             #       "tableName": "docfolio_documents",
                }
                response = requests.post(API_URL, data=body_data, files=form_data)
                status = response.json()
                session['files'].append(filename)
                session.modified = True
                uploaded_files.append(filename)
                print(datetime.datetime.now(), ">>remove filepath")

                os.remove(file_path)
                print(datetime.datetime.now(), ">>removed filepath")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                errors.append(f"Error processing {filename}: {str(e)}")
                print(f"Error uploading and processing {e}")
        else:
            logger.error(f"Invalid file type for {file.filename}")
            errors.append(f"Invalid file type for {file.filename}")

    if uploaded_files:
        message = "Files uploaded successfully"
        print(status)
        if errors:
            message += f", but with some errors: {'; '.join(errors)}"
        return jsonify({"message": message, "filenames": uploaded_files}), 200
    else:
        logger.error(f"No valid files were uploaded. Errors: {'; '.join(errors)}")
        return jsonify({"error": f"No valid files were uploaded. Errors: {'; '.join(errors)}"}), 400

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
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    return jsonify({"documents": session.get('files', [])}), 200

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    print(session['session_id'])
    try:
        # Delete FAISS index
        index_path = os.path.join(app.config['FAISS_INDEX_FOLDER'], f"{session['session_id']}_{filename}.faiss")
        if os.path.exists(index_path):
            if os.path.isdir(index_path):
                shutil.rmtree(index_path)
            else:
                os.remove(index_path)
            logger.info(f"Deleted FAISS index at {index_path}")
        else:
            logger.warning(f"FAISS index not found at {index_path}")
        
        if filename in session['files']:
            session['files'].remove(filename)
            session.modified = True
            logger.info(f"Removed {filename} from session files")
        
        return jsonify({"message": f"Document {filename} deleted successfully"}), 200
    except PermissionError as pe:
        logger.error(f"Permission error while deleting {filename}: {str(pe)}")
        return jsonify({"error": f"Permission denied when deleting {filename}. Please check file permissions."}), 403
    except Exception as e:
        logger.error(f"Error deleting {filename}: {str(e)}")
        return jsonify({"error": f"Failed to delete document {filename}: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "<table style='border:solid 1px gray' border=1><tr style='background-color:gray; color:white'><th>head1</th><th>h2</th></tr><tr><td>some sample</td><td>Some thing else</td></tr></table>"}), 200

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
    return render_template('index.html')

def sanitize_filename(filename):
    filename = unicodedata.normalize('NFKD', filename)
    filename = filename.encode('ASCII', 'ignore').decode('ASCII')
    return secure_filename(filename)

if __name__ == '__main__':
    logger.info("Starting server")
    port = int(os.environ.get('PORT', 8080))
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FAISS_INDEX_FOLDER'], exist_ok=True)
    logger.info("UPLOAD_FOLDER and FAISS_INDEX_FOLDER set")
    app.run(port=port, debug=True)