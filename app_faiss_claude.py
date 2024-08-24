import json
import os
import random
import re
import shutil
import unicodedata
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging
import uuid
import datetime
from langchain_anthropic import ChatAnthropic
import anthropic
from pypdf import PdfReader


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
LLM_MODEL = "claude-3-haiku-20240307"
TEMPERATURE = 0.3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FAISS_INDEX_FOLDER'] = FAISS_INDEX_FOLDER

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

llm = ChatAnthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),model=LLM_MODEL, temperature=TEMPERATURE, max_tokens=1024)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
filecontent = ""


def allowed_file(filename):
    print("Filename is ", filename)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_loader(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'pdf':
        return PyPDFLoader(file_path)
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

    print("Uploading file and creating FAISS index...")
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
        print(">>Filename is ", file.filename)
        if file and allowed_file(file.filename):
            try:

                filename = sanitize_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # save the text to filecontent
                reader = PdfReader(file_path)
                number_of_pages = len(reader.pages)
                text = ''.join(page.extract_text() for page in reader.pages)
                filecontent = text
                print(datetime.datetime.now(), ">>how many pages:", number_of_pages)
                print(filecontent)

                print(datetime.datetime.now(), ">>getting loader...")
                loader = get_loader(file_path)
                documents = loader.load()
                print(datetime.datetime.now(), ">>loader loaded...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
                texts = text_splitter.split_documents(documents)
                
                print(datetime.datetime.now(), ">>FAISS.from_documents started...")
                start = datetime.datetime.now()
                faiss_index = FAISS.from_documents(texts, embeddings)
                end = datetime.datetime.now()
                print(datetime.datetime.now(), ">>FAISS.from_documents duration:", end - start)
                
                index_path = os.path.join(app.config['FAISS_INDEX_FOLDER'], f"{session['session_id']}_{filename}.faiss")
                faiss_index.save_local(index_path)
                
                session['files'].append(filename)
                session.modified = True
                uploaded_files.append(filename)
                print(datetime.datetime.now(), ">>remove filepath")

                os.remove(file_path)
                print(datetime.datetime.now(), ">>removed filepath")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                errors.append(f"Error processing {filename}: {str(e)}")
        else:
            logger.error(f"Invalid file type for {file.filename}")
            errors.append(f"Invalid file type for {file.filename}")

    if uploaded_files:
        message = "Files uploaded successfully"
        if errors:
            message += f", but with some errors: {'; '.join(errors)}"
        return jsonify({"message": message, "filenames": uploaded_files}), 200
    else:
        logger.error(f"No valid files were uploaded. Errors: {'; '.join(errors)}")
        return jsonify({"error": f"No valid files were uploaded. Errors: {'; '.join(errors)}"}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('query')
    messages = [
        {
            "role": "user",
            "content": question
        },       
        {
            "role": "assistant",
            "content": f"You are an AI assistant that answers questions based solely on the following content:\n\n{filecontent}\n\nDo not use any external knowledge. If the answer is not in the provided content, say you don't know."
        },
    ]

    def generate():
        try:
            with client.messages.stream(
                max_tokens=1024,
                messages=messages,
                model="claude-3-sonnet-20240229",
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
        except Exception as e:
            logger.error(f"Error in generate function: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

def query2():

    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=2048,
        temperature=0,
        system=f"""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {filecontent}

            Question: {query}
            Answer:""",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
        ]
    )
    print(">>Messages", message.content)
    return jsonify({"response": message.content[0].text}), 200

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