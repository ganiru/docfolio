import os
import random
import re
import shutil
import unicodedata
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging
import uuid
import datetime
import pickle

load_dotenv()
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # Set a secret key for sessions
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

chroma_client = chromadb.Client()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name="mixtral-8x7b-32768",
    temperature=0.4,
    streaming=True,
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('file')
    uploaded_files = []
    errors = []

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    if 'files' not in session:
        session['files'] = []

    for file in files:
        if file.filename == '':
            continue
        
        if not isinstance(file.filename, str):
            logger.error(f"Unexpected filename type: {type(file.filename)}")

        if file and allowed_file(file.filename):
            try:
                filename = sanitize_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                print(datetime.datetime.now(), "=>>before collectionname: ", f"{session['session_id']}_{filename}")
                collection_name = fix_collection_name(f"{session['session_id']}_{filename}")
                print(datetime.datetime.now(), ">>after collectionname: ", collection_name)
                
                print(datetime.datetime.now(), ">>Chroma.from_documents")
                Chroma.from_documents(texts, embeddings, collection_name=collection_name)
                print(datetime.datetime.now(), ">>session append")
                
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

@app.route('/upload_faiss', methods=['POST', 'OPTIONS'])
def upload_file_faiss():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('file')
    uploaded_files = []
    errors = []

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    if 'files' not in session:
        session['files'] = []

    for file in files:
        if file.filename == '':
            continue
        
        if not isinstance(file.filename, str):
            logger.error(f"Unexpected filename type: {type(file.filename)}")

        if file and allowed_file(file.filename):
            try:
                filename = sanitize_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                print(datetime.datetime.now(), ">>FAISS.from_documents")
                vectorstore = FAISS.from_documents(texts, embeddings)
                
                # Save FAISS index
                index_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['session_id']}_{filename}_faiss")
                vectorstore.save_local(index_path)
                
                print(datetime.datetime.now(), ">>session append")
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
        message = "Files uploaded successfully using FAISS"
        if errors:
            message += f", but with some errors: {'; '.join(errors)}"
        return jsonify({"message": message, "filenames": uploaded_files}), 200
    else:
        logger.error(f"No valid files were uploaded. Errors: {'; '.join(errors)}")
        return jsonify({"error": f"No valid files were uploaded. Errors: {'; '.join(errors)}"}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')
    filenames = data.get('filenames')
    use_faiss = data.get('use_faiss', False)
    
    if not query or not filenames:
        return jsonify({"error": "Missing query or filenames"}), 400
    
    def generate():
        combined_retriever = []
        for filename in filenames:
            if use_faiss:
                index_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['session_id']}_{filename}.faiss")
                vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            else:
                collection_name = fix_collection_name(f"{session['session_id']}_{filename}")
                vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings)
            
            combined_retriever.extend(vectorstore.as_retriever().invoke(query))

        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": lambda _: combined_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    return Response(stream_with_context(generate()), content_type='text/plain')

@app.route('/documents', methods=['GET', 'OPTIONS'])
def get_documents():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    return jsonify({"documents": session.get('files', [])}), 200

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    print(session['session_id'])
    try:
        # Delete Chroma collection
        collection_name = fix_collection_name(f"{session['session_id']}_{filename}")
        chroma_client.delete_collection(collection_name)
        
        # Delete FAISS index if it exists
        faiss_index_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['session_id']}_{filename}_faiss")
        if os.path.exists(faiss_index_path):
            shutil.rmtree(faiss_index_path)
        
        if filename in session['files']:
            session['files'].remove(filename)
            session.modified = True
        return jsonify({"message": f"Document {filename} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 400

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

def fix_collection_name(word):
    fixedCollectionName = word
    if len(word) > 62:
        fixedCollectionName = word[:62]
    
    if not fixedCollectionName[-1].isalnum():
        fixedCollectionName = fixedCollectionName[:-1] + str(random.randint(0, 9))
    
    fixedCollectionName = re.sub(r'\.\.+', '.', fixedCollectionName)
    print(word, "****", fixedCollectionName)
    return fixedCollectionName

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
    logger.info("UPLOAD_FOLDER set")
    app.run(port=port, debug=True)