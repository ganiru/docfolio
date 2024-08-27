import os
import unicodedata
import logging
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_cors import CORS
from langchain_groq import ChatGroq
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from dotenv import load_dotenv
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # Set a secret key for sessions
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Supabase configuration
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Embeddings model
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = OpenAIEmbeddings() # HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

@app.before_request
def set_client_session():
    # Get the ID from the query parameter
    if 'id' in request.args:
        # Set the user_id in the session
        session['user_id'] = request.args.get('id')


@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    # return an error if the user_id is not provided
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    filename = unicodedata.normalize('NFKD', filename)
    filename = filename.encode('ASCII', 'ignore').decode('ASCII')
    return secure_filename(filename)

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
            file.save(file_path)
            
            loader = get_loader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            for text in texts:
                text.metadata["filename"] = filename
                text.metadata["user_id"] = session.get('user_id', 'Not set')
            
            vector_store = SupabaseVectorStore.from_documents(
                documents=texts,
                embedding=embeddings,
                client=supabase,
                table_name="documents",
                query_name="match_documents"
            )
            
            os.remove(file_path)
            uploaded_files.append(filename)
        else:
            errors.append(f"File type not allowed: {file.filename}")
    
    if uploaded_files:
        return jsonify({"message": f"Files uploaded and processed successfully: {', '.join(uploaded_files)}"}), 200
    elif errors:
        return jsonify({"error": "; ".join(errors)}), 400
    else:
        return jsonify({"error": "No valid files were uploaded"}), 400

@app.route('/documents', methods=['GET', 'OPTIONS'])
def get_documents():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    user_id = session.get('user_id', 'Not set')  # This should be determined based on your authentication system
    
    # Query Supabase for documents belonging to the user
    response = supabase.table("documents").select("metadata->filename").eq("metadata->>user_id", user_id).execute()
    
    # Extract unique filenames from the response
    filenames = list(set(item['filename'] for item in response.data if 'filename' in item))
    
    return jsonify({"documents": filenames}), 200

@app.route('/delete/<filename>', methods=['DELETE', 'OPTIONS'])
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


@app.route('/query', methods=['POST', 'OPTIONS'])
def query_document():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    data = request.json
    query = data.get('query')
    user_id = session.get('user_id', 'Not set')
    filenames = data.get('filenames', [])
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    #if not filenames:
    #    return jsonify({"error": "No filenames provided"}), 400
    
    # Count total records
    count_response = supabase.table("documents").select("id", count="exact").execute()
    total_records = count_response.count
    
    logger.info(f"Total records in the vector store: {total_records}")
    
    if total_records == 0:
        return jsonify({"error": "No documents found in the vector store"}), 404
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    
    # Construct the filter for user_id and filenames
    filter_query = supabase.table("documents").select("id", count="exact")
    filter_query = filter_query.eq("metadata->>user_id", user_id)
    filter_query = filter_query.in_("metadata->>filename", filenames)
    
    # Count filtered records
    filtered_count_response = filter_query.execute()
    filtered_records = filtered_count_response.count
    
    logger.info(f"Filtered records: {filtered_records}")
    
    #if filtered_records == 0:
        #return jsonify({"error": "No matching documents found after filtering"}), 404
    
    # Construct the filter for similarity search
    similarity_filter = {
            "user_id": user_id,
        #    "filename": filenames,
    }
    
    # Perform the similarity search with the correct filter
    results = vector_store.similarity_search(
        query,
        filter=similarity_filter
    )
    
    logger.info(f"Query: {query}")
    logger.info(f"Filenames: {filenames}")
    logger.info(f"Filter: {similarity_filter}")
    logger.info(f"Number of results: {len(results)}")
    for doc in results:
        logger.info(f"Document filename: {doc.metadata.get('filename')}")
    
    if not results:
        return jsonify({"error": "No matching documents found in similarity search"}), 404
        
    context = "\n".join([doc.page_content for doc in results])

    chat_groq = ChatGroq(
        temperature=0.4,
        model_name="mixtral-8x7b-32768",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt
        | chat_groq
        | StrOutputParser()
    )
    
    def generate():
        for chunk in chain.stream({"context": context, "query": query}):
            yield chunk
    
    return Response(stream_with_context(generate()), content_type='text/plain')

def handle_options_request():
    response = app.make_default_options_response()
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Credentials': 'true'
    }
    response.headers.extend(headers)
    return response

if __name__ == '__main__':
    logger.info("Starting server")
    port = int(os.environ.get('PORT', 8080))
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=port, debug=True)