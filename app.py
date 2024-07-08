import os
import shutil
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# PORT = 8080

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

chroma_client = chromadb.Client()

embeddings = HuggingFaceEmbeddings()
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

    for file in files:
        if file.filename == '':
            continue
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                print(f"File saved to {file_path}")

                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                print(f"Loaded {len(texts)} chunks from the PDF")
                Chroma.from_documents(texts, embeddings, collection_name=filename)
                
                uploaded_files.append(filename)
            except Exception as e:
                errors.append(f"Error processing {filename}: {str(e)}")
        else:
            errors.append(f"Invalid file type for {file.filename}")

    if uploaded_files:
        message = "Files uploaded successfully"
        # delete the file from the server after loading it into Chroma
        os.remove(file_path)
        if errors:
            message += f", but with some errors: {'; '.join(errors)}"
        return jsonify({"message": message, "filenames": uploaded_files}), 200
    else:
        return jsonify({"error": f"No valid files were uploaded. Errors: {'; '.join(errors)}"}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')
    filenames = data.get('filenames')
    
    if not query or not filenames:
        return jsonify({"error": "Missing query or filenames"}), 400
    
    def generate():
        combined_retriever = []
        for filename in filenames:
            vectorstore = Chroma(collection_name=filename, embedding_function=embeddings)
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
    
    collections = chroma_client.list_collections()
    return jsonify({"documents": [collection.name for collection in collections]}), 200

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    try:
        chroma_client.delete_collection(filename)
        # file_path = os.path.join(UPLOAD_FOLDER, filename)
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        #     print(f"File {file_path} deleted successfully")
        # else:
        #     print(f"File {file_path} not found")
            
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

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=port, debug=True)