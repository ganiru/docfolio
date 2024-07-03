import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# TODO: write a method for deleting uploaded files
def delete_all_files():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
        print("Removed ", file_path)
        os.rmdir(file_path)



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Chroma client
chroma_client = chromadb.Client()
collection = [] #chroma_client.create_collection(name="pdf_collection")

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings()
llm = ChatGroq(
    groq_api_key="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",
    model_name="mixtral-8x7b-32768"  # You can change this to another available Groq model if needed
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"File saved to {file_path}")
            # Process and store the PDF in Chroma
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            print(f"Loaded {len(texts)} chunks from the PDF")
            vectorstore = Chroma.from_documents(texts, embeddings, collection_name=filename)
            
            print(f"File uploaded successfully")
            return jsonify({"message": "File uploaded successfully", "filename": filename}), 200
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    print(f"Some other error occurred")
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')
    filename = data.get('filename')
    
    if not query or not filename:
        return jsonify({"error": "Missing query or filename"}), 400
    
    vectorstore = Chroma(collection_name=filename, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    # Create a prompt template
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Create the retrieval chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Execute the chain
    result = chain.invoke(query)
    
    # For simplicity, we're not returning source documents here.
    # If you need source documents, you'll need to modify the chain and result handling.
    return jsonify({
        "result": result,
    }), 200

@app.route('/documents', methods=['GET', 'OPTIONS'])
def get_documents():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    collections = chroma_client.list_collections()
    # collectionList = []
    # if collections:
        # return jsonify({"error": "No collections found"}), 404
    #    if len(collections) == 1:
    #        collectionList.append(collections[0].name)
    #    else:
    #        for collection in collections:
     #           collectionList.append(collection.name)
    #else:
    print("Found collections: ", collections)
    # [collection.name for collection in collections]
    return jsonify({"documents": [collection.name for collection in collections]}), 200


@app.route('/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    try:
        chroma_client.delete_collection(filename)
        return jsonify({"message": f"Document {filename} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 400

def handle_options_request():
    response = app.make_default_options_response()
    headers = {
        'Access-Control-Allow-Origin': 'http://localhost:5500',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Credentials': 'true'
    }
    response.headers.extend(headers)
    return response

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)