from datetime import datetime
import os
from typing import Dict, List
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
import anthropic

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
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

@app.before_request
def set_client_session():
    # Get the ID from the query parameter
    if 'id' in request.args:
        # Set the user_id in the session
        session['user_id'] = request.args.get('id')


@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    return render_template('index2.html')

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
            print(f"File info {file}")
            filename = sanitize_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Received file: {file.filename}. Saving to {file_path}")
            print(f"Received file: {file.filename}. Saving to {file_path}")
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)
            # Save the file size in a variable called file_size
            file_size = os.path.getsize(file_path)
            # save the file's metadata to a variable called file_metadata
            file_metadata = {
                "filename": filename,
                "user_id": session.get('user_id', 'Not set'),
                "created_date": datetime.now().strftime("%m/%d/%Y %I:%M %p"),
                "file_size": file_size
            }
            logger.info(f"Saved file: {file_path}")
            print(f"Saved file: {file_path}. Metadata: {file_metadata}")

            loader = get_loader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, separators=["\n\n", "\n", " ", ""])
            texts = text_splitter.split_documents(documents)
            
            for text in texts:
                text.metadata["filename"] = file_metadata['filename']
                text.metadata["user_id"] = file_metadata['user_id'] # session.get('user_id', 'Not set')
                text.metadata["created_date"] = file_metadata['created_date'] # datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                text.metadata["file_size"] = file_metadata['file_size'] # file_size
            
            vector_store = SupabaseVectorStore.from_documents(
                documents=texts,
                embedding=embeddings,
                client=supabase,
                table_name="documents",
                query_name="match_documents"
            )
            logger.info(f"Added documents to vector store: {file_path}")
            print(f"Added documents to vector store: {file_path}")

            os.remove(file_path)
            uploaded_files.append(file_metadata)
        else:
            errors.append(f"File type not allowed: {file.filename}")
    

    if uploaded_files:
        # return a comma delimited string of uploaded filenames from the uploaded_files list
        uploaded_files_str = ', '.join([file['filename'] for file in uploaded_files])
        return jsonify({"message": f"Files uploaded and processed successfully: {uploaded_files_str}"}), 200
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

    # response = supabase.table("documents").select("metadata->filename").eq("metadata->>user_id", user_id).execute()
    response = supabase.table("documents").select("metadata").eq("metadata->>user_id", user_id).execute()
    
    #Extract unique metadata from the response
    metadata_list = get_unique_documents(response.data) # [item['metadata'] for item in response.data if 'metadata' in item]
    # Extract unique filenames from the response
    #filenames = list(set(item['filename'] for item in response.data if 'filename' in item))
    
    return jsonify({"documents": metadata_list}), 200

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

def query_document_with_claude():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    data = request.json
    query = data.get('query')
    user_id = session.get('user_id', 'Not set')
    filenames = data.get('filenames', [])

    if not query:
        return jsonify({"error": "No query provided"}), 400

    count_response = supabase.table("documents").select("id", count="exact").execute()
    total_records = count_response.count
    
    logger.info(f"Total records in the vector store: {total_records}")
    
    if total_records == 0:
        return jsonify({"error": "No documents found in the vector store"}), 404
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
        chunk_size=500
    )
    
    # Construct the filter for user_id and filenames
    filter_query = supabase.table("documents").select("id", count="exact")
    filter_query = filter_query.eq("metadata->>user_id", user_id)
    filter_query = filter_query.in_("metadata->>filename", filenames)
    
    # Count filtered records
    filtered_count_response = filter_query.execute()
    filtered_records = filtered_count_response.count
    
    logger.info(f"Filtered records: {filtered_records}")
        
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
    
    # for doc in results:
    #    logger.info(f"Document: {doc.page_content}")
    
    if not results:
        return jsonify({"error": "No matching documents found in similarity search"}), 404
        
    context = "\n".join([doc.page_content for doc in results])
    client = anthropic.Anthropic()
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    }
                ]
            }
        ]
    system_prompt = f"""
        You are an AI assistant for a document chatbot. Your primary function is to answer questions based on the content of various uploaded documents, including PDFs, Word documents, Excel spreadsheets, and text files. These documents have been processed, chunked, and stored in a Supabase database. Your responses should be:
        1. Accurate and directly based on the information provided in the documents
        2. Concise yet comprehensive
        3. Professional in tone

        When answering questions:
        - Use only the information provided in the context. Do not use external knowledge or make assumptions beyond what's explicitly stated in the documents.
        - For factual questions (e.g., word counts, specific data points), ensure your answer is precisely correct based on the provided context. If you're unsure, state that you need to verify the information.
        - If the answer is not contained within the given context, politely state that you don't have enough information to answer the question accurately.
        - If asked about the source of your information, refer to the documents in general terms without specifying file names or types.

        Here is the relevant context from the documents:
        {context}

        Please provide informative responses based on this context. If you need more information or if the question is unclear, ask for clarification.
        """
    stream = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=messages,
        stream=True,
        temperature=0,
        system=system_prompt #f"You are an AI assistant that answers questions based solely on the following content from the Supabase records:\n\n{context}\n\nDo not use any external knowledge. If the answer is not in the content, say so.",
    )
    

    def generate():
        for chunk in stream:
            if chunk.type == 'content_block_delta':
                yield chunk.delta.text

    return Response(stream_with_context(generate()), content_type='text/plain')

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
        temperature=0.1,
        model_name="llama3-8b-8192",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        max_tokens=1024,
    )
    
    template = """You are an AI assistant for a document chatbot. Your primary function is to answer questions based on the content of various uploaded documents, including PDFs, Word documents, Excel spreadsheets, and text files. These documents have been processed, chunked, and stored in a Supabase database.

        Your responses should be:
        1. Accurate and directly based on the information provided in the documents
        2. Concise yet comprehensive
        3. Professional in tone

        When answering questions:
        - Use only the information provided in the context. Do not use external knowledge or make assumptions beyond what's explicitly stated in the documents.
        - If the answer is not contained within the given context, politely state that you don't have enough information to answer the question accurately.
        - If asked about the source of your information, refer to the documents in general terms without specifying file names or types.

        Here is the relevant context from the documents:

        {context}

        Please provide informative responses based on this context. If you need more information or if the question is unclear, ask for clarification.

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
