document.addEventListener('DOMContentLoaded', function() {
    const fileUpload = document.getElementById('fileUpload');
    const documentList = document.getElementById('documentList');
    const chatWindow = document.getElementById('chatWindow');
    const queryInput = document.getElementById('queryInput');
    const sendQuery = document.getElementById('sendQuery');

    fetch('http://127.0.0.1:5000/documents').then(res => res.json())
    .then(data => {
      // do something with data
      console.log(data)
    })
    .catch(rejected => {
        console.log(rejected);
    });

    let currentDocument = null;
    const apiURL = 'http://127.0.0.1:5000/';

    function updateDocumentList() {
        fetch(`${apiURL}documents`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
            .then(data => {
                documentList.innerHTML = '';
                if (data.documents.length === 0) {
                    const docElement = document.createElement('div');
                    docElement.className = 'flex justify-center p-2 bg-brown-200 rounded mb-2';
                    docElement.innerHTML = `
                        <span>No documents found</span>
                    `;
                    documentList.appendChild(docElement);
                } else {
                data.documents.forEach(doc => {
                    const docElement = document.createElement('div');
                    docElement.className = 'flex justify-between items-center p-2 bg-brown-200 rounded mb-2 cursor-pointer';
                    docElement.innerHTML = `
                        <span style="cursor: pointer">${doc}</span>
                        <button class="delete-doc bg-red-500 text-white px-2 py-1 rounded hover:bg-red-600" data-filename="${doc}">Delete</button>
                    `;
                    docElement.querySelector('.delete-doc').addEventListener('click', deleteDocument);
                    docElement.addEventListener('click', () => selectDocument(doc));
                    documentList.appendChild(docElement);
                });
            }
            }).catch(error => console.error('Error:', error));
    }

    function deleteDocument(event) {
        const filename = event.target.getAttribute('data-filename');
        fetch(`${apiURL}delete/${filename}`, { 
            method: 'DELETE',
            credentials: 'include',
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
            updateDocumentList();
        })
        .catch(error => console.error('Error:', error));
    }

    function selectDocument(filename) {
        currentDocument = filename;
        Array.from(documentList.children).forEach(doc => {
            doc.classList.remove('bg-brown-500');
            if (doc.querySelector('span').textContent === filename) {
                doc.classList.add('bg-brown-500');
                document.getElementById('selectedDocument').innerHTML = `Selected document: <b>${filename}</b>`;
            }
        });
    }

    fileUpload.addEventListener('change', function(event) {
        event.preventDefault();
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);
            fetch(`${apiURL}upload`, {
                method: 'POST',
                body: formData,
                credentials: 'include',
                mode: 'cors', // Explicitly set CORS mode
            })
            .then(async response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        throw new Error(`HTTP error! status: ${response.status}, message: ${text}`);
                    });
                }
                return await response.json();
            })
            .then(data => {
                console.log(data.message);
                updateDocumentList();
                fileUpload.value = ''; // Reset file input
            })
            .catch(error => {
                console.error('Error uploading:', {error});
                alert(`Error uploading file: ${error.message}`);
            });
        }
    });

    sendQuery.addEventListener('click', function() {
        sendMessage()
    });

    queryInput.addEventListener('keyup', function(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage()
    {
        if (!currentDocument) {
            alert('Please select a document first.');
            return;
        }

        const query = queryInput.value;
        if (query) {
            const messageElement = document.createElement('div');
            messageElement.className = 'mb-2';
            messageElement.innerHTML = `
                <p class="font-bold">You: ${query}</p>
            `;
            chatWindow.appendChild(messageElement);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            queryInput.value = '';
            fetch(`${apiURL}query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, filename: currentDocument }),
                credentials: 'include',
            })
            .then(response => response.json())
            .then(data => {
                const messageElement = document.createElement('div');
                messageElement.className = 'mb-2';
                messageElement.innerHTML = `
                    <p>Bot: ${data.result}</p>
                `;
                chatWindow.appendChild(messageElement);
                chatWindow.scrollTop = chatWindow.scrollHeight;
                queryInput.value = '';
            })
            .catch(error => console.error('Error:', error));
        }
    }

    updateDocumentList();
});