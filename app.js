document.addEventListener("DOMContentLoaded", function () {
  const fileUpload = document.getElementById("fileUpload");
  const documentList = document.getElementById("documentList");
  const chatWindow = document.getElementById("chatWindow");
  const queryInput = document.getElementById("queryInput");
  const sendQuery = document.getElementById("sendQuery");

  const apiHost = `${window.location.protocol}//${window.location.hostname}`;
  const apiPort = "8080";
  const apiURL = `${apiHost}:${apiPort}`;

  let currentDocuments = [];

  function updateDocumentList() {
    fetch(`${apiURL}/documents`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        documentList.innerHTML = "";
        if (data.documents.length === 0) {
          const docElement = document.createElement("div");
          docElement.className =
            "flex justify-center p-2 bg-brown-200 rounded mb-2";
          docElement.innerHTML = `<span>No documents found</span>`;
          documentList.appendChild(docElement);
        } else {
          data.documents.forEach((doc) => {
            const docElement = document.createElement("div");
            deleteSVG =
              '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash-2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>';
            docElement.className =
              "flex justify-between items-center p-2 bg-brown-200 rounded mb-2 cursor-pointer";
            docElement.innerHTML = `
                <span style="cursor: pointer">${doc}</span>
                <button class="delete-doc bg-red-500 text-white px-2 py-1 rounded hover:bg-red-600" data-filename="${doc}">Delete</button>
              `;
            docElement
              .querySelector(".delete-doc")
              .addEventListener("click", deleteDocument);
            docElement.addEventListener("click", () => toggleDocument(doc));
            documentList.appendChild(docElement);
          });
        }
      })
      .catch((error) => console.error("Error:", error));
  }

  function deleteDocument(event) {
    event.stopPropagation();
    const filename = event.target.getAttribute("data-filename");
    fetch(`${apiURL}/delete/${filename}`, {
      method: "DELETE",
      credentials: "include",
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data.message);
        updateDocumentList();
        currentDocuments = currentDocuments.filter((doc) => doc !== filename);
        updateSelectedDocuments();
      })
      .catch((error) => console.error("Error:", error));
  }

  function toggleDocument(filename) {
    const index = currentDocuments.indexOf(filename);
    if (index > -1) {
      currentDocuments.splice(index, 1);
    } else {
      currentDocuments.push(filename);
    }
    updateSelectedDocuments();
  }

  function updateSelectedDocuments() {
    Array.from(documentList.children).forEach((doc) => {
      const span = doc.querySelector("span");
      if (span && currentDocuments.includes(span.textContent)) {
        doc.classList.add("bg-brown-500");
      } else {
        doc.classList.remove("bg-brown-500");
      }
    });
    document.getElementById(
      "selectedDocument"
    ).innerHTML = `Selected documents: <b>${
      currentDocuments.join(", ") || "None"
    }</b>`;
  }

  fileUpload.addEventListener("change", function (event) {
    event.preventDefault();
    const files = event.target.files;
    if (files.length > 0) {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append("file", files[i]);
      }
      fetch(`${apiURL}/upload`, {
        method: "POST",
        body: formData,
        credentials: "include",
        mode: "cors",
      })
        .then(async (response) => {
          const data = await response.json();
          if (!response.ok) {
            throw new Error(
              data.error || `HTTP error! status: ${response.status}`
            );
          }
          return data;
        })
        .then((data) => {
          console.log(data.message);
          updateDocumentList();
          fileUpload.value = ""; // Reset file input
        })
        .catch((error) => {
          console.error("Error uploading:", error);
          alert(`Error uploading file(s): ${error.message}`);
        });
    }
  });

  sendQuery.addEventListener("click", function () {
    sendMessage();
  });

  queryInput.addEventListener("keyup", function (event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  });

  function sendMessage() {
    if (currentDocuments.length === 0) {
      alert("Please select at least one document first.");
      return;
    }

    const query = queryInput.value;
    if (query) {
      const messageElement = document.createElement("div");
      messageElement.className = "mb-2";
      messageElement.innerHTML = `<p class="font-bold">You: ${query}</p>`;
      chatWindow.appendChild(messageElement);
      chatWindow.scrollTop = chatWindow.scrollHeight;
      queryInput.value = "";

      const responseElement = document.createElement("div");
      responseElement.className = "mb-2";
      responseElement.innerHTML = `<p>Bot: </p>`;
      chatWindow.appendChild(responseElement);

      const botResponse = responseElement.querySelector("p");

      fetch(`${apiURL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query, filenames: currentDocuments }),
      })
        .then((response) => {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          function readStream() {
            return reader.read().then(({ done, value }) => {
              if (done) {
                return;
              }
              const chunk = decoder.decode(value, { stream: true });
              botResponse.textContent += chunk;
              chatWindow.scrollTop = chatWindow.scrollHeight;
              return readStream();
            });
          }

          return readStream();
        })
        .catch((error) => {
          console.error("Error:", error);
          botResponse.textContent +=
            "An error occurred while fetching the response.";
        });
    }
  }

  updateDocumentList();
});
