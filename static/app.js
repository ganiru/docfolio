/* tailwind.config = {
  theme: {
    extend: {
      colors: {
        brown: {
          50: "#fdf8f6",
          100: "#f2e8e5",
          200: "#eaddd7",
          300: "#e0cec7",
          400: "#d2bab0",
          500: "#bfa094",
          600: "#a18072",
          700: "#977669",
          800: "#846358",
          900: "#43302b",
        },
        green: {
          50: "#f0fdf4",
          100: "#dcfce7",
          200: "#bbf7d0",
          300: "#86efac",
          400: "#4ade80",
          500: "#22c55e",
          600: "#16a34a",
          700: "#15803d",
          800: "#166534",
          900: "#14532d",
          950: "#052e16",
        },
        emerald: {
          50: "#ecfdf5",
          100: "#d1fae5",
          200: "#a7f3d0",
          300: "#6ee7b7",
          400: "#34d399",
          500: "#10b981",
          600: "#059669",
          700: "#047857",
          800: "#065f46",
          900: "#064e3b",
          950: "#022c22",
        },
        teal: {
          50: "#f0fdfa",
          100: "#ccfbf1",
          200: "#99f6e4",
          300: "#5eead4",
          400: "#2dd4bf",
          500: "#14b8a6",
          600: "#0d9488",
          700: "#0f766e",
          800: "#115e59",
          900: "#134e4a",
          950: "#042f2e",
        },
        slate: {
          50: "#f8fafc",
          100: "#f1f5f9",
          200: "#e2e8f0",
          300: "#cbd5e1",
          400: "#94a3b8",
          500: "#64748b",
          600: "#475569",
          700: "#334155",
          800: "#1e293b",
          900: "#0f172a",
          950: "#020617",
        },
      },
    },
  },
}; */
document.addEventListener("DOMContentLoaded", function () {
  const fileUpload = document.getElementById("fileUpload");
  const documentList = document.getElementById("documentList");
  const chatWindow = document.getElementById("chatWindow");
  const queryInput = document.getElementById("queryInput");
  const sendQuery = document.getElementById("sendQuery");

  const apiURL = window.location.origin;

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
          document.getElementById("documentsHeader").style.display = "none";
          const docElement = document.createElement("div");
          docElement.className =
            "flex justify-center p-2 bg-gray-200 rounded mb-2";
          docElement.innerHTML = `<span>No documents found</span>`;
          documentList.appendChild(docElement);
        } else {
          document.getElementById("documentsHeader").style.display = "block";

          let documentLabel = "";
          if (data.documents.length === 1) {
            documentLabel = "Select the document to start chatting";
          } else {
            documentLabel = `${data.documents.length} documents. Select one or more to start chatting`;
          }
          document.getElementById("documentsHeader").innerText = documentLabel;

          data.documents.forEach((doc) => {
            const docElement = document.createElement("div");
            // save the filename in a variable and truncate it if it's more than 20 characters
            const filename =
              doc.length > 20 ? doc.substring(0, 20) + "..." : doc;
            deleteSVG =
              '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash-2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>';
            docElement.className =
              "flex justify-between items-center p-2 bg-gray-200 rounded mb-2 cursor-pointer";
            docElement.innerHTML = `
                <span style="cursor: pointer" title='${doc}'>${filename}</span>
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

      if (span && currentDocuments.includes(span.title)) {
        doc.classList.add("bg-gray-400");
      } else {
        doc.classList.remove("bg-gray-400");
      }
    });
    document.getElementById(
      "selectedDocument"
    ).innerHTML = `Selected document(s): <b>${
      currentDocuments.join(", ") || "None"
    }</b>`;
  }

  fileUpload.addEventListener("change", function (event) {
    event.preventDefault();
    const files = event.target.files;
    if (files.length > 0) {
      // Show loading indicator
      document.getElementById("loading").style.display = "flex";
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
          // Show loading indicator
          document.getElementById("loading").style.display = "none";
          updateDocumentList();
          fileUpload.value = ""; // Reset file input
        })
        .catch((error) => {
          console.error("Error uploading:", error);
          document.getElementById("loading").style.display = "none";
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
