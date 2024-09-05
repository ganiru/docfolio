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
        console.log({ data });
        if (data.documents.length === 0) {
          //document.getElementById("documentsHeader").style.display = "none";
          const docElement = document.createElement("div");
          docElement.className =
            "flex justify-center p-2 text-gray-600 rounded mb-2";
          docElement.innerHTML = `<span>No documents</span>`;
          documentList.appendChild(docElement);
        } else {
          // document.getElementById("documentsHeader").style.display = "block";

          let documentLabel = "";
          if (data.documents.length > 1) {
            documentLabel = `${data.documents.length} documents.`; // Select one or more to start chatting`;
          } /* else {
            documentLabel = "Select the document to start chatting";
            } */
          // document.getElementById("documentsHeader").innerText = documentLabel;

          data.documents.forEach((doc) => {
            const docElement = document.createElement("div");
            // save the filename in a variable and truncate it if it's more than 20 characters
            const filename =
              doc.filename.length > 33
                ? doc.filename.substring(0, 33) + "..."
                : doc.filename;
            deleteSVG =
              '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash-2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>';
            // Convert bytes to kilobytes or megabytes
            let file_size = "";
            if (doc.file_size) {
              const sizeInKB = doc.file_size / 1024;
              const sizeInMB = sizeInKB / 1024;
              file_size =
                sizeInKB < 1024
                  ? `${sizeInKB.toFixed(2)} KB`
                  : `${sizeInMB.toFixed(2)} MB`;
            }
            let total_pages = doc.total_pages
              ? `, ${doc.total_pages} pages`
              : "";

            docElement.innerHTML = `
               <div class="flex align-center gap-4 bg-[#f8fafb] px-4 py-3">
                    <div class="flex flex-1 flex-col justify-center">
                        <p class="text-[#0e141b] text-base font-medium leading-normal" title="${
                          doc.filename
                        }">${filename}</p>
                        <p class="text-[#4f7396] text-sm font-normal leading-normal">${
                          doc.created_date && file_size
                            ? doc.created_date + ", " + file_size
                            : ""
                        }${total_pages}</p>
                    </div>
                    <svg class="delete-button" data-filename="${
                      doc.filename
                    }" style="width:20px;" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-6">
                      <path stroke-linecap="round" stroke-linejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                    </svg>

                </div>
              `;
            docElement
              .querySelector(".delete-button")
              .addEventListener("click", deleteDocument); /* */
            //docElement.addEventListener("click", () => toggleDocument(doc));
            documentList.appendChild(docElement);
          });
        }
      })
      .catch((error) => console.error("Error:", error));
  }

  function deleteDocument(event) {
    event.stopPropagation();
    const filename = event.target.getAttribute("data-filename");
    console.log("deleting", filename);
    fetch(`${apiURL}/delete/${filename}`, {
      method: "DELETE",
      credentials: "include",
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data.message);
        updateDocumentList();
        currentDocuments = currentDocuments.filter((doc) => doc !== filename);
        // updateSelectedDocuments();
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
    //updateSelectedDocuments();
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
    const query = queryInput.value;
    if (query.trim() === "") {
      alert("Please enter a query.");
      return;
    }
    console.log("Sending query:", query);
    const messageElement = document.createElement("article");
    const currentTime = new Date().toLocaleString("en-US", {
      month: "numeric",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      hour12: true,
    });
    const datenow = Date.now();

    messageElement.className = "mb-2";
    messageElement.innerHTML = `<p><span class="font-bold">You</span> <span style='color:gray'>${currentTime}</span><div>${query}</div></p>`;
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    queryInput.value = "";

    const responseElement = document.createElement("article");
    responseElement.className = "mb-2";
    responseElement.innerHTML = `<p><span class="font-bold">Bot</span> 
    <span style='color:gray' class='text-sm' id='bot-response-timestamp-${datenow}'></span> 
    <div class="bot-response" style="white-space: pre-wrap;"></div>
    </p>`;
    chatWindow.appendChild(responseElement);

    const botResponse = responseElement.querySelector(".bot-response");

    // Show loading indicator
    const loadingIndicator = document.createElement("span");
    loadingIndicator.className = "loading-indicator";
    loadingIndicator.textContent = "Thinking...";
    botResponse.appendChild(loadingIndicator);

    // Send request to server
    fetch("/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        filenames: currentDocuments,
      }),
    })
      .then((response) => {
        console.log("Received response:", response);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        // clear the 'thinking'
        loadingIndicator.remove();
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let markdown = "";

        function readStream() {
          reader
            .read()
            .then(({ done, value }) => {
              if (done) {
                // Convert markdown to HTML
                var converter = new showdown.Converter();
                botResponse.innerHTML = converter.makeHtml(markdown);

                // Render final markdown
                // botResponse.innerHTML = markdown;
                chatWindow.scrollTop = chatWindow.scrollHeight;
                return;
              }
              let chunk = decoder.decode(value);
              markdown += chunk;

              // Render markdown as it comes in
              botResponse.innerHTML = markdown;
              chatWindow.scrollTop = chatWindow.scrollHeight;

              // Update timestamp
              const currentTime = new Date().toLocaleString("en-US", {
                month: "numeric",
                day: "numeric",
                year: "numeric",
                hour: "numeric",
                minute: "numeric",
                second: "numeric",
                hour12: true,
              });
              const botResponseTimeStamp = document.getElementById(
                `bot-response-timestamp-${datenow}`
              );
              botResponseTimeStamp.innerHTML = currentTime;

              readStream();
            })
            .catch((error) => {
              console.error("Error reading stream:", error);
              botResponse.innerHTML += `Error: ${error.message}`;
            });
        }

        readStream();
      })
      .catch((error) => {
        console.error("Error:", error);
        loadingIndicator.remove();
        botResponse.innerHTML = `Error: ${error.message}`;
        botResponse.classList.add("error-message");
      });
  }
  updateDocumentList();
});
