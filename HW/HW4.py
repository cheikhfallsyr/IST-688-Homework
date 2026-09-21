import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup


# Use pysqlite3 when it is available on Streamlit Community Cloud.
# Codespaces can use Python's normal sqlite3 module if it is unavailable.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb


# Create the OpenAI client.
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(
        api_key=st.secrets.OPENAI_API_KEY
    )


# Create embeddings and add the HTML chunks to ChromaDB.
def add_to_collection(
    collection,
    documents,
    document_ids,
    metadatas,
):
    client = st.session_state.openai_client
    embeddings = []

    # Process 100 chunks at a time because the folder contains
    # hundreds of HTML files.
    for batch_start in range(0, len(documents), 100):
        batch = documents[
            batch_start:batch_start + 100
        ]

        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small",
        )

        embeddings.extend(
            item.embedding for item in response.data
        )

    collection.add(
        documents=documents,
        ids=document_ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# Extract readable text from one HTML file.
def extract_text_from_html(html_path):
    with open(
        html_path,
        "r",
        encoding="utf-8",
    ) as html_file:
        soup = BeautifulSoup(
            html_file,
            "html.parser",
        )

    # Remove code that does not contain useful organization information.
    for element in soup(["script", "style"]):
        element.decompose()

    return " ".join(
        soup.get_text(
            " ",
            strip=True,
        ).split()
    )


# Chunking method:
# Each document is divided at the midpoint of its words.
# This method was chosen because the homework requires exactly two
# mini-documents for every HTML file. It also creates two chunks that
# are approximately the same size.
def chunk_document(text):
    words = text.split()
    midpoint = (len(words) + 1) // 2

    return [
        " ".join(words[:midpoint]),
        " ".join(words[midpoint:]),
    ]


# Load every HTML file and create two chunks per file.
def load_html_to_collection(
    folder_path,
    collection,
):
    folder = Path(folder_path)

    html_files = sorted(
        list(folder.glob("*.html"))
        + list(folder.glob("*.htm"))
    )

    if not html_files:
        st.error(
            "The su_orgs folder must contain the provided HTML files."
        )
        st.stop()

    documents = []
    document_ids = []
    metadatas = []

    for html_file in html_files:
        text = extract_text_from_html(
            html_file
        )

        if not text:
            continue

        chunks = chunk_document(text)

        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):
            if chunk:
                documents.append(chunk)

                document_ids.append(
                    f"{html_file.name}-chunk-{chunk_number}"
                )

                metadatas.append(
                    {
                        "source_file": html_file.name,
                        "chunk_number": chunk_number,
                    }
                )

    add_to_collection(
        collection,
        documents,
        document_ids,
        metadatas,
    )


# Create the persistent database only if it is empty.
def create_vector_database():
    chroma_client = chromadb.PersistentClient(
        path="./ChromaDB_for_HW4"
    )

    collection = (
        chroma_client.get_or_create_collection(
            "HW4Collection"
        )
    )

    if collection.count() == 0:
        load_html_to_collection(
            "./su_orgs/",
            collection,
        )

    st.session_state.HW4_VectorDB = (
        collection
    )


if "HW4_VectorDB" not in st.session_state:
    with st.spinner(
        "Creating the student organization database..."
    ):
        create_vector_database()


#### MAIN APP ####

st.title(
    "HW 4: iSchool Student Organization Chatbot"
)

st.write(
    "Ask a question about Syracuse University student organizations."
)


# Create the conversation history.
if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [
        {
            "role": "assistant",
            "content": (
                "What would you like to know about "
                "the student organizations?"
            ),
        }
    ]


# Display previous messages.
for message in st.session_state.hw4_messages:
    with st.chat_message(
        message["role"]
    ):
        st.write(
            message["content"]
        )


# Get the user's next message.
if prompt := st.chat_input(
    "Ask about a student organization"
):
    st.session_state.hw4_messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    client = (
        st.session_state.openai_client
    )

    # Create an embedding for the user's question.
    response = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small",
    )

    query_embedding = (
        response.data[0].embedding
    )

    # Retrieve the three closest chunks.
    results = (
        st.session_state.HW4_VectorDB.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=min(
                3,
                st.session_state.HW4_VectorDB.count(),
            ),
        )
    )

    # Combine the retrieved chunks for the LLM.
    rag_context = "\n\n---\n\n".join(
        (
            f"Source: {metadata['source_file']} "
            f"(chunk {metadata['chunk_number']})"
            f"\n\n{document}"
        )
        for document, metadata in zip(
            results["documents"][0],
            results["metadatas"][0],
        )
    )

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful Syracuse University "
            "student organization chatbot. Answer the "
            "user's question using only the retrieved "
            "organization pages below. Name the source "
            "HTML file or files used in your answer. "
            "If the information is unavailable, say that "
            "you could not find it. Do not make up "
            "information.\n\n"
            "Retrieved organization pages:\n\n"
            + rag_context
        ),
    }

    # Keep no more than the last five interactions.
    user_message_indexes = [
        index
        for index, message in enumerate(
            st.session_state.hw4_messages
        )
        if message["role"] == "user"
    ]

    buffer_start = (
        user_message_indexes[-5]
        if len(user_message_indexes) >= 5
        else user_message_indexes[0]
    )

    conversation_buffer = (
        st.session_state.hw4_messages[
            buffer_start:
        ]
    )

    messages_for_model = [
        system_message
    ] + conversation_buffer

    # Generate and stream the response.
    stream = (
        client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages_for_model,
            stream=True,
        )
    )

    with st.chat_message(
        "assistant"
    ):
        answer = st.write_stream(
            stream
        )

    st.session_state.hw4_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )