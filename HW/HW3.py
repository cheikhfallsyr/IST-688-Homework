import requests
import streamlit as st
from anthropic import Anthropic
from bs4 import BeautifulSoup
from openai import OpenAI


def read_url_content(url):
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


st.title("HW 3: A Streaming Chatbot that Discusses a URL")

st.write(
    "Enter one or two URLs in the sidebar and choose an LLM. The chatbot "
    "uses the URL contents as permanent context, streams each response, and "
    "remembers the most recent six messages, or three user-agent exchanges."
)

url_1 = st.sidebar.text_input("URL 1")
url_2 = st.sidebar.text_input("URL 2 (optional)")

llm_name = st.sidebar.selectbox(
    "Select an LLM",
    (
        "OpenAI (gpt-6-astra)",
        "Anthropic (claude-fable-5-1)",
    ),
)

if not url_1 and not url_2:
    st.info("Enter at least one URL in the sidebar to start chatting.")
    st.stop()

url_contexts = []

if url_1:
    content_1 = read_url_content(url_1)

    if content_1:
        url_contexts.append(f"URL: {url_1}\n\n{content_1}")
    else:
        st.error(f"Unable to read {url_1}")

if url_2:
    content_2 = read_url_content(url_2)

    if content_2:
        url_contexts.append(f"URL: {url_2}\n\n{content_2}")
    else:
        st.error(f"Unable to read {url_2}")

if not url_contexts:
    st.stop()

system_prompt = (
    "You are a helpful question-answering chatbot. Answer questions using "
    "the supplied URL content. Explain every answer in simple language that "
    "a 10-year-old can understand. When the user asks a new question, answer "
    "it and end with exactly: Do you want more info? If the user answers yes, "
    "provide more information about the same topic and end with exactly: Do "
    "you want more info? If the user answers no, reply with exactly: How can "
    "I help you?\n\nURL content:\n\n"
    + "\n\n---\n\n".join(url_contexts)
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I help you?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about the URL content"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Keep only the six most recent conversation messages.
    conversation_buffer = st.session_state.messages[-6:]

    with st.chat_message("assistant"):
        if llm_name.startswith("OpenAI"):
            client = OpenAI(
                api_key=st.secrets["OPENAI_API_KEY"]
            )

            messages_for_model = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ] + conversation_buffer

            stream = client.chat.completions.create(
                model="gpt-6-astra",
                messages=messages_for_model,
                stream=True,
            )

            response = st.write_stream(stream)

        else:
            client = Anthropic(
                api_key=st.secrets["ANTHROPIC_API_KEY"]
            )

            with client.messages.stream(
                model="claude-fable-5-1",
                max_tokens=1024,
                system=system_prompt,
                messages=conversation_buffer,
            ) as stream:
                response = st.write_stream(stream.text_stream)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )