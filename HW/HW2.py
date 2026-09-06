import requests
import streamlit as st
from anthropic import Anthropic
from bs4 import BeautifulSoup
from openai import OpenAI


def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


st.title("URL Summarizer with Multiple LLMs")

url = st.text_input("Web page URL")

summary_type = st.sidebar.selectbox(
    "Select the type of summary",
    (
        "100 words",
        "2 connecting paragraphs",
        "5 bullet points",
    ),
)

output_language = st.selectbox(
    "Select the output language",
    (
        "English",
        "French",
        "Spanish",
    ),
)

llm_name = st.sidebar.selectbox(
    "Select an LLM",
    (
        "OpenAI",
        "Anthropic",
    ),
)

use_advanced_model = st.sidebar.checkbox("Use advanced model")

if llm_name == "OpenAI":
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        client.models.list()
    except Exception:
        st.error("Invalid OpenAI key. Check secrets.toml for a valid key.")
        st.stop()

    model_name = "gpt-5-mini" if use_advanced_model else "gpt-5-nano"

else:
    try:
        client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        client.models.list()
    except Exception:
        st.error("Invalid Anthropic key. Check secrets.toml for a valid key.")
        st.stop()

    model_name = (
        "claude-sonnet-5"
        if use_advanced_model
        else "claude-haiku-4-5-20251001"
    )

if url:
    document = read_url_content(url)

    if document:
        prompt = (
            f"Summarize the following web page in {summary_type}. "
            f"Write the summary in {output_language}.\n\n{document}"
        )

        if llm_name == "OpenAI":
            messages = [{"role": "user", "content": prompt}]

            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
            )

            st.write_stream(stream)

        else:
            with client.messages.stream(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                st.write_stream(stream.text_stream)