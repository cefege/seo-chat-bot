import datetime
import streamlit as st
import pinecone
import openai
from streamlit_chat import message
from deta import Deta


PINECONE_API_KEY = st.secrets["API"]["PINECONE_API_KEY"]
OPEN_AI_API_KEY = st.secrets["API"]["OPEN_AI_API_KEY"]
deta = Deta(st.secrets["API"]["DETA_KEY"])


def generate_output_string(list_of_dicts):
    result_string = ""
    for item in list_of_dicts:
        result_string += "title: {}\nurl: {}\ntext: {}\n-----\n".format(
            item["title"], item["url"], item["text"]
        )
    return result_string


def get_relevant_contexts(query_embedding, index):
    index = pinecone.Index(index_name=index)
    res = index.query(query_embedding, top_k=8, include_metadata=True)
    contexts = []
    for item in res["matches"]:
        metadata = item["metadata"]
        text = metadata.get("text", "")
        url = metadata.get("url", "")
        title = metadata.get("title", "")
        context = {"text": text, "url": url, "title": title}
        contexts.append(context)

    contexts = str(contexts)
    return contexts


def augment_query(contexts, query):
    augmented_query = f"{contexts} \n\n-----\n\n {query}"
    return augmented_query


def get_query_embedding(query):
    """
    This function takes a query string as input and returns its embedding as computed by OpenAI's text-embedding-ada-002 engine.

    Args:
    query (str): The query to be embedded.

    Returns:
    query_embedding (list): A list containing the embedding vector of the query.
    """

    # Embed the query using OpenAI's text-embedding-ada-002 engine
    query_embedding = openai.Embedding.create(
        input=[query], engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    return query_embedding


def run_chatbot(augmented_query):
    primer = f"""
Your task is to answer user questions based on the information given above each question. Mention refferance URLs at the end. Say "I don't know" if the information is missing and be as detailed as possible. End each sentence with a period. Please begin.
              """

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query},
        ],
    )

    return res["choices"][0]["message"]["content"]


def add_to_database(query, response):
    db = deta.Base("topical_q_a")
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    db.put({"query": query, "response": response, "timestamp": timestamp})


def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


def print_markdown_from_file(file_path):
    with open(file_path, "r") as f:
        markdown_content = f.read()
        st.markdown(markdown_content)


def main():
    hide_streamlit_header_footer()
    st.title("SEO Q&A Chatbot")
    st.write("Ask any question related to SEO")

    # Input field for user's question
    query = st.text_input("Enter your question")

    if query:
        # retrieve from Pinecone
        query_embedding = get_query_embedding(query)

        # get relevant contexts (including the questions)
        contexts = get_relevant_contexts(query_embedding, index="gpt-4-langchain-docs")

        augmented_query = augment_query(contexts, query)

        # ------------------------------------------------------------------
        # Save history

        # This is used to save chat history and display on the screen
        if "answer" not in st.session_state:
            st.session_state["answer"] = []

        if "question" not in st.session_state:
            st.session_state["question"] = []

        # ------------------------------------------------------------------
        # Display the current response. Chat history is displayed below

        # Get the resonse from LLM
        # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
        # stuff chain type sends all the relevant text chunks from the document to LLM
        response = run_chatbot(augmented_query)
        # Add the question and the answer to display chat history in a list
        # Latest answer appears at the top
        st.session_state.question.insert(0, query)
        st.session_state.answer.insert(0, response)

        add_to_database(query, response)

        # Display the chat history
        for i in range(len(st.session_state.question)):
            message(st.session_state["question"][i], is_user=True, key=f"question_{i}")
            message(st.session_state["answer"][i], is_user=False, key=f"answer_{i}")
    print_markdown_from_file("case_studies.md")


if __name__ == "__main__":
    openai.api_key = OPEN_AI_API_KEY
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west4-gcp")
    main()
