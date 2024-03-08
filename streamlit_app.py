import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import datetime
from deta import Deta

# Initialize Pinecone client with the API key
pinecone_client = Pinecone(api_key=st.secrets["API"]["PINECONE_API_KEY"])

PINECONE_INDEX_NAME = st.secrets["API"]["PINECONE_INDEX_NAME"]
openai.api_key = st.secrets["API"]["OPEN_AI_API_KEY"]
deta = Deta(st.secrets["API"]["DETA_KEY"])


def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def add_user_message_to_session(prompt):
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


def generate_assistant_response(augmented_query):
    primer = f"""
Your task is to answer user questions based on the information given above each question.It is crucial to cite sources accurately by using the [[number](URL)] notation after the reference. Say "I don't know" if the information is missing and be as detailed as possible. End each sentence with a period. Please begin.
              """
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query},
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )
    return full_response


def get_query_embedding(query):
    # Embed the query using OpenAI's text-embedding-ada-002 engine
    query_embedding = openai.Embedding.create(
        input=[query], engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    return query_embedding


def get_relevant_contexts(query_embedding, index_name):
    # Ensure the index exists or create it if it doesn't
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=len(query_embedding),
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )

    # Connect to the existing index
    index = pinecone_client.Index(index_name=index_name)
    
    res = index.query(query_embedding, top_k=6, include_metadata=True)
    contexts = []
    for item in res["matches"]:
        metadata = item["metadata"]
        text = metadata.get("text", "")
        url = metadata.get("url", "")
        title = metadata.get("title", "")
        relevance_score = item.get("score", "")
        context = {
            "search_results_text": text,
            "search_results_url": url,
            "search_results_title": title,
            "search_relevance_score": relevance_score,
        }
        contexts.append(context)

    contexts = str(contexts)
    return contexts


def augment_query(contexts, query):
    augmented_query = (
        f"###Search Results: \n{contexts} #End of Search Results\n\n-----\n\n {query}"
    )
    return augmented_query


def add_to_database(query, response):
    db = deta.Base("topical_q_a")
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    db.put({"query": query, "response": response, "timestamp": timestamp})


def print_markdown_from_file(file_path):
    with open(file_path, "r") as f:
        markdown_content = f.read()
        st.markdown(markdown_content)


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


def main():
    st.title("SEO Q&A Chatbot")
    st.write(
        "“The great aim of education is not knowledge but action.” ― Herbert Spencer"
    )

    hide_streamlit_header_footer()
    display_existing_messages()

    query = st.chat_input("Ask any question related to SEO")
    if query:
        add_user_message_to_session(query)
        query_embedding = get_query_embedding(query)
        contexts = get_relevant_contexts(query_embedding, index_name=PINECONE_INDEX_NAME)
        augmented_query = augment_query(contexts, query)
        response = generate_assistant_response(augmented_query)
        add_to_database(query, response)
    with st.sidebar:
        print_markdown_from_file("case_studies.md")


if __name__ == "__main__":
    main()
