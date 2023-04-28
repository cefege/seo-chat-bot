import streamlit as st
import pinecone
import openai


PINECONE_API_KEY = st.secrets["API"]["PINECONE_API_KEY"]
OPEN_AI_API_KEY = st.secrets["API"]["OPEN_AI_API_KEY"]


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
As an AI Q&A bot, your main function is to provide detailed answers to user questions based on the information provided above each question. It is crucial to cite sources accurately by using the [[number](URL)] notation after the reference. In cases where the search results refer to multiple subjects with the same name, it is important to write separate answers for each subject. If the information cannot be found in the given data, it is important to truthfully respond with "I don't know". All answers must be written in markdown format, and it is recommended to use as much information as possible when responding to questions. Allways finish a sentence. Please commence your operations.
              """

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query},
        ],
    )

    return res["choices"][0]["message"]["content"]


def main():
    st.title("SEO Q&A Chatbot")
    st.write("Ask any question based on the given context.")

    # Input field for user's question
    query = st.text_input("Enter your question")

    if query:
        st.write(query)
        # retrieve from Pinecone
        query_embedding = get_query_embedding(query)

        # get relevant contexts (including the questions)
        contexts = get_relevant_contexts(query_embedding, index="gpt-4-langchain-docs")

        augmented_query = augment_query(contexts, query)

        # Run chatbot
        response = run_chatbot(augmented_query)
        st.markdown(response)
        st.write("Raw Semantic Search Results:")
        st.json(contexts)


if __name__ == "__main__":
    openai.api_key = OPEN_AI_API_KEY
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west4-gcp")
    main()
