# SEO Q&A Chatbot

This Python code implements a chatbot that answers questions related to SEO.  The chatbot is based on OpenAI's GPT-3.5 language model, which is a powerful and versatile natural language processing model.
Sources used:
- seobythesea.com
- holisticseo.digital
- guest posts published by Koray Tuğberk GÜBÜR on other publications.

## Dependencies

This code uses the following libraries:
- `streamlit`: for building the user interface.
- `pinecone`: for retrieving relevant text chunks based on a user's question.
- `openai`: for generating responses to user questions.
- `streamlit_chat`: for displaying chat history in the user interface.

To install these libraries, use the following command:
```
pip install streamlit pinecone openai streamlit_chat
```

## Usage

To run this code, first set the `PINECONE_API_KEY` and `OPEN_AI_API_KEY` environment variables with your Pinecone and OpenAI API keys, respectively. You can get an OpenAI API key by creating an account on the OpenAI website.

Then, run the following command:
```
streamlit run streamlit_app.py
```

This will start the Streamlit server, and you can access the chatbot by opening a web browser and navigating to `http://localhost:8501`.

## How it Works

The chatbot works as follows:
1. The user enters a question in the input field.
2. The chatbot retrieves relevant text chunks based on the user's question using the Pinecone similarity search service.
3. The chatbot adds the user's question to the retrieved text chunks to create an augmented query.
4. The chatbot generates a response to the augmented query using OpenAI's GPT-3.5 (Chat GPT) language model.
5. The chatbot displays the response to the user, along with the chat history.

The chat history is saved in the `st.session_state` dictionary, which is a dictionary that persists across Streamlit sessions. The `message` function from the `streamlit_chat` library is used to display the chat history in the user interface.
