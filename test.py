from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage

OPEN_AI_API_KEY = "aaaa"


llm = OpenAI(
    streaming=True,
    openai_api_key=OPEN_AI_API_KEY,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0,
)
resp = llm("Write me a song about sparkling water.")
llm.generate(["Tell me a joke."])
