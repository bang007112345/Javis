from elasticsearch import Elasticsearch
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_elasticsearch import ElasticsearchStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key="AIzaSyCZgi7RHUPi1ts0c2fc1bM1aU8abqeJ4H8",
)
# Create Pinecone index
index_name = "chat-history"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

elastic_vector_search = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name=index_name,
    embedding=embeddings,
    es_user="elastic",
    es_password="123456",
)


async def get_message(message: str):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("chat_history"),
            HumanMessage(content=message),
        ]
    )
    # Retrieve the chat history
    chat_history = await get_chat_history(message)

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"message": message, "chat_history": chat_history})

    await save_chat_history(message, response)

    return response


async def save_chat_history(user_message: str, ai_response: str):
    elastic_vector_search.add_texts([user_message, ai_response])


# Retrieve chat history from Pinecone
async def get_chat_history(user_message: str):
    return elastic_vector_search.similarity_search(user_message, k=10)
