import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TEMPERATURE = 0.2


def initialize_chat_history():
    st.session_state.chat_history = [
        AIMessage(content="Hi! I can answer questions about the website you load."),
    ]


def get_vectorstore_from_url(url: str, embedding_model: str):
    loader = WebBaseLoader(url)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    return Chroma.from_documents(chunks, embeddings)


def build_retriever_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Create a concise search query for retrieving information relevant "
                "to the latest user message in this conversation.",
            ),
        ]
    )
    return create_history_aware_retriever(llm, retriever, prompt)


def build_conversation_chain(retriever_chain, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful website assistant. Use the retrieved context to "
                "answer. If the answer is not in context, say you do not know.\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, answer_chain)


def get_response(user_input: str):
    llm = ChatOpenAI(
        model=st.session_state.model_name,
        temperature=st.session_state.temperature,
    )
    retriever_chain = build_retriever_chain(st.session_state.vector_store, llm)
    rag_chain = build_conversation_chain(retriever_chain, llm)
    response = rag_chain.invoke(
        {
            "chat_history": st.session_state.chat_history,
            "input": user_input,
        }
    )
    return response["answer"]


st.set_page_config(page_title="Interactive Website Chatbot", page_icon="🤖")
st.title("Interactive Website Chatbot")

if "chat_history" not in st.session_state:
    initialize_chat_history()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL
if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_TEMPERATURE

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL", value=st.session_state.loaded_url)
    model_name = st.text_input("Chat model", value=st.session_state.model_name)
    embedding_model = st.text_input(
        "Embedding model", value=st.session_state.embedding_model
    )
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=st.session_state.temperature
    )

    load_clicked = st.button("Load / Refresh Website", type="primary")
    clear_chat = st.button("Clear Chat Memory")

if clear_chat:
    initialize_chat_history()

if load_clicked:
    if not website_url:
        st.warning("Please enter a website URL before loading.")
    else:
        with st.spinner("Loading website and building memory..."):
            st.session_state.vector_store = get_vectorstore_from_url(
                website_url, embedding_model
            )
        st.session_state.loaded_url = website_url
        st.session_state.model_name = model_name
        st.session_state.embedding_model = embedding_model
        st.session_state.temperature = temperature
        initialize_chat_history()
        st.success("Website loaded. Ask a question below.")

if st.session_state.vector_store is None:
    st.info("Enter a URL in the sidebar and click 'Load / Refresh Website' to start.")
else:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

    user_query = st.chat_input("Ask about the loaded website...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_response(user_query)
            st.write(answer)

        st.session_state.chat_history.append(AIMessage(content=answer))
