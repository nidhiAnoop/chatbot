import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain


# this function will load our openAi secret key from .env file 
load_dotenv()

def get_response(user_input):
    return 'I dont know it!'

def get_data(website_url):
    # loading website url
    loader = WebBaseLoader(website_url)

    # actually getting html codes and content
    documents = loader.load()

    # splitting data of the website into senctences, pharagraphs, etc
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    # convering data in docs to numerical/binary format for easier understanding for machine learning/ai
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_question(vector_store):
    # we are here talking about the question from the user and we will consider it as chat and we need our model to analyse that message 
    llm = ChatOpenAI()
    # reading numerical data from vector store 
    retrieve_vector = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        # for both human and bot
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}" ),
        ("user", "Given the above conversation, generate a search query to look up.")
    ]
    )
    # this funtion analyses the chat history and sees if the user already talked about a specific topi
    # llm: large language models: that is the machine learning model which we are using to actually analyse
    # retrieve vector
    # prompt: the whole chat template/chat history
    retriever_chain = create_history_aware_retriever(llm, retrieve_vector, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system" , "Answer the user's questions based on the below context: \n{context}"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user", "{input}"),
    ])

    stuffed_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuffed_documents_chain)


st.set_page_config(page_title = "Chatbot" , page_icon = "??")
st.title("AI CHATBOT")

# fix local hosting error(streamlit installation)

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter website url - ")

if website_url is None or website_url == "":
    st.info("Please enter a valid website url.")
    
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content = "Hello, how may I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_data(website_url)
    
    retriever_chain = get_question(st.session_state.vector_store)
    stuffed_chain = get_conversational_rag_chain(retriever_chain)


    user_message = st.chat_input("Type your message here...")
    if user_message is not None and user_message != "":
        response = stuffed_chain.invoke({
            "chat_history" : st.session_state.chat_history,
            "input" : user_message
        })

        st.write(response)
        ai_message = get_response(user_message)
        # session: amazon example
        #st.session_state.chat_history.append(HumanMessage(content = user_message))
        #st.session_state.chat_history.append(AIMessage(content = ai_message))

        retrieved_documents = retriever_chain.invoke({
            "chat_history" : st.session_state.chat_history,
            "input" : user_message
        }
        )


    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Nidhi"):
                st.write(message.content)
