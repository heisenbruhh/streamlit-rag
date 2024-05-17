import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import PyPDF2
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
import tkinter as tk
from tkinter import filedialog
# Define the prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

n_gpu_layers = 35
n_batch = 30

# Load the model of choice
def load_llm():
    llm = LlamaCpp(
        model_path="F:/Aakash/models/mistral-7b-instruct-v0.2.Q8_0.gguf",
        max_tokens=512,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
        n_ctx=10000,
        stop=['USER:'],
        temperature=0.4,
    )
    return llm

# Set the title for the Streamlit app
st.title("L&T Chat")

# Initialize data storage

def select_folder():
   root = tk.Tk()
   root.attributes('-topmost',True)
   root.withdraw()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path
data = []
selected_folder_path = st.session_state.get("folder_path", None)
folder_select_button = st.sidebar.button("Select Folder")
if folder_select_button:
  selected_folder_path = select_folder()
  st.session_state.folder_path = selected_folder_path
print(selected_folder_path)

uploaded_files = ''
# Handle file upload
if selected_folder_path:
    loader = PyPDFDirectoryLoader(selected_folder_path)
    data = loader.load()
    # if uploaded_file.type == "text/plain":
    #     data.append(uploaded_file.read().decode())
    # elif uploaded_file.type == "application/pdf":
    #     pdf_reader = PyPDF2.PdfReader(uploaded_file)
    #     text = "".join([page.extract_text() for page in pdf_reader.pages])
    #     data.append(text)
    # elif uploaded_file.type == "text/csv":
    #     df = pd.read_csv(uploaded_file)
    #     data.append(df.to_string())
    # print(data);
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)    
    # print(text);
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Load the language model
    llm = load_llm()

    # Create a conversational chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=new_db.as_retriever())

    # Function for conversational chat
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me (LLAMA2) about " + ", ".join([file.name for file in uploaded_files])]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your data", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Me")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
else:
    st.write("No files uploaded yet.")

