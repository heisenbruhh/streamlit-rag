# Import necessary libraries
import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch

print(torch.cuda.is_available())

# Define the path for generated embeddings
# DB_PATH = 'F:\Aakash\dbstore'


template =  """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Helpful Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context","question"])

n_gpu_layers = 80  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Load the model of choice
def load_llm():

    llm = LlamaCpp(
    model_path="F:\Aakash\models\TheBloke\Llama-2-7B-Chat-GGUF\llama-2-7b-chat.Q8_0.gguf",
    max_tokens=1024,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
    n_ctx=4096, # Context window
    stop = ['USER:'], # Dynamic stopping when such token is detected.
    temperature = 0.4,
    )
    return llm

# Set the title for the Streamlit app
st.title("L&T Chat")

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="txt")

# Handle file upload
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load CSV data using CSVLoader
    data = [uploaded_file.read().decode()]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(data)
    # Create embeddings using Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

    # Create a FAISS vector store and save embeddings
    db = Chroma.from_documents(texts, embeddings)
    # db.save_local(DB_PATH)

    # Load the language model
    llm = load_llm()

    # Create a conversational chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # Function for conversational chat
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        # st.session_state['history'].append((query, "Hello"))
        
        return result["answer"]

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me(LLAMA2) about " + uploaded_file.name ]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey !"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your data ", key='input')
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

