import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import chroma
import tempfile

#sk for openai
os.environ['OPENAI_API_KEY'] = ''

#load the desired llm
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")


st.title('PDF Summariser')
st.subheader('Upload your pdf and let GPT summarise it for you')


#load the pdf file and return a temporary path for it 

#note streamlit loads files in as bytes
uploaded_file = st.file_uploader('', type=(['pdf']))

        
if uploaded_file is not None:

    #save the uploaded file to a temporary location so it can be loaded with a pdfloader

    temp_location = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_location.name, uploaded_file.name)

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.write("Full path of the uploaded file:", temp_file_path)


    #load pdf file and split pages
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    #create embeddings
    embeddings = OpenAIEmbeddings()

    #load documents into vector database 
    db = chroma.from_documents(pages, embeddings)

    #create LC chain that summarises 
    #the chain type refine generates responses by iterively updating answer by analysing each document
    chain = load_summarize_chain(llm, chain_type="refine")

    #run the chain
    search = db.similarity_search(" ")
    summary = chain.run(input_documents=search, question="Write a summary of the text.")

    #write the output to streamlit
    st.write(summary) 

    


