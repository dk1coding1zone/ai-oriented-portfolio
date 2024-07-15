import json
from PIL import Image
import requests  # pip install requests
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

selected = option_menu(
    menu_title=None,  # required
    options=["😊 Welcome to my Portfolio Site"],  # required
    icons=["house"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
)

st.subheader("Hi, I am Deepak :wave:")
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as brief as possible from the provided context, make sure to provide all the details answer length 
    should not be more than 150 words, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

def main():
    st.text("Chat to know more about me💁")

    user_question = st.text_input("Ask a Question e.g Tell me about rohit")

    if user_question:
        # Load or create FAISS index
        index_path = "faiss_index"
        if not os.path.exists(index_path):
            path = os.path.join(os.path.dirname(__file__), "Rohit_Resume.pdf")
            raw_text = get_pdf_text(path)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)

        user_input(user_question)

if __name__ == "__main__":
    main()
st.title("A Data Analyst From India")
st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry.")
st.write("[Learn More >](https://google.com)")

# what I do 
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What I do")
        st.write(
            '''
            Contrary to popular belief, Lorem Ipsum is not simply random text. 
            - It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. 
            - Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words,
            - consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source.
            - Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 
            - 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance.
            '''
        )

st.write("[YouTube Channel >](https://youtube.com)")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    

lottie_coding = load_lottiefile("lottiefile.json")  # replace link to local lottie file
with right_column:
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",  # Set quality to high
        height=300,  # Adjust height for better visibility
        width=300,  # Adjust width for better visibility
        key="coding_lottie",
        )
image_1=Image.open("images/img1.png")
image_2=Image.open("images/img2.png")
with st.container():
    st.write("---")
    st.header("My Projects")
    st.write("##")
    image_column,text_column=st.columns((1,2))
    with image_column:
        st.image(image_1)
    with text_column:
        st.subheader("Intreagate Lottie Animation Inside Your Streamlit App")
        st.write(
            '''
         - It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. 
         - Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words,
         - consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source.
            '''
        )
        st.markdown("[YouTube Channel >](https://youtube.com)")
    with st.container():
            image_column,text_column=st.columns((1,2))
    with image_column:
        st.image(image_2)
    with text_column:
        st.subheader("Intreagate Lottie Animation Inside Your Streamlit App")
        st.write(
            '''
         - It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. 
         - Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words,
         - consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source.
            '''
        )
        st.markdown("[YouTube Channel >](https://youtube.com)")
    with st.container():
            image_column,text_column=st.columns((1,2))
    with image_column:
        st.image(image_2)
    with text_column:
        st.subheader("Intreagate Lottie Animation Inside Your Streamlit App")
        st.write(
            '''
         - It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. 
         - Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words,
         - consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source.
            '''
        )
        st.markdown("[YouTube Channel >](https://youtube.com)")

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
contact_form = """
<form action="https://formsubmit.co/deepaksinghnitsurat@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="text" name="name" placeholder="Your name" required style="width: 100%; padding: 10px; margin: 5px 0;">
    <input type="email" name="email" placeholder="Your email" required style="width: 100%; padding: 10px; margin: 5px 0;">
    <textarea name="message" placeholder="Your message here" required style="width: 100%; padding: 10px; margin: 5px 0;"></textarea>
    <button type="submit" style="width: 100%; padding: 10px; margin: 5px 0; background-color: #4CAF50; color: white; border: none; cursor: pointer;">Send</button>
</form>
"""

left_column, right_column = st.columns(2)
with left_column:
    st.markdown(contact_form, unsafe_allow_html=True)
with right_column:
    st.empty()
