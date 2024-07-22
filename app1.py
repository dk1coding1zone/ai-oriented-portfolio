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
from langchain_community.vectorstores import FAISS
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

    user_question = st.text_input("Ask a Question e.g Tell me about deepak")

    if user_question:
        # Load or create FAISS index
        index_path = "faiss_index"
        if not os.path.exists(index_path):
            path = os.path.join(os.path.dirname(__file__), "Deepak_Resume.pdf")
            raw_text = get_pdf_text(path)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)

        user_input(user_question)

if __name__ == "__main__":
    main()
st.title(" A passionate Data Analytics & Machine Learning Enthusiast from India")
st.write("MS.c Mathematics final year student, Data Analytics and Machine Learning Enthusiast.")
st.write("[Learn More >](https://github.com/dk1coding1zone)")

# what I do 
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What I do")
        st.write(
    '''
    My name is Deepak Singh, and here is a brief overview of my background and experience:

    1. **Education:** Pursuing an Integrated M.Sc. in Mathematics from SVNIT, Surat, with a current CGPA of 8.28.
    2. **Work Experience:** Completed a Machine Learning Internship at Uptricks Services Pvt. Ltd., focusing on data analysis and machine learning applications.
    3. **Projects:** Worked on Sales Data Forecasting, SQL Action: Real-World Case Studies, and developed an EDA Helper Function to automate exploratory data analysis tasks.
    4. **Technical Skills:** Proficient in Python and SQL, and experienced with tools like Power BI, Github, and Google Collab.
    5. **Certifications:** Earned certifications in Data Analysis with Python from Google-Coursera and Power BI from Microsoft-Press.
    '''
)

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
image_1=Image.open("images/image1.jpg")
image_2=Image.open("images/image2.jpg")
image_3=Image.open("images/image3.jpg")
with st.container():
    st.write("---")
    st.header("My Projects")
    st.write("##")
    image_column,text_column=st.columns((1,2))
    with image_column:
        st.image(image_1)
    with text_column:
        st.subheader("Sales Data Forecasting")
        st.write(
            '''
        – Forecasting: Conducted an in-depth sales analysis and forecasting for a business entity, applied time series analysis
         to generate accurate sales forecasts for the next 15 days.
        
        – Preprocessing: The project included stages such as importing and cleaning data, creating the dashboard, performing
        DAX queries and data analysis, and providing actionable insights and recommendations. (Python, Power BI, DAX)

            '''
        )
        st.markdown("[See More >](https://github.com/dk1coding1zone)")
    with st.container():
            image_column,text_column=st.columns((1,2))
    with image_column:
        st.image(image_2)
    with text_column:
        st.subheader("SQL Action: Real-World Case Studies")
        st.write(
            '''
        – SQL Analysis: Conducted in-depth SQL analysis on 5+ real-world datasets (Data Science Jobs, Google Play Store,
        Shark Tank India, Swiggy, Indian Tourism) to extract actionable insights. Gained hands-on experience in problem solving and data manipulation, enhancing SQL proficiency and analytical capabilities. (SQL, MySQL, Excel)
        
        – LLM Integration: Integrated some datasets with an LLM model to automate SQL queries, allowing users to get
        answers to their questions directly, saving 20+ hours a week by automating queries (Generative AI).
            '''
        )
        st.markdown("[See More >](https://github.com/dk1coding1zone)")
    with st.container():
            image_column,text_column=st.columns((1,2))
    with image_column:
        st.image(image_3)
    with text_column:
        st.subheader("EDA Helper Function")
        st.write(
            '''
        – EDA: Created an EDA helper function that saves 15+ hours a week by automating tasks like detecting missing
        values, generating plots, and performing hypothesis tests for both numeric and categorical data.
        
        – Time Saving: Basic and advanced exploratory data analysis (EDA) tasks can be quickly performed by importing
        the EDA helper function, which speeds up the analysis process (Python, NumPy, Pandas, Seaborn, Matplotlib).

            '''
        )
        st.markdown("[See More >](https://github.com/dk1coding1zone)")

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
