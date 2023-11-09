import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import time
# get a token: https://platform.openai.com/account/api-keys







 
load_dotenv()

 
def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
 
        # Initialize chats
        if "chats" not in st.session_state:
            st.session_state.chat_id = 0
            st.session_state.chats = [[]]
            st.session_state.messages = []


        with st.sidebar:
            #Create a button to increment and display the number
            if st.button('New chat+'):
                st.session_state.chats.append([])
                st.session_state.messages=st.session_state.chats[len(st.session_state.chats)-1]
                st.session_state.chat_id=len(st.session_state.chats)-1
            for chat in range(len(st.session_state.chats)):
                #button_key = f'button_{i}'
                if st.button(f'chat-{chat+1}'):
                    st.session_state.messages = st.session_state.chats[chat]
                    st.session_state.chat_id=chat
            add_vertical_space(5)
            st.write('Made with ❤️ by Ajeer')

        # st.title("Simple chat")
        # if st.button('New chat+'):
        #         st.session_state.chats.append([])
        # for chat in range(len(st.session_state.chats)):
        #     #button_key = f'button_{i}'
        #     if st.button(f'chat-{chat}'):
        #         st.write(chat)


        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.chats[st.session_state.chat_id]=st.session_state.messages
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                docs = VectorStore.similarity_search(query=prompt, k=3)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    assistant_response = chain.run(input_documents=docs, question=prompt)

                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
 
if __name__ == '__main__':
    main()