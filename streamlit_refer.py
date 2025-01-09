import streamlit as st
import tiktoken
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(page_title="LLMChat", page_icon=":random:")
    st.title("RAG 기반 LangChain ChatBot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
        
    vectorestore = None
    results_with_scores = []
    
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorestore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorestore, openai_api_key) 
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

   

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents'] 

                if vectorestore is not None:
                        # FAISS 유사도 검색
                    results_with_scores = similarity_search_with_relevance_scores(vectorestore, query, k=3)
                        

                st.markdown(response)
                
                with st.expander("참고 문서 확인"):
                    for i, result in enumerate(results_with_scores):
                        st.markdown(f"**Document {i+1}:** {result['document'].metadata['source']}")
                        st.markdown(f"**Similarity Score:** {result['score']:.4f}")
                        st.markdown(result['document'].page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

# FAISS 유사도 점수 계산
def similarity_search_with_relevance_scores(vectorestore, query, k=3):
    # 쿼리 벡터 추출
    query_embedding = vectorestore.embedding_function(query)

    # FAISS에서 가장 가까운 벡터 검색
    distances, indices = vectorestore.index.search(query_embedding.reshape(1, -1), k)
    
    # 코사인 유사도 계산 (거리 -> 유사도로 변환)
    scores = 1 / (1 + distances)  # 거리가 작을수록 유사도가 높음
    
    # 문서와 점수 묶음 반환
    results_with_scores = []
    for i in range(k):
        doc = vectorestore.get_documents()[indices[0][i]]  # 검색된 문서 추출
        results_with_scores.append({"document": doc, "score": scores[0][i]})
    
    return results_with_scores

def get_conversation_chain(vectorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0) 
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorestore.as_retriever(search_type='mmr', verbose=True), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

if __name__ == '__main__':
    main()
