import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

class DocumentQASystem:
    def __init__(self,pdf_path):
        load_dotenv()

        self.text = self._extract_text_from_pdf(pdf_path)
        self.text_chunks = self._split_text()
        self.vector_store = self._create_vector_store()
        self.qa_chain = self._create_retrieval_chain()
    
    def _extract_text_from_pdf(self, pdf_path):
        with open(pdf_path,'rb') as file:
            pdf_reader  = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    
    def _split_text(self,chunk_size = 1000,chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(self.text)
    
    def _create_vector_store(self):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        return Chroma.from_texts(
            texts = self.text_chunks,
            embedding=embeddings
        )
    
    def _create_retrieval_chain(self):
        llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-pro",
            temperature = 0.3
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = "stuff",
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k":3}
            )
        )
    
    def ask_question(self,question):
        response =self.qa_chain.invoke({'query':question})
        return response['result']
    
def main():
    pdf_path = "D:\Projects\LangChain\PDFS\Forbidden-Prophecies-Abu-Zakariyah.pdf"
    qa_system = DocumentQASystem(pdf_path)

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        try:
            answer = qa_system.ask_question(question)
            print("\nAnswer:", answer)
            print("\n" + "="*50 + "\n")           
        except Exception as e:
            print(f"An error occurred: {e}")
if __name__ == "__main__":
    main()