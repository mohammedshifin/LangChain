from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import chainlit as cl
import PyPDF2
from chromadb.config import Settings

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Define the system prompt template
system_template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Context: {context}

Question: {question}

Answer the question based on the context provided."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
])

@cl.on_chat_start
async def on_chat_start():
    # Display welcome message
    elements = [
        cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    await cl.Message(
        content="Hello there, Welcome to AskAnyQuery related to Data!", 
        elements=elements
    ).send()

    # Request PDF file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing '{file.name}'...")
    await msg.send()

    try:
        # Process PDF file
        with open(file.path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()

        # Split text into chunks
        texts = text_splitter.split_text(pdf_text)
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        # Initialize embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vectorstore = Chroma.from_texts(
            texts,
            embeddings,
            metadatas=metadatas,
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

        # Create LCEL chain
        chain = (
            RunnableParallel(
                {"context": retriever, "question": RunnablePassthrough()}
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # Store session data
        cl.user_session.set("chain", chain)
        cl.user_session.set("retriever", retriever)

        msg.content = f"Processing '{file.name}' done. You can now ask questions!"
        await msg.update()

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        await cl.Message(content=error_msg).send()
        raise

@cl.on_message
async def main(message: str):
    try:
        chain = cl.user_session.get("chain")
        if not chain:
            await cl.Message(content="Please upload a PDF file first.").send()
            return

        # Initialize callback handler for streaming
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        # Stream the response
        async for chunk in chain.astream(
            message,
            config={"callbacks": [cb]}
        ):
            await cl.Message(content=chunk).send()

        # Get sources from retriever
        retriever = cl.user_session.get("retriever")
        if retriever:
            docs = await retriever.ainvoke(message)
            source_elements = []
            
            for doc in docs:
                source_elements.append(
                    cl.Text(content=doc.page_content, name=doc.metadata["source"])
                )
            
            if source_elements:
                await cl.Message(
                    content="Sources referenced:", 
                    elements=source_elements
                ).send()

    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        await cl.Message(content=error_msg).send()