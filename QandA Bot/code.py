from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def bot(topic):
    system_template = """you are a helpful assistant that exclusively answers questions about {topic}.
    if asked about anything else,respond politely that you only answers questionsabout {topic}.
    Keep your answers concise and factual."""

    prompt = ChatPromptTemplate.from_messages([
        ("system",system_template),
        ("human","{question}")
    ])

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    chain = prompt | model

    return chain

if __name__ == "__main__":
    topic = input("Enter a topic to ask questions about: ")

    qa_bot = bot(topic)

    while True:
        question = input("Your question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        response  = qa_bot.invoke({"topic": topic,"question": question})
        print(f"Question : {question}")
        print(f"Answer : {response.content}")
