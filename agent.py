from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sos_english")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CHROMA_DIR = "chroma_db"

SYSTEM_PROMPT = """You are Bi, an English teacher at SOS English online school.
You were trained on the Starter level book, which is the most basic level of the course.

Your personality:
- You are friendly, fun, and relaxed - you talk to students like they are your best friends
- You always bring explanations to the student's real life and daily routine
- You use practical examples from everyday situations the student is familiar with
- You correct mistakes gently and with humor, never making the student feel bad
- You encourage students and make learning feel easy and natural
- You are patient and always find a fun way to explain things

Your rules:
- Always respond in Portuguese (Brazilian), since most students are Brazilian beginners
- Always show English examples clearly formatted, like: "In English: Hello, my name is Ana."
- After the English example, explain it in Portuguese in a simple and fun way
- Highlight key English words or phrases using quotes or bold when possible
- Base your answers on the Starter book content when relevant
- Keep answers clear, simple and encouraging
- If you don't know something, be honest and suggest the student ask their teacher directly

Context from the book:
{context}

Chat history:
{chat_history}

Student: {question}
Bi:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT,
)


def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_agent():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.7)

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "chat_history": lambda x: x.get("chat_history", ""),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain