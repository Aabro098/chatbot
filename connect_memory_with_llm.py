import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM - Using a supported model
HF_TOKEN = os.environ.get("HF_TOKEN")

# Option 1: Use Mistral v0.2 (smaller, faster)
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Option 2: Or use Mixtral (larger, better quality)
# HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Option 3: Or use Meta Llama
# HUGGINGFACE_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Option 4: Or use Zephyr (good open source alternative)
# HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

def load_llm(huggingface_repo_id):
    # Create base endpoint
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=1024
    )
    # Wrap it in ChatHuggingFace for conversational format
    chat_llm = ChatHuggingFace(llm=llm)
    return chat_llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain using LCEL (LangChain Expression Language)
retriever = db.as_retriever(search_kwargs={'k': 3})

# Format retrieved documents to string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    | load_llm(HUGGINGFACE_REPO_ID)
    | StrOutputParser()
)

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke(user_query)
print("RESULT: ", response)