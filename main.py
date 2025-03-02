import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from prompt_template import PROMPT

app = FastAPI()

BIGGER_DEEPSEEK = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M"  # catches irrelevant information, omits key information
SMALLER_DEEPSEEK = "DeepSeek-R1-Distill-Llama-8B-Q2_K"  # even incorrect sentences in polish
MISTRAL = "mistral-7b.Q3_K_S"  # The best model on 4GB GPU, often accurate and short answer!
SMALL_PHI4 = "phi-4-mini-q6_k_m"  # really fast, accurate answers
LLAMA3 = "Meta-Llama-3-8B-Instruct-Q4_K_S"  # doesn't want to answer in polish, bad answer, slow


# load model in LlamaCpp
llm = LlamaCpp(
    model_path=f"models/{SMALL_PHI4}.gguf",  # path to a model in gguf format
    # chat_format="phi",  # required for SMALLER_PHI4
    temperature=0.7,
    max_tokens=256,
    n_ctx=8192,  # size of context
    verbose=True,
    n_gpu_layers=-1,
    n_threads=8,
    n_batch=1024,
)


def load_documents(folder_path):
    documents_ = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        else:
            continue  # TODO: implement also support for videos
        documents_.extend(loader.load())
    return documents_


documents = load_documents("data/")

# chunking + overlapping
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# convert to vectors
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # may be loaded from file,
# to reduce problems with proxy, firewalls etc

# any vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# Connect LLM with RAG
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever(),
                                       return_source_documents=True, chain_type_kwargs={
        "prompt": PROMPT
    })


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_pdf(request: QueryRequest):
    result = qa_chain({"query": request.question})
    answer = result['result']

    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            sources.append({
                "txt_part": doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            })

    return {"answer": answer, "sources": sources}
