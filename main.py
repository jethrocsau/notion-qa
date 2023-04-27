from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationChain, RetrievalQAWithSourcesChain
from langchain import HuggingFaceHub,PromptTemplate, LLMChain
from transformers import AutoTokenizer
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup

#Load config
load_dotenv()

#Load pdfs
pdf_dir =  f"./PDF_DB/"
loaders = [UnstructuredPDFLoader(os.path.join(pdf_dir,fn)) for fn in os.listdir(pdf_dir)]
documents = []
for loader in loaders:
    documents.extend(loader.load())

#split text
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
documents = text_splitter.split_documents(documents)

# load embeddings
embeddings = HuggingFaceEmbeddings()

# Vector Index creation
docsearch = FAISS.from_documents(documents,embeddings)

# Model selection
repo_id = "google/flan-t5-base"
model = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":1e-10})

#Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

### Simple LLM Chain
#Defome prompt template & chain
prompt_template = """Read the documents provided and answer the question below based on the information presented only:

Talk me through your answer.

Question: {question}

Answer:"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["question"]
)


#Simple langchain init
llm_chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose = True  
)


#### Retrieval chain type
#Defome prompt template & chain

prompt_template = """Use the context below to write a 300 word reponse about the question asked below:
    Context: {context}
    Topic: {topic}
    Blog post:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "topic"]
)

#Init retrieval QA chain
r_qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=model, chain_type="map_reduce", retriever=docsearch.as_retriever())

#run query
query = "Where are you created?"
r_qa_chain({"question": query}, return_only_outputs=False)