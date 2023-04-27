from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub,PromptTemplate, LLMChain
from transformers import AutoTokenizer
from langchain.memory import ConversationBufferMemory

#Load config
load_dotenv()

#Load pdfs
pdf_dir =  f"./PDF_DB/"
loaders = [UnstructuredPDFLoader(os.path.join(pdf_dir,fn)) for fn in os.listdir(pdf_dir)]
documents = []
for loader in loaders:
    documents.extend(loader.load())

#split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=250)
documents = text_splitter.split_documents(documents)

# load embeddings
embeddings = HuggingFaceEmbeddings()

# Vector Index creation
docsearch = FAISS.from_documents(documents,embeddings)

# Model selection
repo_id = "waifu-workshop/pygmalion-6b"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"max_length":500})

#Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#Initialize chain
qa = ConversationalRetrievalChain.from_llm(llm,docsearch.as_retriever(), memory=memory)

# Start conversation
count = 0
query = "What are the article names of these documents provided?"
result = qa({"question": query})
print(result["answer"])
count+=1


#output = chain.run(input_documents=docs, question = query)
#Test 
#template = """: {question}
#
#Answer: Lets summarize only based on the inputs above."""

#prompt = PromptTemplate(template=template, input_variables=["question"])
#llm_chain = LLMChain(prompt=prompt, llm=llm)

#rint(llm_chain.run(question))