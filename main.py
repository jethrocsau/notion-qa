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
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(documents)

# load embeddings
embeddings = HuggingFaceEmbeddings()

# Vector Index creation
docsearch = FAISS.from_documents(documents,embeddings)

# Model selection
repo_id = "google/flan-t5-xxl"
model = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":1e-10})

#Initialize memory
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
chain_1 = LLMChain(
    llm=model,
    prompt=prompt,
    memory=ConversationBufferMemory(),
    verbose = True  
)


#### Retrieval chain type
#Defome prompt template & chain

prompt_template_2 = """Use the context below to answer the question asked. Anchor any statistics and facts only on the context provided.:
    CONTEXT: 

    {context}
    
    QUESTION: {question}
    
    ANSWER:"""

prompt_2 = PromptTemplate(
    template=prompt_template_2, input_variables=["context", "question"]
)

#RQA langchain init
chain_2 = LLMChain(
    llm=model,
    prompt=prompt_2,
    memory=ConversationBufferMemory(),
    verbose = True  
)

#Init retrieval QA chain
def generate_context_answer(question):
    docs = docsearch.similarity_search(question, k=4)
    context = ''
    for i,doc in enumerate(docs):
        context += "CONTEXT " + str(i) + ": " + doc.page_content + "\n\n"
    input = [{"context":context,"question":question}]
    print(chain_2.apply(input))

count =0
while True and count <5:
    query = input("Question: ")
    generate_context_answer(query)
    count+=1

#generate_context_answer("What has research articles said about CO2 levels and its impact on students?")

#r_qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=model, chain_type="map_reduce", retriever=docsearch.as_retriever())
#r_qa_chain({"question": query}, return_only_outputs=False)

# Conversational Promp
prompt_template_3 = """Below is a friendly conversation between a Human and an AI. The AI is talkative and explain its thought process out to the human.

    CONVERSATION:
    HUMAN: Hello!
    AI: Hello how are you how can I help you?
    HUMAN: Can you explain what is the meaning of life?
    AI: 
    """

#RQA langchain init
chain_2 = LLMChain(
    llm=model,
    prompt=prompt_2,
    memory=ConversationBufferMemory(),
    verbose = True  
)