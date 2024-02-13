import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Replace book.pdf with any pdf of your choice

# loader = UnstructuredPDFLoader("docs/Matthew.pdf")
# pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()

#docsearch = Chroma.from_documents(pages, embeddings, persist_directory="./chroma_Matthew").as_retriever()

#saveEmbeddings = createChromafromEmbeddings("docs/Matthew.pdf",embeddings)

# load from disk
#docsearch = Chroma(persist_directory="./chroma_Matthew", embedding_function=embeddings).as_retriever()
# docs = db3.similarity_search(query)
# print(docs[0].page_content)

def createChromafromEmbeddings(filePath, embedding):
    loader = UnstructuredPDFLoader(filePath)
    pages = loader.load_and_split()
    fileName = get_filename_without_extension(filePath)
    chromaPath = "dbs/chroma_"+fileName
    docsearch = Chroma.from_documents(pages, embedding, persist_directory=chromaPath).as_retriever()
    
    return docsearch

def get_filename_without_extension(path):
    filename_with_extension = os.path.basename(path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

def returnChromafromEmbeddings(chromaPath, embedding):
    docsearch = Chroma(persist_directory=chromaPath, embedding_function=embedding).as_retriever()
    
    return docsearch

print("\n\n")

#chromapath = createChromafromEmbeddings("docs/Genesis.pdf", embeddings)

#chromapath = returnChromafromEmbeddings("./chroma_Genesis", embeddings)

fileName = "Ruth"

if os.path.exists("dbs/chroma_"+fileName):
    print("Chroma exists so returning from Embeddings")
    chromapath = returnChromafromEmbeddings("dbs/chroma_"+fileName, embeddings)
else:
    print("Chroma does not exist so creating from Embeddings")
    chromapath = createChromafromEmbeddings("docs/"+ fileName +".pdf", embeddings)
      

print(chromapath)

print("\n\n")

while True: 
# Choose any query of your choice
    print("\n\n")
    query = input("Ask a question: ")
    
    if query.lower() == 'quit':
        print("Thank you!")
        break    
    query += "Please respond only from the current context and do not mention current context in your response"
    docs = chromapath.get_relevant_documents(query)
    # chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    output = chain.run(input_documents=docs, question=query)
    print(output)

#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
#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