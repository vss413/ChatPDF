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

# load from disk
docsearch = Chroma(persist_directory="./chroma_Matthew", embedding_function=embeddings).as_retriever()
# docs = db3.similarity_search(query)
# print(docs[0].page_content)


print("\n\n")

print(docsearch)

print("\n\n")

while True: 
# Choose any query of your choice
    print("\n\n")
    query = input("Ask a question: ")
    
    if query.lower() == 'quit':
        print("Thank you!")
        break    
    query += "Please respond only from the current context"
    docs = docsearch.get_relevant_documents(query)
    # chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    output = chain.run(input_documents=docs, question=query)
    print(output)

#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