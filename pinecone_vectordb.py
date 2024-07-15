# import Libraries

import os    
import configparser
#import openai
import langchain
#import pinecone 
from pinecone import Pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import Pinecone
# from langchain.llms import OpenAI
#from langchain.vectorstores import pinecone
from langchain_pinecone import PineconeVectorStore

from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA 
from langchain.chains import RetrievalQAWithSourcesChain 

global chain 
global index
#from langchain_pinecone import Pinecone

def get_api_key(source: str):
    config = configparser.ConfigParser()
    config.read('..\\config.ini')
    api_key = config.get('KEYS',source)
    return api_key


def get_open_ai_key():
    config = configparser.ConfigParser()
    config.read('..\\config.ini')
    api_key = config.get('KEYS','OPENAI_API_KEY')
    return api_key

def init_pinecone():
    ## Vector Search DB In Pinecone
    '''
    pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="aws"
    )
    '''
    PINECONE_API_KEY=get_api_key("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name="myvectorindex"
    #pc = Pinecone(api_key=PINECONE_API_KEY)
    #index = pc.Index("myvectorindex")
        
## Read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

## Divide the docs into chunks
### https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html#
def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

'''
## Create embedding and get embedding
def get_embedding():
    """
    :param text: The text to compute an embedding for
    :return: The embedding for the text
    """
    ## Embedding Technique Of OPENAI
    embeddings=OpenAIEmbeddings(api_key=get_open_ai_key())
    embeddings

    #vectors=embeddings.embed_query("How are you?")
    #len(vectors)
    #print(len(vectors))
   '''
 ## Cosine Similarity Retreive Results from VectorDB
def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results  

'''
## Search answers from VectorDB
def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response
'''

## Search answers from VectorDB
def retrieve_answers(query,index):
    #doc_search=retrieve_query(query)
    doc_search=index.similarity_search(query,k=2)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response

def main():
    print("starting the program")
    PINECONE_API_KEY=get_api_key("PINECONE_API_KEY")


    doc=read_doc('documents/')
    len(doc)
    documents=chunk_data(docs=doc)
    len(documents)
    #get_embedding()
    embeddings=OpenAIEmbeddings(api_key=get_open_ai_key())
    embeddings
    print(embeddings)
    #init_pinecone()

    ## Vector Search DB In Pinecone
    '''
    pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="aws"
    )
    '''
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    os.environ['OPENAI_API_KEY'] = get_open_ai_key()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name="myvectorindex"
    #pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("myvectorindex")

#    index_name="myvectorindex"
    #index=Pinecone.from_documents(doc,embeddings,index_name=index_name)

    vectorstore = PineconeVectorStore(  
    index, embeddings)  
    
    #create vector records in pinecone with metadata
    '''
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        doc,
        index_name=index_name,
        embedding=embeddings
    )
    '''

    #query = "Can you pull the travelling dates from the itineray"
    query = "Can you pull the travellers  from the itineray"  
    
    doc_search = vectorstore.similarity_search(query,k=3)

    #doc_search=index.similarity_search(our_query,k=2)
    print(doc_search)


    llm = ChatOpenAI(  
    openai_api_key=get_open_ai_key(),  
    model_name='gpt-3.5-turbo',  
    temperature=0.0)

    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=vectorstore.as_retriever()  
    )  
    response = qa_with_sources(query)


    #llm=OpenAI(model_name="gpt-3.5-turbo",temperature=0.5)
    #llm=OpenAI(model_name="text-davinci-003",temperature=0.5)
    #chain=load_qa_chain(llm,chain_type="stuff")
    
    #our_query = "Can you pull the travelling dates from the itineray"
    #answer = retrieve_answers(our_query,index)


    #doc_search = vectorstore.similarity_search(our_query,k=3)

    #doc_search=index.similarity_search(our_query,k=2)
    #print(doc_search)
    #response=llm. invoke(input_documents=doc_search,question=our_query)
    print(response)
    
    '''
    doc_search=retrieve_query(query,index)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    '''

    print("ending the program")


if __name__ == "__main__":
 main()