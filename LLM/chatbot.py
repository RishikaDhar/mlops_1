from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# 1) Load docs

df = pd.read_csv("data/Student_Performance.csv")

# Convert each row into a Document, one student record is considered as one document, splitting rows can break meaning
#langchain.textsplitter not suited
docs = []
for i, row in df.iterrows():              #iterate through rows
    text = " | ".join([f"{col}: {row[col]}" for col in df.columns])             #join name : value | age : value
    docs.append(
        Document(
            page_content=text,
            metadata={"row_id": i}
        )
    )



#Create Prompt 
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an academic performance assistant.
Use ONLY the provided student data to answer.

If the question asks for insights:
- summarize performance clearly
- highlight strengths/weaknesses
- suggest actionable improvements

Context:
{context}

Question:
{question}

Answer:
"""
)



#Embeddings + Vector DB
db = FAISS.from_documents(docs, OpenAIEmbeddings())


# Build RetrievalQA Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={'k':5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)


#Ask questions
result = qa_chain.invoke({"query": "Give insights for student_id 12"})
print(result["result"])

# show metadata
for doc in result["source_documents"]:
    print(doc.page_content)







