import os

import mlflow
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_crusoe import ChatCrusoe

load_dotenv(override=True)
mlflow.langchain.autolog()

llm = ChatCrusoe(model=os.environ["CRUSOE_MODEL"], temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a support assistant for Crusoe Cloud. Answer in {sentences} sentences."),
        ("human", "{question}"),
    ]
)
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"question": "How does Serverless Fine-Tuning work?", "sentences": 3}))
