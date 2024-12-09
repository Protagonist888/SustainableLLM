# Examples from Neo4J LLM and KG fundamentals courses
# initializing an LLM using Langchain

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-instruct",
    temperature=.14556
    )


template = PromptTemplate(template="""
          You are a cockney fruit and vegetable seller.
            Your role is to assist your customer with their fruit and vegetable needs.
            Respond using cockney rhyming slang.
                          
            Output JSON as {{"description": "your response here"}}

            Tell me about the following fruit: {fruit}                
""", input_variables=["fruit"])

llm_chain = template | llm | SimpleJsonOutputParser()

# response = llm.invoke("What is Neo4j?")
# response = llm.invoke(template.format(fruit="apple"))
response = llm_chain.invoke({"fruit": "apple"})

print(response)