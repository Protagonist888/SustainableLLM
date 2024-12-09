from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

chat_llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-instruct",
    temperature=.9999
    )


# Below code is for a grounded solution
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the Hawaiin north shore surf conditions on the beach. Respond using surfer slang.",
        ),
        ( "system", "{context}" ),
        ( "human", "{question}" ),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

current_weather = """
    {
        "surf": [
            {"beach": "North Shore", "conditions": "100ft monster waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds with sand break"}
            {"beach": "Mavericks", "conditions": "5 ft rolling waves with coral break"}
        ]
    }"""

response = chat_chain.invoke(
    {
        "context": current_weather,
        "question": "What is the weather like on North Shore? Where do you recommend surfing?",
    }
)

print(response)



''' Below is V1
instructions = SystemMessage(content="""
    You are a Hawaiin surfer dude, having a conversation about the surf conditions
    on the Hawaiin North Shore during a 100 year storm. Respond using surfer slang.
    """ )

question = HumanMessage(content="What are the wave conditions like?")

response = chat_llm.invoke([
    instructions,
    question
])

# For some reason print(response.content) does not work
print(response)
'''