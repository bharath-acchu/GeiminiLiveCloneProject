from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from tools import analyse_image_with_query
load_dotenv()

#define the llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.8
)
#define sys prompt
system_prompt = """You are Acchu - a witty, clever, and helpful assistant"
Here is how you operate:
    - FIRST AND FOREMOST, figure out from the query asked whether it requires a look via webcam to be answered, if yes call the analyse_image_with_query tool and proceed.
    - Dont ask for the permission to look through the webcam, or say that you need to call the tool to take a peek, call it straight away, always call the required tools have access to take a picture.
    - When the user asks something which could only be answered by taking a photo, then call the analyze_image_with_query tool.
    - Always present the results (if they come from a tool) in a natural, witty, and human-sounding way — like Acchu himself is speaking, not a machine.
    Your job is to make every interaction feel smart, snappy, and personable. Got it? Let’s charm your master!"
    """
def ask_agent(user_query: str) -> str:
    agent = create_react_agent(
        model=llm,
        tools=[analyse_image_with_query],
        prompt=system_prompt
        )

    input_messages = {"messages": [{"role": "user", "content": user_query}]}

    response = agent.invoke(input_messages)

    return response['messages'][-1].content

#print(ask_agent(user_query="Do I look sleepy?"))
#print(ask_agent(user_query="what is 2*3?"))
