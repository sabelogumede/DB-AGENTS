import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API key from environment variables
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize the model with your chosen parameters
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022", 
    temperature=0,                      
    max_tokens=1000,
    api_key=anthropic_api_key  # Pass the API key here
)

# Create a conversation with defined roles
messages = [
    SystemMessage (
        content="You are a knowledgeable historian specializing in world history, including the formation of countries and their timelines. Your name is Belo, and you are here to provide accurate historical insights.",
    ),
    HumanMessage(content="Could you explain what the capital of South Africa is and provide some context about its significance?")
]

# Invoke the model with the messages and Print the content of the response
# response = model.invoke(messages)
# print("Response from model:", response.content)

def first_agent(messages):
    res = model.invoke(messages)
    return res

def run_agent():
    print("Simple AI Agent: Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print("AI Agent is thinking...")
        messages = [HumanMessage(content=user_input)]
        response = first_agent(messages)
        print("AI Agent: getting the response...")
        print(f"AI Agent: {response.content}")

if __name__ == "__main__":
    run_agent()