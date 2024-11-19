import os
import traceback
from dotenv import load_dotenv
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# Retrieve the API key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")

# Initialize the ChatAnthropic model
try:
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        max_tokens=1000,
        api_key=anthropic_api_key,
    )
except Exception as e:
    print("Error initializing ChatAnthropic model:")
    traceback.print_exc()
    raise

# Load the CSV
try:
    df = pd.read_csv('./data/salaries_2023.csv').fillna(value=0)
    # print("CSV loaded successfully.")
    # print(df.head())
except FileNotFoundError:
    print("Error: The file './data/salaries_2023.csv' was not found.")
    print("Current working directory:", os.getcwd())
    raise
except pd.errors.EmptyDataError:
    raise ValueError("The file './data/salaries_2023.csv' is empty.")
except pd.errors.ParserError:
    raise ValueError("The file './data/salaries_2023.csv' contains invalid data.")

# Create the pandas dataframe agent
try:
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        verbose=True,
        allow_dangerous_code=True
        # For production, instead of using "allow_dangerous_code=True" above, you should:
        # - Limit the agent's capabilities to specific, safe operations
        # - Implement input validation
        # - Use predefined functions or a restricted set of allowed operations
    )
except Exception as e:
    print("Error creating pandas dataframe agent:")
    traceback.print_exc()
    raise

# User query
query = "What is the average salary?"
try:
    # Use invoke() instead of run()
    res = agent.invoke({"input": query})
    print("Agent response:", res['output'])
except Exception as e:
    print("Error running the agent:")
    traceback.print_exc()
    raise