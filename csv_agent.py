import os
import traceback
from dotenv import load_dotenv
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# add pre and sufix prompt
CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""

QUERY = "Which grade has the highest average base salary, and compare the average female pay vs male pay?"

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
        # For production, instead of using "allow_dangerous_code=True" above, on production you should:
        # - Limit the agent's capabilities to specific, safe operations
        # - Implement input validation
        # - Use predefined functions or a restricted set of allowed operations
    )
except anthropic.APIStatusError as e:
    print("Error creating pandas dataframe agent:")
    traceback.print_exc()
    raise

# User query/question
# query = "What is the average salary?"
try:
    # Use invoke() instead of run()
    res = agent.invoke({"input": CSV_PROMPT_PREFIX + QUERY + CSV_PROMPT_SUFFIX})
    # print(f"Agent response:, {res["output"]}")
except anthropic.APIStatusError as e:
    print("Error running the agent:")
    traceback.print_exc()
    raise

# UI for better data visualisation ---------------------------------------------------------------

import streamlit as st

# create a title
st.title("Database AI Agent with LangChain")

st.write("### Dataset Preview")
st.write(df.head())

# User input for Question
st.write("## Ask a Question")
question = st.text_input(
    "Enter your question about the dataset:",
    "Which grade has the highest avarage base salary, and compare the average female pay vs male pay?"
)

# Run the agent and display the result
if st.button("Run Query"):
    QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
    res = agent.invoke(QUERY)
    st.write("### Final Answer")
    st.markdown(res["output"])
    