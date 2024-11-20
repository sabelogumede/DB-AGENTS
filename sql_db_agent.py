import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import pandas as pd
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        max_tokens=1000,
        api_key=anthropic_api_key,
)

# Langchain imports
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# Path to SQLite database file - data extracted from our csv file
database_file_path = "./db/salary.db"

# Created an engine to connect to the SQLite database
engine = create_engine(f"sqlite:///{database_file_path}")
file_url = "./data/salaries_2023.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)

# Read data from CSV and fill missing values with 0
df = pd.read_csv(file_url).fillna(value=0)

# Write the DataFrame to the SQLite database
df.to_sql("salaries_2023", con=engine, if_exists="replace", index=False)

# print(f"Database created successfully! {df}")

# Part 2: Prepare the SQL prompt
MSSQL_AGENT_PREFIX = """
You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most interesting examples in the database.
- Never query for all the columns from a specific table; only ask for the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE; ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running a SQL Query in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer in a section that starts with: "Explanation:". Include the SQL query as part of this explanation section.
- If the question does not seem related to the database, just return "I don’t know" as the answer.
- Only use the below tools. Only use the information returned by these tools to construct your query and final answer.
- Do not make up table names; only use tables returned by any of the tools below.
- As part of your final answer, please include the SQL query you used in JSON format or code format.

## Tools:
"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """
## Use the following format:

Question: The input question you must answer.
Thought: You should always think about what to do.
Action: The action to take; should be one of [{tool_names}].
Action Input: The input to the action.
Observation: The result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: The final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT TOP (10) [base_salary], [grade] 
FROM salaries_2023 WHERE state = 'Division'

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought: I now know the final answer.
Final Answer: There were 27437 workers making 100,000.

Explanation:
I queried the `xyz` table for the `salary` column where the department is 'IGM' and the date starts with '2020'. The query returned a list of tuples with base salaries for each day in 2020. To answer the question,
I took the sum of all salaries in that list, which is 27437.
I used the following query:

```sql
SELECT [salary] FROM xyztable WHERE department = 'IGM' AND date LIKE '2020%'

===> End of Example
"""

# Create Agent
db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

# QUESTION = """How many employees are in the ABS 85 Administrative and their avg salaries, and also How many of them are female? """
QUESTION = """What is the highest avarage salary by department, and give me the number?"""

# passing multiple parameters
sql_agent = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=model,
    toolkit=toolkit,
    top_k=30,
    verbose=True,
)

# res = sql_agent.invoke(QUESTION)

# print(res)

# ---------- import streamlit for UI ---------------- 
import streamlit as st

st.title("ZuluAI")
st.header("SQL Query AI Agent")

question = st.text_input("Enter your query:")

if st.button("Run Query"):
    if question:
        res = sql_agent.invoke(question)
        # st.markdown("# Intermediate Steps")

        st.markdown(res["output"])
else:
    st.error("Please enter a query.")