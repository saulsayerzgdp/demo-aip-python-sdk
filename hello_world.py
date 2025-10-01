"""Simple CV reader agent demo using glaip_sdk."""

from glaip_sdk import Client
from utils import create_agent, read_file_as_binary

client = Client()

agent = create_agent(client)       

query = "Read the CV from sample_cv.pdf and tell me the candidate's name."
print(f"Query: {query}")

with open("sample_cv.pdf", "rb") as f:
    response = agent.run(query, files=[f])
print(f"Response: {response}")

agent.delete()
