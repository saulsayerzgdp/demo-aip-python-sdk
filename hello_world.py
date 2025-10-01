"""Simple CV reader agent demo using glaip_sdk."""

from glaip_sdk import Client
from utils import create_agent

client = Client()

agent = create_agent(client)

query = "Read the CV from sample_cv.pdf and tell me the candidate's name."
print(f"Query: {query}")

response = agent.run(query, files=["sample_cv.pdf"])
print(f"Response: {response}")

agent.delete()
