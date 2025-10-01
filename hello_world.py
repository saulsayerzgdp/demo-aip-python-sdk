"""Simple CV reader agent demo using glaip_sdk."""

from glaip_sdk import Client
from utils import create_agent, read_file_as_binary

client = Client()

agent = create_agent(client)       

file_content, filename = read_file_as_binary("sample_cv.pdf")

query = "Read the CV from sample_cv.pdf and tell me the candidate's name."
print(f"Query: {query}")

response = agent.run(query, files={file_content})
print(f"Response: {response}")

agent.delete()
