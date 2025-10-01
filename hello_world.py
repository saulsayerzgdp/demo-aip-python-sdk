"""Simple CV reader agent demo using glaip_sdk."""
import uuid

from glaip_sdk import Client
from utils import create_agent, read_file_as_binary

# Need env variables AIP_API_KEY, AIP_API_URL
client = Client()

pdf_reader_tool = client.tools.find_tools("pdf_reader_tool")[0]
agent = client.create_agent(
    name=f"cv-reader-agent-{uuid.uuid4().hex[:8]}",
    instruction="""You are a helpful assistant that can read and analyze CV/resume files using provided tools.
Keep your answer concise and to the point, only output the direct answer without any additional explanation.""",
    tools=[pdf_reader_tool],
)      

query = "Read the CV from sample_cv.pdf and tell me the candidate's name."
print(f"Query: {query}")

with open("sample_cv.pdf", "rb") as f:
    response = agent.run(query, files=[f])
print(f"Response: {response}")

agent.delete()
