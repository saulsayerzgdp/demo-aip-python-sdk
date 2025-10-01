"""End-to-end CV reader agent demo with evaluation using glaip_sdk.

This script:
1. Loads queries from a CSV file
2. Processes them using a CV reader agent
3. Saves the results back to CSV
4. Evaluates the results using GEval
"""

import asyncio
import uuid

from glaip_sdk import Client
from utils import evaluate_results

CSV_FILE = "cv_agent_results.csv"


async def main():
    """Main function to run the CV agent end-to-end pipeline."""
    
    # Need env variables AIP_API_KEY, AIP_API_URL
    client = Client()
    pdf_reader_tool = client.tools.find_tools("pdf_reader_tool")[0]
    agent = client.create_agent(
        name=f"cv-reader-agent-{uuid.uuid4().hex[:8]}",
        instruction="""You are a helpful assistant that can read and analyze CV/resume files using provided tools.
Keep your answer concise and to the point, only output the direct answer without any additional explanation.""",
        tools=[pdf_reader_tool],
    )

    # Need env variable OPENAI_API_KEY
    await evaluate_results(agent, CSV_FILE)
    
    agent.delete()


if __name__ == "__main__":
    asyncio.run(main())
