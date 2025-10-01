"""End-to-end CV reader agent demo with evaluation using glaip_sdk.

This script:
1. Loads queries from a CSV file
2. Processes them using a CV reader agent
3. Saves the results back to CSV
4. Evaluates the results using GEval
"""

import asyncio

from glaip_sdk import Client
from utils import create_agent, evaluate_results, load_queries, process_queries, save_results

CSV_FILE = "cv_agent_results.csv"


async def main():
    """Main function to run the CV agent end-to-end pipeline."""
    
    # Initialize client
    client = Client()
    
    queries = load_queries(CSV_FILE)
    
    agent = create_agent(client)
    
    try:
        results = process_queries(agent, queries)
        save_results(results, CSV_FILE)
        await evaluate_results(CSV_FILE)
    finally:
        agent.delete()


if __name__ == "__main__":
    asyncio.run(main())
