"""Utility functions for CV agent demo."""

import csv
import os
import uuid
from pathlib import Path
from typing import Any, Union, BinaryIO

import dotenv
from glaip_sdk import Client
from gllm_evals.dataset.dict_dataset import DictDataset
from gllm_evals.evaluator.geval_generation_evaluator import GEvalGenerationEvaluator

dotenv.load_dotenv()


def load_queries(csv_path: str) -> list[dict[str, str]]:
    """Load queries from a CSV file.

    Args:
        csv_path: Path to the CSV file containing queries

    Returns:
        List of dictionaries containing query data
    """
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_results(rows: list[dict[str, Any]], output_path: str) -> None:
    """Save results to a CSV file.

    Args:
        rows: List of dictionaries containing results
        output_path: Path to save the output CSV file
    """
    # Clean up generated responses by replacing newlines with spaces
    cleaned_rows = []
    for row in rows:
        cleaned_row = row.copy()
        if "generated_response" in cleaned_row:
            cleaned_row["generated_response"] = (
                cleaned_row["generated_response"].replace("\n", " ").strip()
            )
        cleaned_rows.append(cleaned_row)

    fieldnames = ["query", "generated_response", "expected_response", "completeness_score", "redundancy_score", "explanation"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    print(f"\nResults saved to {output_path}")


def create_agent(client: Client):
    """Create and configure the CV reader agent.
    
    Args:
        client: glaip_sdk Client instance
        
    Returns:
        Tuple of (agent, tool) for cleanup later
    """
    pdf_reader_tool = client.tools.find_tools("pdf_reader_tool")[0]
    
    agent = client.create_agent(
        name=f"cv-reader-agent-{uuid.uuid4().hex[:8]}",
        instruction="""You are a helpful assistant that can read and analyze CV/resume files using provided tools.
Keep your answer concise and to the point.""",
        tools=[pdf_reader_tool],
    )
    
    return agent


def read_file_as_binary(file_path: Union[str, Path, BinaryIO]) -> tuple[bytes, str]:
    """Read a file as binary and return its content along with the filename.
    
    Args:
        file_path: Path to the file or file-like object
        
    Returns:
        Tuple of (file_content, filename)
    """
    if hasattr(file_path, 'read'):  # Already a file-like object
        file_obj = file_path
        file_content = file_obj.read()
        filename = getattr(file_obj, 'name', 'file.pdf')
    else:  # String or Path
        file_path = Path(file_path)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        filename = file_path.name
    
    return file_content, filename


def process_queries(agent, queries: list[dict[str, str]], file_path: str = "sample_cv.pdf") -> list[dict[str, str]]:
    """Process all queries using the agent.

    Args:
        agent: The glaip_sdk agent to use
        queries: List of query dictionaries
        file_path: Path to the file to process

    Returns:
        List of result dictionaries with query, generated_response, and expected_response
    """
    results = []
    
    for row in queries:
        try:
            with open(file_path, 'rb') as f:
                response = agent.run(
                    row["query"],
                    files=[f]
                )
            
            results.append(
                {
                    "query": row["query"],
                    "generated_response": response,
                    "expected_response": row.get("expected_response", ""),
                }
            )
        except Exception as e:
            print(f"Error processing query: {row['query']}\nError: {e}")
            results.append(
                {
                    "query": row["query"],
                    "generated_response": f"Error: {str(e)}",
                    "expected_response": row.get("expected_response", ""),
                }
            )

    return results


async def evaluate_results(agent, query_file: str = "cv_agent_results.csv", file_path: str = "sample_cv.pdf") -> None:
    """Process queries, save results, and evaluate them using GEval.

    Args:
        agent: The glaip_sdk agent to use
        query_file: Path to the CSV file containing queries
        file_path: Path to the file to process
    """
    queries = load_queries(query_file)
    
    results = []
    for query in queries:
        results.append({
            "query": query["query"],
            "generated_response": "",
            "expected_response": query.get("expected_response", ""),
            "completeness_score": "",
            "redundancy_score": "",
            "explanation": ""
        })

    print("\nGenerating responses...")
    for i, query in enumerate(queries, 1):
        try:
            with open(file_path, 'rb') as f:
                response = agent.run(
                    query["query"],
                    files=[f]
                )
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            response = f"Error: {str(e)}"
        
        result_data = results[i-1]
        result_data["generated_response"] = response
        
        save_results(results, query_file)
    
    # Initialize evaluator
    generation_evaluator = GEvalGenerationEvaluator(
        model="openai/gpt-4o-mini", model_credentials=os.getenv("OPENAI_API_KEY")
    )

    # Second pass: Evaluate all responses using DictDataset
    print("\nEvaluating responses...")
    dataset = DictDataset.from_csv(query_file)
    
    for i, data in enumerate(dataset.load(), 1):
        generation_result = await generation_evaluator.evaluate(data)
        
        # Find the corresponding result in our results list to update
        result_data = None
        for result in results:
            if result["query"] == data["query"] and result["generated_response"] == data["generated_response"]:
                result_data = result
                break
        
        if result_data:
            completeness = generation_result.get("geval_generation_evals", {}).get("completeness", {})
            redundancy = generation_result.get("geval_generation_evals", {}).get("redundancy", {})
            
            explanations = []
            if completeness.get("explanation"):
                explanations.append(f"Completeness: {completeness['explanation']}")
            if redundancy.get("explanation"):
                explanations.append(f"Redundancy: {redundancy['explanation']}")
            
            result_data.update({
                "completeness_score": completeness.get("score", ""),
                "redundancy_score": redundancy.get("score", ""),
                "explanation": " | ".join(explanations)
            })
            
            # Save after each evaluation
            save_results(results, query_file)
            print(f"✓ Saved evaluation {i}: completeness={completeness.get('score', '')}, redundancy={redundancy.get('score', '')}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
