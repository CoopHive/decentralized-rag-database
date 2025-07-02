"""
Evaluation agent module.

This module provides an EvaluationAgent class for assessing the quality
of document processing and query results.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from src.query.query_db import query_collection, discover_user_collections
from src.utils.logging_utils import get_logger, get_user_logger

# Get module logger
logger = get_logger(__name__)

load_dotenv()


class EvaluationAgent:
    """
    Agent for evaluating and ranking query results from different collections.

    This class provides functionality to:
    1. Query multiple collections with the same query
    2. Store results in a temporary JSON file
    3. Evaluate and rank results using LLM agents via OpenRouter
    """

    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        """
        Initialize the evaluation agent.

        Args:
            model_name: Full model name in the format "provider/model"
                       (e.g., "openai/gpt-3.5-turbo", "anthropic/claude-3-opus-20240229")
        """
        self.model_name = model_name
        self.temp_dir = Path(__file__).parents[1].parent / "temp"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Get OpenRouter API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.error("OpenRouter API key is not set in environment variables")
            raise ValueError("OpenRouter API key not configured")

    def query_collections(
        self, query: str, db_path: Optional[str] = None, user_email: Optional[str] = None
    ) -> str:
        """
        Query multiple collections with the same query and store results.

        Args:
            query: Natural language query string
            db_path: Optional path to ChromaDB directory
            user_email: Optional user email for user-specific temp directory and auto-discovery

        Returns:
            Path to the temporary JSON file containing all results
        """
        # Use user-specific logger if user_email is available
        user_logger = get_user_logger(user_email, "evaluation_agent") if user_email else logger
        
        timestamp = int(time.time())
        
        # Use user-specific temp directory if user_email is provided
        if user_email:
            user_temp_dir = self.temp_dir / user_email / "evaluation"
            os.makedirs(user_temp_dir, exist_ok=True)
            results_file = user_temp_dir / f"query_results_{timestamp}.json"
        else:
            results_file = self.temp_dir / f"query_results_{timestamp}.json"

        # Auto-discover collections if not provided and user_email is available
        user_logger.info(f"Auto-discovering collections for user: {user_email}")
        collection_names = discover_user_collections(user_email, db_path)
        if not collection_names:
            user_logger.warning(f"No collections found for user: {user_email}")
            collection_names = []

        collection_results: Dict[str, Any] = {}
        all_results: Dict[str, Any] = {
            "query": query,
            "user_email": user_email,
            "total_collections": len(collection_names),
            "collection_names": collection_names,
            "collection_results": collection_results,
        }

        for collection_name in collection_names:
            user_logger.info(f"Querying collection: {collection_name}")
            try:
                result_json = query_collection(collection_name, query, db_path, user_email)
                result_data = json.loads(result_json)
                collection_results[collection_name] = result_data
            except Exception as e:
                user_logger.error(f"Error querying collection {collection_name}: {e}")
                collection_results[collection_name] = {"error": str(e)}

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        user_logger.info(f"Saved query results to {results_file}")
        return str(results_file)

    def evaluate_results(self, results_file: str) -> Dict[str, Any]:
        """
        Evaluate and rank results from different collections.

        Args:
            results_file: Path to JSON file containing query results

        Returns:
            Dictionary with evaluation results and rankings
        """

        with open(results_file, "r") as f:
            all_results = json.load(f)

        original_query = all_results["query"]
        user_email = all_results.get("user_email")
        collections = all_results["collection_results"]

        # Use user-specific logger if user_email is available
        user_logger = get_user_logger(user_email, "evaluation_agent") if user_email else logger

        evaluation = {
            "query": original_query,
            "rankings": {},
            "overall_best_collection": "",
            "reasoning": "",
        }

        prompt = self._generate_evaluation_prompt(original_query, collections)

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://coophive.com",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that evaluates search results.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                },
            )

            response.raise_for_status()
            response_data = response.json()
            evaluation_text = response_data["choices"][0]["message"]["content"]

            try:
                eval_data = json.loads(evaluation_text)
                evaluation.update(eval_data)
            except json.JSONDecodeError:
                user_logger.error(f"Failed to parse evaluation results: {evaluation_text}")
                evaluation["error"] = "Failed to parse evaluation results"
                evaluation["raw_response"] = evaluation_text
        except Exception as e:
            user_logger.error(f"Error evaluating results with model {self.model_name}: {e}")
            evaluation["error"] = str(e)

        eval_file = Path(results_file).with_name(
            f"{Path(results_file).stem}_evaluation.json"
        )
        with open(eval_file, "w") as f:
            json.dump(evaluation, f, indent=2)

        user_logger.info(f"Saved evaluation to {eval_file}")
        return evaluation

    def _generate_evaluation_prompt(
        self, query: str, collections: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for LLM to evaluate results.

        Args:
            query: Original query string
            collections: Dictionary of collection results

        Returns:
            Prompt string for LLM
        """
        prompt = f"""
    I need you to evaluate search results from different collections for the query: "{query}"

    Below are the results from {len(collections)} different collections:

    """

        for collection_name, results in collections.items():
            prompt += f"\n# Collection: {collection_name}\n"

            if "error" in results:
                prompt += f"Error: {results['error']}\n"
                continue

            if "results" in results and results["results"]:
                prompt += f"Found {len(results['results'])} results:\n"

                for i, result in enumerate(results["results"]):
                    prompt += f"\nResult {i+1}:\n"
                    prompt += f"Content: {result.get('document', 'N/A')[:300]}...\n"

                    if result.get("metadata"):
                        prompt += f"Metadata: {json.dumps(result['metadata'])}\n"

                    if "distance" in result:
                        prompt += f"Distance: {result['distance']}\n"
            else:
                prompt += "No results found in this collection.\n"

        prompt += """\n
        Please evaluate these search results and respond with a JSON object that includes:

        1. "rankings": A dictionary where keys are collection names and values are objects with:
        - "score": A score from 1-10 measuring relevance to the query
        - "reasoning": Brief explanation for the score

        2. "overall_best_collection": The name of the collection with the best results

        3. "reasoning": Explanation for why the best collection was chosen

        The response should be valid JSON like this:
        {
        "rankings": {
            "collection1": {"score": 8, "reasoning": "Contains comprehensive and relevant information..."},
            "collection2": {"score": 6, "reasoning": "Has some relevant points but lacks depth..."}
        },
        "overall_best_collection": "collection1",
        "reasoning": "Collection1 has more detailed and directly relevant information..."
        }
        """
        return prompt
