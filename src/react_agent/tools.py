"""This module provides tools for the ReAct agent.

Available tools:
1. Weather tool - Fetches real-time weather data from OpenWeatherMap API
2. RAG Retriever - Retrieves relevant information from ingested documents

These tools are intended as examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

import httpx
from langgraph.runtime import get_runtime

from react_agent.context import Context

# Add src directory to path to import RAG retriever
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

try:
    from rag.retriever import get_concatenated_results
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG retriever not available. Install required dependencies.")


async def get_weather(city: str, country_code: Optional[str] = None) -> dict[str, Any]:
    """Get current weather information for a specified city.

    This function fetches real-time weather data from the OpenWeatherMap API,
    including temperature, humidity, wind speed, and weather conditions.
    It's useful for answering questions about current weather in any city worldwide.

    Args:
        city: The name of the city to get weather for
        country_code: Optional two-letter country code (e.g., 'US', 'GB', 'IN')

    Returns:
        A dictionary containing weather information including temperature,
        humidity, wind speed, and conditions.
    """
    runtime = get_runtime(Context)
    api_key = runtime.context.openweathermap_api_key

    if not api_key:
        return {
            "error": "OpenWeatherMap API key not configured. Please set OPENWEATHERMAP_API_KEY environment variable."
        }

    # Build location string
    location = f"{city},{country_code}" if country_code else city

    # API parameters
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric",  # Use Celsius
    }

    base_url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params, timeout=10.0)
            response.raise_for_status()
            weather_data = response.json()

            # Format the response
            formatted_response = {
                "city": weather_data["name"],
                "country": weather_data["sys"]["country"],
                "temperature": weather_data["main"]["temp"],
                "feels_like": weather_data["main"]["feels_like"],
                "humidity": weather_data["main"]["humidity"],
                "description": weather_data["weather"][0]["description"],
                "wind_speed": weather_data["wind"]["speed"],
            }

            return formatted_response

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {
                "error": f"City '{city}' not found. Please check the city name or try a different city."
            }
        else:
            return {"error": f"Weather API error: HTTP {e.response.status_code}"}
    except httpx.RequestError as e:
        return {"error": f"Weather API error: {str(e)}"}


async def search_documents(query: str) -> str:
    """Search through ingested documents to find relevant information.

    This function performs semantic search over documents that have been ingested
    into the vector store. It retrieves the top 5 most relevant chunks based on the
    query and returns them as a formatted string. The search uses dense embeddings
    and LLM-based re-ranking for high-quality results.

    Use this tool when you need to:
    - Answer questions based on the ingested documents
    - Find specific information from the knowledge base
    - Retrieve context about topics covered in the documents

    Args:
        query: The search query or question to find relevant information for

    Returns:
        A formatted string containing the top 5 most relevant document chunks with
        source information. Returns an error message if the RAG system is not
        available or if the search fails.
    """
    if not RAG_AVAILABLE:
        return "Error: RAG retriever is not available. Please ensure all dependencies are installed and the vector store is populated."

    try:
        # Always retrieve exactly 5 results to keep context length manageable
        result = await asyncio.to_thread(
            get_concatenated_results,
            query=query,
            top_k=5,
            separator="\n\n--- NEXT RESULT ---\n\n"
        )
        return result

    except Exception as e:
        return f"Error retrieving documents: {str(e)}. Please ensure the vector store is populated with documents."


# Build tools list based on availability
TOOLS: List[Callable[..., Any]] = [get_weather]

if RAG_AVAILABLE:
    TOOLS.append(search_documents)
