"""
Weather Agent using OpenWeatherMap API
Fetches real-time weather data and provides natural language responses
"""

import requests
from typing import Dict, Optional
from src.config import settings
from src.models.llm import get_llm


class WeatherAgent:
    """
    Agent for fetching and processing weather data
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Weather Agent

        Args:
            api_key: OpenWeatherMap API key
        """
        self.api_key = api_key or settings.openweathermap_api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.llm = get_llm(temperature=0.3)

    def fetch_weather(self, city: str, country_code: str = None) -> Dict:
        """
        Fetch weather data from OpenWeatherMap API

        Args:
            city: City name
            country_code: Optional country code (e.g., 'US', 'GB')

        Returns:
            Weather data dictionary

        Raises:
            Exception: If API call fails
        """
        # Build location string
        location = f"{city},{country_code}" if country_code else city

        # API parameters
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric",  # Use Celsius
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Weather API error: {str(e)}")

    def format_weather_data(self, weather_data: Dict) -> str:
        """
        Format raw weather data into a readable string

        Args:
            weather_data: Raw weather data from API

        Returns:
            Formatted weather information
        """
        try:
            city = weather_data["name"]
            country = weather_data["sys"]["country"]
            temp = weather_data["main"]["temp"]
            feels_like = weather_data["main"]["feels_like"]
            humidity = weather_data["main"]["humidity"]
            description = weather_data["weather"][0]["description"]
            wind_speed = weather_data["wind"]["speed"]

            formatted = f"""
Weather Information for {city}, {country}:
- Temperature: {temp}°C (Feels like: {feels_like}°C)
- Condition: {description.capitalize()}
- Humidity: {humidity}%
- Wind Speed: {wind_speed} m/s
"""
            return formatted.strip()
        except KeyError as e:
            return f"Error formatting weather data: Missing key {str(e)}"

    def get_weather(self, city: str, country_code: str = None) -> str:
        """
        Get weather information and return formatted response

        Args:
            city: City name
            country_code: Optional country code

        Returns:
            Formatted weather information
        """
        weather_data = self.fetch_weather(city, country_code)
        return self.format_weather_data(weather_data)

    def get_weather_with_llm(self, query: str) -> str:
        """
        Get weather information with LLM-enhanced response

        Args:
            query: Natural language query (e.g., "What's the weather in London?")

        Returns:
            Natural language weather response
        """
        # Extract city from query using LLM
        extraction_prompt = f"""Extract the city name from this query.

Rules:
- If a city is mentioned, return ONLY the city name
- If a country code is mentioned, include it in format: city,country_code
- If NO city is mentioned, return exactly: "NO_CITY"

Query: {query}

City:"""

        city_response = self.llm.invoke(extraction_prompt)
        city = city_response.content.strip()

        # Check if no city was found
        if city.upper() == "NO_CITY" or "no city" in city.lower():
            return "I'd be happy to tell you about the weather! Could you please specify which city you're interested in?"

        # Handle city,country format
        parts = city.split(",")
        city_name = parts[0].strip()
        country_code = parts[1].strip() if len(parts) > 1 else None

        # Validate city name is reasonable
        if len(city_name) < 2 or len(city_name) > 100:
            return "I couldn't identify a valid city name in your query. Could you please specify which city you'd like to know about?"

        try:
            # Fetch weather data
            weather_data = self.fetch_weather(city_name, country_code)
            formatted_weather = self.format_weather_data(weather_data)

            # Generate natural language response
            response_prompt = f"""Based on the following weather data, provide a natural, conversational response to the user's query.

User Query: {query}

Weather Data:
{formatted_weather}

Provide a helpful, concise response:"""

            llm_response = self.llm.invoke(response_prompt)
            return llm_response.content
        except Exception as e:
            if "404" in str(e):
                return f"I couldn't find weather information for '{city_name}'. Could you please check the city name or try a different city?"
            else:
                raise


def create_weather_agent() -> WeatherAgent:
    """Factory function to create a WeatherAgent instance"""
    return WeatherAgent()
