from utilities.utils import find_syllabus, read_and_split_file, compute_similarity, generate_query_passage_pairs
from utilities.constants import full_course_info

import os
import requests
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv(dotenv_path=r'D:\CSCI544_project_code\.env')


def check_params(query, model, reranker, database):
    missing_params = []
    if query is None:
        missing_params.append('query')
    if model is None:
        missing_params.append('model')
    if reranker is None:
        missing_params.append('reranker')
    if database is None:
        missing_params.append('database')

    if missing_params:
        raise ValueError(f"The following parameter(s) are missing: {', '.join(missing_params)}")


def RAG(**kwargs):
    """
    The RAG retrival function. Retrieve three most relevant segments in the syllabus.
    :return: The prompt contains the course information
    """
    query = kwargs.get('query', None)
    model = kwargs.get('model', None)
    reranker = kwargs.get('reranker', None)
    database = kwargs.get('database', None)
    check_params(query, model, reranker, database)

    k = 3
    cos_simi_thred = 0.5  # shown by experiments, 0.5 is the best

    final_syllabus = find_syllabus(query, model, database)
    # print(final_syllabus)

    file_path = r'D:\CSCI544_project_code\syllabus' + os.sep + final_syllabus
    knowledge_base = read_and_split_file(file_path)

    query = [query]
    all_similarity = compute_similarity(query, knowledge_base, model)

    results = []
    for score, segment in all_similarity:
        if score > cos_simi_thred:
            results.append(segment)

    query_segment_pairs = generate_query_passage_pairs(query, results)

    if query_segment_pairs:
        scores = reranker.compute_score(query_segment_pairs)
        sorted_query_segment_pairs = sorted(zip(scores, query_segment_pairs), key=lambda x: x[0], reverse=True)

        sorted_results = [(score, pair[1]) for score, pair in sorted_query_segment_pairs]

        final_results = []
        for score, passage in sorted_results:
            # if score > 0:  # no threshold for reranker results -> not necessary
            final_results.append((score, passage))

        final_results = final_results[:k]
    else:
        final_results = [(0.0, 'There is no relevant information for the given query!')]

    all_segments = '\n\n'.join([result[1] for result in final_results])
    class_found = final_syllabus[:-len('.txt')]

    return f'Here are class information for {full_course_info[class_found]}:\n{all_segments}'


def general_news_report(**kwargs):
    """
    Fetches news articles based on the user's query.
    :return: A formatted string containing the titles and descriptions of the retrieved articles.
    """
    query = kwargs.get('query', None)
    if query is None:
        raise ValueError('Your query cannot be empty!')

    # Retrieve the News API key from user data or configuration
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    max_articles = 5
    # News API endpoint
    endpoint = 'https://newsapi.org/v2/everything'

    # Set query parameters
    params = {
        'q': query,                # Use the user's query
        'language': 'en',          # Language preference
        'sortBy': 'publishedAt',   # Order by most recent
        'apiKey': NEWS_API_KEY,
    }

    # Send request to News API
    response = requests.get(endpoint, params=params)

    # Check for successful response
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])[:max_articles]  # Limit the number of articles
        final_res = ""

        # Parse and append each article's title and description
        for article in articles:
            title = article.get('title', 'No Title')
            description = article.get('description', 'No Description')
            final_res += f"Title: {title}\n"
            final_res += f"Description: {description}\n\n"

        # Return the formatted string
        return f'Here are news for {query} today:\n\n' + final_res.strip()

    else:
        # Handle errors and return the response status
        error_message = f"Failed to retrieve news: {response.status_code} - {response.reason}"
        # print(error_message)
        return error_message


def fetch_weather(**kwargs) -> str:
    """
    Fetch weather information based on the provided location and date. Can only predict 3 days (maximum) in the future.
    :return: A stringified dictionary containing the weather details or an error/warning message.
    """
    params = kwargs.get('params', None)  # e.g., 'Los Angeles, December 5, 2024'
    if params is None:
        raise ValueError('Your query cannot be empty!')
    elif not isinstance(params, str):
        return "Error: Invalid input format. Expected a string 'Location, Date'."

    try:
        # Split the input into location and date
        location, date_str = map(str.strip, params.split(',', 1))
        # Parse the date into a standardized format
        query_date = datetime.strptime(date_str, "%B %d, %Y").date()
    except ValueError:
        return "Error: 'params' format is invalid. Expected format: 'Location, Month Day, Year'."

    # Validate the query date (ensure it is not beyond the forecast range)
    today = datetime.now().date()
    max_forecast_date = today + timedelta(days=3)  # wttr.in provides a 3-day forecast
    if query_date > max_forecast_date:
        return f"Warning: Weather data for {query_date} is not available. Please provide a date within the next 3 days."

    # Make the API call to wttr.in
    try:
        response = requests.get(f"https://wttr.in/{location}?format=j1", timeout=10)
        response.raise_for_status()
        weather_data = response.json()
    except requests.RequestException as e:
        return f"Error fetching weather data: {e}"

    # Handle current and forecast weather
    try:
        if query_date == today:
            # Return current weather
            current_condition = weather_data.get("current_condition", [{}])[0]
            result = {
                "FeelsLikeC": current_condition.get("FeelsLikeC", "N/A"),
                "temp_C": current_condition.get("temp_C", "N/A"),
                "weatherDesc": current_condition.get("weatherDesc", [{}])[0].get("value", "N/A"),
                "humidity": current_condition.get("humidity", "N/A"),
            }
            return f"The current weather in {location} is: {result}"

        else:
            # Return forecast weather
            forecast = weather_data.get("weather", [])
            for day_forecast in forecast:
                forecast_date = datetime.strptime(day_forecast["date"], "%Y-%m-%d").date()

                if forecast_date == query_date:
                    hourly_data = day_forecast.get("hourly", [{}])
                    result = {
                        "max_temp": day_forecast.get("maxtempC", "N/A"),
                        "min_temp": day_forecast.get("mintempC", "N/A"),
                        "description": hourly_data[0].get("weatherDesc", [{}])[0].get("value", "N/A"),
                    }
                    return f"The forecast weather in {location} on {query_date} is: {result}"

            return f"Weather forecast for {query_date} is not available. Please try another date."

    except KeyError as e:
        return f"Error processing weather data: Key {e} not found in the response."


if __name__ == '__main__':
    args = {'query': "Los Angeles",
            'params': "Los Angeles, December 16, 2024"}

    # print(general_news_report(**args))
    print(fetch_weather(**args))
