from typing import Any, Annotated, Dict, List
import json
import requests
import os
import sys

ENV_API_TOKEN_NAME = 'WIKIMEDIA_API_TOKEN'
wikimedia_api_token = os.environ.get(ENV_API_TOKEN_NAME, "")
if not wikimedia_api_token:
    print(f"\033[31m[wikipedia_search tool] warning: no Wikimedia access token in env var '{ENV_API_TOKEN_NAME}'\033[0m", file=sys.stderr)

HEADERS = {
    'Authorization': f'Bearer {wikimedia_api_token}',
    'User-Agent': 'Mlux-Reactly Wikipedia-Search-Tool'
}

def query_page_titles(query: str) -> List[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 3,
    }
    response = requests.get(url, params=params, headers=HEADERS, timeout=5)
    response.raise_for_status()
    return [hit["title"] for hit in response.json()["query"]["search"]]


def load_summary(title: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    response = requests.get(url, timeout=5, headers=HEADERS)
    response.raise_for_status()
    return response.json()["extract"]



def wikipedia_search(query: Annotated[str, """search terms"""]) -> Dict[str, str]:
    """Search Wikipedia when factual or encyclopedic knowledge is needed."""

    titles = query_page_titles(query)
    summaries = [load_summary(title) for title in titles]
    return summaries


