from typing import List, Annotated, Dict
from ddgs import DDGS


def web_search(query: Annotated[str, """The search query"""]) -> List[Dict[str, str]]:
    """Tool for performing a web search."""

    with DDGS() as ddgs:
        duck_results = ddgs.text(query, max_results=10)
    results = [{'result_title': r.get('title'), 'url': r.get('href'), 'content_snippet': r.get('body')} for r in duck_results]
    return results
