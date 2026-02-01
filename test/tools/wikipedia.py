from typing import Any, Annotated, Dict, List
from dataclasses import dataclass
import json
import requests
import os
import sys
import wikipedia


NR_PAGES = 3

def wikipedia_search(query: Annotated[str, """search terms"""]) -> Dict[str, str]:
    """Search Wikipedia when factual or encyclopedic knowledge is needed. Will return a list of looked-up pages and and a list of additional page titles that are similar to the query."""
    print(f'[tool] wikipedia_search called: {query}')

    titles = wikipedia.search(query)
    print(f"[tool] wikipedia_search titles: {titles}")

    pages = []
    for title in titles[:NR_PAGES]:
        try:
            page = wikipedia.page(title)
            pages.append({
                'page_title': page.title,
                'page_content': page.content
            })
        except:
            pass

    return {
        'results': pages,
        'similar_page_titles': titles[NR_PAGES:]
    }


