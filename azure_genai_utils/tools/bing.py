import os
import json
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Literal, Union, Optional, List
from langchain_core.callbacks import CallbackManagerForToolRun

DEFAULT_BING_WEB_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
DEFAULT_BING_NEWS_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/news/search"
DEFAULT_BING_ENTITY_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/entities"


def get_webpage_context(url: str) -> Optional[str]:
    """
    Fetch the main content from a webpage.
    Args:
        url (str): The URL of the webpage.
    Returns:
        str: Extracted content from the webpage.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        # print(f"Detected encoding: {response.encoding}")
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract main content (simple heuristic using <p> tags)
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs])

        # Limit content to avoid excessive data
        return content
    except requests.RequestException as e:
        print(f"Error fetching context from {url}: {e}")
        return None


def tag_search_result(result: dict):
    """
    Format search results as XML.
    Args:
        result (dict): The original search result.
    Returns:
        str: The search result formatted as XML.
    """

    if result["kind"] == "web":
        title = json.dumps(result["title"], ensure_ascii=False)[1:-1]
        snippet = json.dumps(result["snippet"], ensure_ascii=False)[1:-1]
        content = json.dumps(result["content"], ensure_ascii=False)[1:-1]
        url = result["url"]
        # thumbnail_url = result["thumbnail_url"]
        source = result["source"]
        tag_result = (
            f"<document><kind>web</kind><title>{title}</title>"
            f"<snippet>{snippet}</snippet><content>{content}</content>"
            f"<url>{url}</url><source>{source}</source></document>"
        )
    elif result["kind"] == "news":
        title = json.dumps(result["title"], ensure_ascii=False)[1:-1]
        snippet = json.dumps(result["snippet"], ensure_ascii=False)[1:-1]
        content = json.dumps(result["content"], ensure_ascii=False)[1:-1]
        url = result["url"]
        image = result["image"]
        tag_result = (
            f"<document><kind>news</kind><title>{title}</title>"
            f"<snippet>{snippet}</snippet><content>{content}</content>"
            f"<url>{url}</url><image>{image}</image></document>"
        )
    elif result["kind"] == "entity":
        name = json.dumps(result["name"], ensure_ascii=False)[1:-1]
        content = json.dumps(result["content"], ensure_ascii=False)[1:-1]
        url = result["url"]
        image = result["image"]
        info = json.dumps(result["info"], ensure_ascii=False)[1:-1]
        tag_result = (
            f"<document><kind>entity</kind><name>{name}</name>"
            f"<content>{content}</content><info>{info}</info>"
            f"<url>{url}</url><image>{image}</image></document>"
        )
    else:
        tag_result = ""

    return tag_result


class BingSearchInput(BaseModel):
    """Input for the BingSearch tool."""

    query: str = Field(description="Search query")


class BingSearch(BaseTool):
    """
    Tool that queries the Bing Search API and gets back json
    """

    name: str = "bing_search_results"
    description: str = (
        "A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. [IMPORTANT] Input(query) should be over 5 characters."
    )

    args_schema: type[BaseModel] = BingSearchInput

    api_key: Optional[str] = None
    max_results: int = 3
    locale: str = "en-US"
    include_news: bool = False
    include_entity: bool = False
    news_freshness: Optional[Literal["Day", "Week", "Month"]] = None
    topic: Literal["general", "news"] = "general"

    format_output: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 3,
        locale: str = "en-US",
        include_news: bool = False,
        include_entity: bool = False,
        news_freshness: Optional[Literal["Day", "Week", "Month"]] = None,
        format_output: bool = False,
    ):
        super().__init__()

        if api_key is None:
            api_key = os.environ.get("BING_SUBSCRIPTION_KEY", None)

        if api_key is None:
            raise ValueError("bing_subscription_key is not set.")

        self.name = "bing_search_results"
        self.api_key = api_key
        self.max_results = max_results
        self.locale = locale
        self.include_news = include_news
        self.include_entity = include_entity
        self.news_freshness = news_freshness
        self.format_output = format_output

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Union[List[dict], List[str]]:
        """Implementing BaseTool's _run(...)"""
        websearch_results = self._bing_websearch_results(
            query, max_results=self.max_results, locale=self.locale
        )

        newssearch_results = []
        entitysearch_results = []

        if self.include_news:
            newssearch_results = self._bing_newssearch_results(
                query,
                max_results=self.max_results,
                locale=self.locale,
                news_freshness=self.news_freshness,
            )
        if self.include_entity:
            entitysearch_results = self._bing_entitysearch_results(
                query, max_results=self.max_results, locale="en-US"
            )
            print(entitysearch_results)

        results = websearch_results + newssearch_results + entitysearch_results

        if self.format_output:
            return [tag_search_result(r) for r in results]
        else:
            return results

    def _bing_websearch_results(
        self, query: str, max_results: int, locale: str, **kwargs
    ) -> List[dict]:
        """
        Get Bing Web search results.
        """
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": max_results,
            "textDecorations": True,
            "textFormat": "HTML",
            "mkt": locale,
            **kwargs,
        }

        response = requests.get(
            DEFAULT_BING_WEB_SEARCH_ENDPOINT,
            headers=headers,
            params=params,
            timeout=20,  # Add a timeout of 20 seconds
        )
        response.raise_for_status()
        search_results = response.json()

        metadata_web_results = []
        if "webPages" in search_results:
            for result in search_results["webPages"]["value"]:
                metadata_result = {
                    "kind": "web",
                    "title": result["name"],
                    "snippet": result["snippet"],
                    "content": (
                        get_webpage_context(result["url"])
                        if "url" in result
                        else result["snippet"]
                    ),
                    "url": result["url"] if "url" in result else None,
                    "thumbnail_url": (
                        result["thumbnailUrl"] if "thumbnailUrl" in result else None
                    ),
                    "source": result["siteName"] if "siteName" in result else None,
                }
                metadata_web_results.append(metadata_result)

        return metadata_web_results

    def _bing_newssearch_results(
        self,
        query: str,
        max_results: int,
        locale: str,
        news_freshness: Optional[Literal["Day", "Week", "Month"]],
        **kwargs,
    ) -> List[dict]:
        """
        Get Bing News search results.
        """
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": max_results,
            "textDecorations": True,
            "textFormat": "HTML",
            "mkt": locale,
            **kwargs,
        }
        if news_freshness is not None:
            params["freshness"] = news_freshness

        response = requests.get(
            DEFAULT_BING_NEWS_SEARCH_ENDPOINT,
            headers=headers,
            params=params,
            timeout=20,  # Add a timeout of 10 seconds
        )
        response.raise_for_status()
        search_results = response.json()

        metadata_news_results = []
        if "value" in search_results:
            for result in search_results["value"]:
                metadata_result = {
                    "kind": "news",
                    "title": result["name"],
                    "snippet": result["description"],
                    "content": (
                        get_webpage_context(result["url"])
                        if "url" in result
                        else result["description"]
                    ),
                    "url": result["url"] if "url" in result else None,
                    "image": (
                        json.dumps(result["image"], ensure_ascii=False)
                        if "image" in result
                        else None
                    ),
                }
                metadata_news_results.append(metadata_result)

        return metadata_news_results

    def _bing_entitysearch_results(
        self, query: str, max_results: int, locale: str, **kwargs
    ) -> List[dict]:
        """
        Get Bing Entity search results.
        """
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": max_results,
            "textDecorations": True,
            "textFormat": "HTML",
            "mkt": "en-US",  # entity search usually works in en-US region.
            **kwargs,
        }
        response = requests.get(
            DEFAULT_BING_ENTITY_SEARCH_ENDPOINT,
            headers=headers,
            params=params,
            timeout=20,  # Add a timeout of 20 seconds
        )
        response.raise_for_status()
        search_results = response.json()

        metadata_entity_results = []
        if "entities" in search_results:
            for result in search_results["entities"]["value"]:
                metadata_result = {
                    "kind": "entity",
                    "name": result["name"],
                    "content": result["description"],
                    "url": result["url"] if "url" in result else None,
                    "image": (
                        json.dumps(result["image"], ensure_ascii=False)
                        if "image" in result
                        else None
                    ),
                    "info": (
                        json.dumps(result["entityPresentationInfo"], ensure_ascii=False)
                        if "entityPresentationInfo" in result
                        else None
                    ),
                }
                metadata_entity_results.append(metadata_result)

        return metadata_entity_results
