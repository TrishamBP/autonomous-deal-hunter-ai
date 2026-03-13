"""
Deal Models and Fetching

This module provides data models for deals scraped from RSS feeds and
the logic for extracting and structuring deal information.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Self
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time
import logging

logger = logging.getLogger(__name__)

# RSS feeds for deal discovery
DEAL_FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]


def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up HTML snippet and extract useful text.
    
    Args:
        html_snippet: HTML content to parse
        
    Returns:
        Cleaned text content
    """
    try:
        soup = BeautifulSoup(html_snippet, "html.parser")
        snippet_div = soup.find("div", class_="snippet summary")

        if snippet_div:
            description = snippet_div.get_text(strip=True)
            description = BeautifulSoup(description, "html.parser").get_text()
            description = re.sub("<[^<]+?>", "", description)
            result = description.strip()
        else:
            result = html_snippet
        return result.replace("\n", " ")
    except Exception as e:
        logger.warning(f"Failed to extract HTML snippet: {e}")
        return html_snippet.replace("\n", " ")


class ScrapedDeal:
    """
    Represents a deal retrieved from an RSS feed.
    
    Extracts title, summary, URL, and detailed content from feed entries.
    """

    def __init__(self, entry: Dict[str, str]):
        """
        Initialize ScrapedDeal from feed entry dictionary.
        
        Args:
            entry: RSS feed entry dictionary
        """
        self.title = entry.get("title", "Unknown")
        self.summary = extract(entry.get("summary", ""))
        
        # Extract URL
        links = entry.get("links", [])
        self.url = links[0]["href"] if links else ""
        
        # Fetch and parse content from URL
        self.details = ""
        self.features = ""
        
        try:
            if self.url:
                response = requests.get(self.url, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")
                content_section = soup.find("div", class_="content-section")
                
                if content_section:
                    content = content_section.get_text()
                    content = content.replace("\nmore", "").replace("\n", " ")
                    
                    if "Features" in content:
                        self.details, self.features = content.split("Features", 1)
                    else:
                        self.details = content
                        self.features = ""
                else:
                    self.details = self.summary
        except Exception as e:
            logger.warning(f"Failed to fetch content from {self.url}: {e}")
            self.details = self.summary
        
        self.truncate()

    def truncate(self):
        """Limit fields to sensible length to avoid excessive token usage."""
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<ScrapedDeal: {self.title}>"

    def describe(self) -> str:
        """
        Return formatted description for use in prompts.
        
        Returns:
            Formatted deal description
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from RSS feeds.
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            List of ScrapedDeal instances
        """
        deals = []
        feed_iter = tqdm(DEAL_FEEDS, desc="Fetching deals") if show_progress else DEAL_FEEDS
        
        for feed_url in feed_iter:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    try:
                        deals.append(cls(entry))
                        time.sleep(0.05)  # Rate limiting
                    except Exception as e:
                        logger.warning(f"Failed to create deal from entry: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Failed to parse feed {feed_url}: {e}")
                continue
        
        return deals


class Deal(BaseModel):
    """
    Pydantic model representing a deal with structured fields.
    
    Fields:
        product_description: Clear summary of the product (3-4 sentences)
        price: Actual advertised price of the product
        url: URL to the deal
    """

    product_description: str = Field(
        description="Clear summary of the product in 3-4 sentences. Focus on the item itself, not the deal terms. Avoid mentioning discounts or coupons."
    )
    price: float = Field(
        description="Actual advertised price of the product. If discounted (e.g., $100 off $300), report the final price ($200)."
    )
    url: str = Field(description="URL of the deal")


class DealSelection(BaseModel):
    """
    Pydantic model representing a selection of deals.
    
    Used with structured outputs from OpenAI to ensure consistent formatting.
    """

    deals: List[Deal] = Field(
        description="List of 5 deals with most detailed descriptions and clear prices"
    )


class Opportunity(BaseModel):
    """
    Represents a market opportunity: a deal where estimated value exceeds asking price.
    
    Fields:
        deal: The deal information
        estimate: Estimated fair market value
        discount: Difference between estimate and deal price
    """

    deal: Deal
    estimate: float = Field(description="Estimated fair market value")
    discount: float = Field(description="Discount (estimate - deal price)")
