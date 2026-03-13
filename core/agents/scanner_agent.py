"""
Scanner Agent for Deal Discovery

This agent monitors RSS feeds for promising deals and uses OpenAI
structured outputs to identify high-quality deals with accurate prices.
"""

import logging
from typing import Optional, List
from openai import OpenAI

from core.agents.base_agent import Agent
from core.agents.deals import ScrapedDeal, DealSelection, Opportunity

logger = logging.getLogger(__name__)


class ScannerAgent(Agent):
    """
    Scans RSS feeds for promising deals.
    
    Features:
    - Fetches deals from configured RSS feeds
    - Filters out already-seen deals using memory
    - Uses OpenAI structured outputs for consistent parsing
    - Only returns deals with verified pricing information
    """

    name = "Scanner Agent"
    color = Agent.CYAN
    MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price."""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price.

Deals:

"""

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    def __init__(self):
        """Initialize Scanner Agent and OpenAI client."""
        self.log("Scanner Agent is initializing")
        try:
            self.openai = OpenAI()
            self.log("Scanner Agent is ready")
        except Exception as e:
            logger.error(f"Failed to initialize Scanner Agent: {e}")
            raise

    def fetch_deals(self, memory: List[str] = None) -> List[ScrapedDeal]:
        """
        Fetch deals from RSS feeds and filter out known deals.
        
        Args:
            memory: List of URLs representing already-processed deals
            
        Returns:
            List of new ScrapedDeal instances
        """
        if memory is None:
            memory = []
            
        self.log("Scanner Agent is fetching deals from RSS feeds")
        try:
            urls = set(memory)  # Convert to set for O(1) lookup
            scraped = ScrapedDeal.fetch(show_progress=False)
            result = [scrape for scrape in scraped if scrape.url not in urls]
            self.log(f"Scanner Agent received {len(result)} new deals")
            return result
        except Exception as e:
            logger.error(f"Failed to fetch deals: {e}")
            return []

    def make_user_prompt(self, scraped: List[ScrapedDeal]) -> str:
        """
        Create user prompt from scraped deals.
        
        Args:
            scraped: List of ScrapedDeal instances
            
        Returns:
            Formatted user prompt
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List[str] = None) -> Optional[DealSelection]:
        """
        scan RSS feeds for promising deals.
        
        Uses OpenAI structured outputs to identify the 5 best deals
        based on description quality and price clarity.
        
        Args:
            memory: List of URLs representing already-processed deals
            
        Returns:
            DealSelection with up to 5 deals, or None if no new deals found
        """
        if memory is None:
            memory = []
            
        scraped = self.fetch_deals(memory)
        
        if not scraped:
            self.log("Scanner Agent found no new deals")
            return None

        try:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling OpenAI with structured outputs")
            
            response = self.openai.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=DealSelection,
            )
            
            result = response.choices[0].message.parsed
            
            # Filter out any deals with price <= 0
            result.deals = [deal for deal in result.deals if deal.price > 0]
            
            self.log(f"Scanner Agent identified {len(result.deals)} quality deals")
            return result
            
        except Exception as e:
            logger.error(f"Failed to scan deals with OpenAI: {e}")
            return None

    def test_scan(self, memory: List[str] = None) -> Optional[DealSelection]:
        """
        Return test DealSelection for testing purposes.
        
        Args:
            memory: Unused for testing
            
        Returns:
            Test DealSelection with sample deals
        """
        from core.agents.deals import Deal
        
        return DealSelection(
            deals=[
                Deal(
                    product_description="65-inch 4K Ultra HD LED Smart TV with HDR10, Dolby Vision, and built-in smart TV platform supporting Netflix, Disney+, and other streaming services. Features 3 HDMI ports, 60Hz refresh rate, and remote control with voice search capability.",
                    price=349.99,
                    url="https://example.com/tv1",
                ),
                Deal(
                    product_description="Wireless noise-canceling headphones with 40-hour battery life, Bluetooth 5.0 connectivity, and premium sound quality. Includes carrying case and 1-year warranty.",
                    price=129.99,
                    url="https://example.com/headphones1",
                ),
            ]
        )
