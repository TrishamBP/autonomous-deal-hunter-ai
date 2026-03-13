"""
Messaging Agent for Deal Notifications

This agent sends alerts about discovered deals via Pushover push notifications.
Integrates with LLM to craft compelling notification messages.
"""

import os
import logging
from typing import Optional
import requests

from core.agents.base_agent import Agent
from core.agents.deals import Opportunity

logger = logging.getLogger(__name__)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    """
    Sends push notifications about deals via Pushover.
    
    Features:
    - Pushover integration for iOS/Android/desktop notifications
    - Crafts exciting alert messages
    - Supports custom messaging with Claude
    """

    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self):
        """Initialize Messaging Agent with Pushover credentials."""
        self.log("Messaging Agent is initializing")
        
        self.pushover_user = os.getenv("PUSHOVER_USER", "")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "")
        
        if not self.pushover_user or not self.pushover_token:
            self.log("Warning: Pushover credentials not configured")
        else:
            self.log("Messaging Agent ready with Pushover integration")

    def push(self, text: str, priority: int = 0) -> bool:
        """
        Send a push notification via Pushover.
        
        Args:
            text: Message text to send
            priority: Priority level (-2 to 2, default 0)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.pushover_user or not self.pushover_token:
            self.log("Pushover not configured, skipping notification")
            return False
        
        try:
            self.log("Messaging Agent is sending push notification")
            payload = {
                "user": self.pushover_user,
                "token": self.pushover_token,
                "message": text,
                "sound": "cashregister",
                "priority": priority,
            }
            response = requests.post(PUSHOVER_URL, data=payload, timeout=10)
            
            if response.status_code == 200:
                self.log("Push notification sent successfully")
                return True
            else:
                logger.error(f"Failed to send push notification: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False

    def alert(self, opportunity: Opportunity) -> bool:
        """
        Send alert about a specific opportunity.
        
        Args:
            opportunity: Opportunity instance with deal and pricing info
            
        Returns:
            True if successful, False otherwise
        """
        try:
            text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
            text += f"Est. Value=${opportunity.estimate:.2f}, "
            text += f"Save ${opportunity.discount:.2f}: "
            text += opportunity.deal.product_description[:50] + "... "
            text += opportunity.deal.url
            
            return self.push(text, priority=1)
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return False

    def craft_message(
        self, 
        description: str, 
        deal_price: float, 
        estimated_true_value: float
    ) -> str:
        """
        Use Claude to craft an exciting message about a deal.
        
        Args:
            description: Product description
            deal_price: Current offer price
            estimated_true_value: Estimated fair market value
            
        Returns:
            Crafted message (2-3 sentences)
        """
        try:
            from anthropic import Anthropic
            
            client = Anthropic()
            
            user_prompt = (
                "Please summarize this great deal in 2-3 sentences to be sent as an "
                "exciting push notification alerting the user about this deal.\n"
                f"Item Description: {description}\n"
                f"Offered Price: ${deal_price:.2f}\n"
                f"Estimated true value: ${estimated_true_value:.2f}\n\n"
                "Respond only with the 2-3 sentence message which will be used to alert "
                "& excite the user about this deal"
            )
            
            self.log("Messaging Agent is using Claude to craft message")
            
            message = client.messages.create(
                model=self.MODEL,
                max_tokens=200,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            return message.content[0].text
            
        except ImportError:
            logger.warning("Anthropic client not available, using default message")
            return f"Deal found! {description[:50]}... at ${deal_price:.2f}"
        except Exception as e:
            logger.error(f"Error crafting message with Claude: {e}")
            return f"Great deal: {description[:50]}... ${deal_price:.2f}"

    def notify(
        self, 
        description: str, 
        deal_price: float, 
        estimated_true_value: float, 
        url: str
    ) -> bool:
        """
        Send a crafted notification about specific deal details.
        
        Args:
            description: Product description
            deal_price: Current offer price
            estimated_true_value: Estimated fair market value
            url: URL to the deal
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.log("Messaging Agent is crafting notification message")
            text = self.craft_message(description, deal_price, estimated_true_value)
            
            # Truncate to ensure message fits in Pushover limit
            text = text[:200] + ("..." if len(text) > 200 else "")
            text += f"\n{url}"
            
            return self.push(text, priority=1)
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
