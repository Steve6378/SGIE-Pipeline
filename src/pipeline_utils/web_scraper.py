"""
Web Scraper Wrapper Module
Provides a clean interface to the local web scraping module
"""

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class WebScraperWrapper:
    """
    Wrapper for the web scraping module to provide consistent interface
    """
    
    def __init__(self):
        try:
            from .web_scraping import scrape_steam_game
            self.scraper_class = scrape_steam_game
            logger.info("Web scraping module loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import web scraping module: {e}")
            raise
    
    def scrape_reviews(self, game_id: int, progress_callback=None) -> Optional[pd.DataFrame]:
        """
        Scrape reviews for a given game ID
        
        Args:
            game_id: Steam game ID
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with reviews or None if failed
        """
        try:
            logger.info(f"Initializing scraper for game ID: {game_id}")
            scraper = self.scraper_class(game_id)
            
            logger.info("Starting review scraping...")
            reviews_df = scraper.scrape_review_info()
            
            if reviews_df is None or len(reviews_df) == 0:
                logger.warning("No reviews scraped")
                return None
            
            logger.info(f"Successfully scraped {len(reviews_df)} reviews")
            return reviews_df
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            return None
    
    def scrape_guides(self, game_id: int) -> Optional[list]:
        """
        Scrape guides for a given game ID
        
        Args:
            game_id: Steam game ID
            
        Returns:
            List of guides or None if failed
        """
        try:
            logger.info(f"Initializing scraper for guides - game ID: {game_id}")
            scraper = self.scraper_class(game_id)
            
            logger.info("Starting guide scraping...")
            guides = scraper.scrape_guides()
            
            if guides is None:
                logger.warning("No guides scraped")
                return None
            
            logger.info(f"Successfully scraped {len(guides)} guides")
            return guides
            
        except Exception as e:
            logger.error(f"Error during guide scraping: {e}")
            return None
    
    def get_scraper_params(self) -> dict:
        """Get default scraper parameters"""
        return {
            "json": 1,
            "language": 'english',
            "cursor": "*",
            "num_per_page": 100,
            "filter": "recent"
        }