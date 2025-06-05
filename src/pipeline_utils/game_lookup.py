"""
Steam Game Lookup Module
Provides game search and ID lookup functionality
"""

import requests
from bs4 import BeautifulSoup
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SteamGameLookup:
    """
    Handles Steam game search and ID resolution
    """
    
    def __init__(self):
        self.search_url = "https://store.steampowered.com/search/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_games(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for Steam games by name
        
        Args:
            query: Game name to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of dicts with game info: {id, name, release_date, price}
        """
        logger.info(f"Searching Steam for: {query}")
        
        params = {
            'term': query,
            'category1': '998',  # Games only
        }
        
        try:
            response = requests.get(self.search_url, params=params, headers=self.headers)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to search Steam: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Find all search result items
        search_items = soup.find_all('a', class_='search_result_row')[:max_results]
        
        for item in search_items:
            try:
                # Extract game ID from URL
                game_url = item.get('href', '')
                game_id_match = re.search(r'/app/(\d+)', game_url)
                if not game_id_match:
                    continue
                    
                game_id = int(game_id_match.group(1))
                
                # Extract game name
                title_div = item.find('span', class_='title')
                game_name = title_div.text.strip() if title_div else 'Unknown'
                
                # Extract release date
                released_div = item.find('div', class_='search_released')
                release_date = released_div.text.strip() if released_div else 'Unknown'
                
                # Extract price
                price_div = item.find('div', class_='search_price')
                if price_div:
                    price_text = price_div.text.strip()
                    # Clean up price text (remove extra whitespace)
                    price = ' '.join(price_text.split())
                else:
                    price = 'Unknown'
                
                results.append({
                    'id': game_id,
                    'name': game_name,
                    'release_date': release_date,
                    'price': price
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse search result: {e}")
                continue
        
        logger.info(f"Found {len(results)} games matching '{query}'")
        return results
    
    def validate_game_id(self, game_id: int) -> Optional[Dict]:
        """
        Validate a game ID and return basic info if valid
        
        Args:
            game_id: Steam game ID to validate
            
        Returns:
            Dict with game info if valid, None if invalid
        """
        logger.info(f"Validating game ID: {game_id}")
        
        url = f"https://store.steampowered.com/app/{game_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 404:
                logger.warning(f"Game ID {game_id} not found")
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract game name from page title or app name
            app_name = soup.find('div', class_='apphub_AppName')
            if app_name:
                game_name = app_name.text.strip()
            else:
                # Fallback to page title
                title = soup.find('title')
                if title and 'on Steam' in title.text:
                    game_name = title.text.split(' on Steam')[0].strip()
                else:
                    game_name = f"Game {game_id}"
            
            return {
                'id': game_id,
                'name': game_name,
                'url': url
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to validate game ID {game_id}: {e}")
            return None
    
    def resolve_input(self, user_input: str) -> Optional[int]:
        """
        Resolve user input to a game ID
        Handles both direct ID input and game name search
        
        Args:
            user_input: Either a game ID or game name
            
        Returns:
            Game ID if resolved, None if not found
        """
        # Check if input is a number (direct ID)
        if user_input.isdigit():
            game_id = int(user_input)
            game_info = self.validate_game_id(game_id)
            if game_info:
                logger.info(f"Validated game ID {game_id}: {game_info['name']}")
                return game_id
            else:
                logger.warning(f"Invalid game ID: {game_id}")
                return None
        
        # Otherwise, search for the game
        search_results = self.search_games(user_input, max_results=1)
        if search_results:
            game = search_results[0]
            logger.info(f"Found game: {game['name']} (ID: {game['id']})")
            return game['id']
        else:
            logger.warning(f"No games found matching: {user_input}")
            return None
    
    def interactive_search(self, query: str) -> Optional[int]:
        """
        Perform interactive search with user selection
        
        Args:
            query: Game name to search for
            
        Returns:
            Selected game ID or None if cancelled
        """
        results = self.search_games(query, max_results=10)
        
        if not results:
            print(f"No games found matching '{query}'")
            return None
        
        print(f"\nFound {len(results)} games matching '{query}':\n")
        for i, game in enumerate(results, 1):
            print(f"{i}. {game['name']} ({game['release_date']}) - {game['price']}")
        
        print("\n0. Cancel search")
        
        while True:
            try:
                choice = input("\nSelect a game (enter number): ").strip()
                if choice == '0':
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    selected = results[idx]
                    print(f"\nSelected: {selected['name']} (ID: {selected['id']})")
                    return selected['id']
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nSearch cancelled.")
                return None