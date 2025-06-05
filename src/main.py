"""
Steam Games Insight Engine - Main Pipeline
Orchestrates the complete analysis pipeline for Steam game reviews
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
from tqdm import tqdm

from pipeline_utils import SteamGameLookup, DataCleaner, CredibilityFilter

# Configure logging
def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class SteamGamesPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.viz_dir = self.base_dir / "visualizations"
        
        # Create directories
        for dir_path in [self.data_dir / "raw", self.data_dir / "cleaned", 
                         self.data_dir / "results", self.logs_dir,
                         self.viz_dir / "credibility", self.viz_dir / "absa", 
                         self.viz_dir / "emotions"]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.logs_dir)
        self.logger.info("Steam Games Pipeline initialized")
        
        # Initialize components
        self.game_lookup = SteamGameLookup()
        self.data_cleaner = DataCleaner()
        self.credibility_filter = CredibilityFilter()
        
        # Pipeline configuration
        self.min_reviews = 1000
        self.min_reviews_absolute = 500
        
    def resolve_game(self, user_input: str, interactive: bool = True) -> Optional[Dict]:
        """Resolve user input to game information"""
        self.logger.info(f"Resolving game input: {user_input}")
        
        # Try direct resolution first
        game_id = self.game_lookup.resolve_input(user_input)
        
        if game_id:
            game_info = self.game_lookup.validate_game_id(game_id)
            if game_info:
                return game_info
        
        # If failed and interactive mode, try interactive search
        if interactive and not user_input.isdigit():
            self.logger.info("Attempting interactive search")
            game_id = self.game_lookup.interactive_search(user_input)
            if game_id:
                return self.game_lookup.validate_game_id(game_id)
        
        self.logger.error(f"Could not resolve game: {user_input}")
        return None
    
    def scrape_reviews(self, game_id: int, game_name: str) -> Optional[pd.DataFrame]:
        """Scrape reviews for the given game"""
        self.logger.info(f"Starting review scraping for {game_name} (ID: {game_id})")
        
        try:
            # Import the local scraper
            from pipeline_utils.web_scraper import WebScraperWrapper
            
            # Initialize scraper
            web_scraper = WebScraperWrapper()
            
            # Scrape reviews with progress tracking
            print(f"\nScraping reviews for {game_name}...")
            reviews_df = web_scraper.scrape_reviews(game_id)
            
            if reviews_df is None or len(reviews_df) == 0:
                self.logger.error("No reviews scraped")
                return None
            
            self.logger.info(f"Scraped {len(reviews_df)} reviews")
            
            # Check minimum review count
            if len(reviews_df) < self.min_reviews_absolute:
                self.logger.warning(f"Only {len(reviews_df)} reviews found. Minimum is {self.min_reviews_absolute}")
                print(f"\n⚠️  Warning: Only {len(reviews_df)} reviews found. Results may not be representative.")
                if len(reviews_df) < self.min_reviews_absolute:
                    print(f"❌ Too few reviews for meaningful analysis (minimum: {self.min_reviews_absolute})")
                    return None
            elif len(reviews_df) < self.min_reviews:
                print(f"\n⚠️  Note: {len(reviews_df)} reviews found. Recommended minimum is {self.min_reviews} for best results.")
            
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file = self.data_dir / "raw" / f"{game_id}_{timestamp}_raw.csv"
            reviews_df.to_csv(raw_file, index=False)
            self.logger.info(f"Saved raw data to {raw_file}")
            
            return reviews_df
            
        except Exception as e:
            self.logger.error(f"Failed to scrape reviews: {e}")
            return None
    
    def process_reviews(self, reviews_df: pd.DataFrame, game_id: int) -> Dict[str, pd.DataFrame]:
        """Process reviews through cleaning and credibility filtering"""
        results = {}
        
        try:
            # Step 1: Data cleaning
            self.logger.info("Starting data cleaning")
            cleaned_df = self.data_cleaner.clean_reviews(reviews_df)
            results['cleaned'] = cleaned_df
            
            # Generate data summary
            summary = self.data_cleaner.get_data_summary(cleaned_df)
            self.logger.info(f"Data summary: {json.dumps(summary, indent=2, default=str)}")
            
            # Step 2: Credibility filtering
            self.logger.info("Starting credibility filtering")
            credible_df, fake_df = self.credibility_filter.filter_reviews(cleaned_df)
            results['credible'] = credible_df
            results['fake'] = fake_df
            
            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cleaned_file = self.data_dir / "cleaned" / f"{game_id}_{timestamp}_cleaned.csv"
            credible_file = self.data_dir / "cleaned" / f"{game_id}_{timestamp}_credible.csv"
            fake_file = self.data_dir / "cleaned" / f"{game_id}_{timestamp}_fake.csv"
            
            cleaned_df.to_csv(cleaned_file, index=False)
            credible_df.to_csv(credible_file, index=False)
            if len(fake_df) > 0:
                fake_df.to_csv(fake_file, index=False)
            
            self.logger.info(f"Saved cleaned data: {len(cleaned_df)} reviews")
            self.logger.info(f"Saved credible data: {len(credible_df)} reviews")
            self.logger.info(f"Saved fake data: {len(fake_df)} reviews")
            
            # Generate credibility summary
            if len(fake_df) > 0:
                fake_summary = self.credibility_filter.get_fake_review_summary(fake_df)
                self.logger.info(f"Fake review summary: {json.dumps(fake_summary, indent=2)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process reviews: {e}")
            return results
    
    def run_pipeline(self, user_input: str, interactive: bool = True, 
                    skip_scraping: bool = False, existing_data_path: Optional[str] = None):
        """Run the complete analysis pipeline"""
        start_time = datetime.now()
        self.logger.info(f"Starting pipeline run at {start_time}")
        
        # Step 1: Resolve game
        game_info = self.resolve_game(user_input, interactive)
        if not game_info:
            print("❌ Could not find the specified game.")
            return None
        
        print(f"\n✅ Found game: {game_info['name']} (ID: {game_info['id']})")
        
        # Step 2: Get review data
        if skip_scraping and existing_data_path:
            self.logger.info(f"Loading existing data from {existing_data_path}")
            try:
                reviews_df = pd.read_csv(existing_data_path)
                print(f"✅ Loaded {len(reviews_df)} reviews from file")
            except Exception as e:
                self.logger.error(f"Failed to load existing data: {e}")
                return None
        else:
            reviews_df = self.scrape_reviews(game_info['id'], game_info['name'])
            if reviews_df is None:
                return None
        
        # Step 3: Process reviews
        print("\n🔄 Processing reviews...")
        processed_data = self.process_reviews(reviews_df, game_info['id'])
        
        if 'credible' not in processed_data:
            print("❌ Failed to process reviews")
            return None
        
        credible_df = processed_data['credible']
        print(f"✅ Processing complete: {len(credible_df)} credible reviews retained")
        
        # Prepare for next steps
        pipeline_state = {
            'game_info': game_info,
            'raw_reviews': len(reviews_df),
            'credible_reviews': len(credible_df),
            'fake_reviews': len(processed_data.get('fake', [])),
            'data_files': {
                'credible': credible_df,
                'fake': processed_data.get('fake', pd.DataFrame())
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save pipeline state
        state_file = self.data_dir / "results" / f"{game_info['id']}_pipeline_state.json"
        with open(state_file, 'w') as f:
            state_data = {k: v for k, v in pipeline_state.items() if k != 'data_files'}
            json.dump(state_data, f, indent=2)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.logger.info(f"Pipeline completed in {duration:.2f} seconds")
        
        print(f"\n✅ Initial pipeline stages complete!")
        print(f"   Total time: {duration:.2f} seconds")
        print(f"   Raw reviews: {len(reviews_df)}")
        print(f"   Credible reviews: {len(credible_df)} ({len(credible_df)/len(reviews_df)*100:.1f}%)")
        print(f"   Fake reviews filtered: {len(processed_data.get('fake', []))} ({len(processed_data.get('fake', []))/len(reviews_df)*100:.1f}%)")
        
        return pipeline_state


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Steam Games Insight Engine - Analyze Steam game reviews"
    )
    parser.add_argument(
        "game",
        help="Steam game ID or game name to analyze"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive game selection"
    )
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Skip scraping and use existing data"
    )
    parser.add_argument(
        "--data-file",
        help="Path to existing review data CSV (requires --skip-scraping)"
    )
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = SteamGamesPipeline()
    
    # Run pipeline
    result = pipeline.run_pipeline(
        args.game,
        interactive=not args.non_interactive,
        skip_scraping=args.skip_scraping,
        existing_data_path=args.data_file
    )
    
    if result:
        print("\n🎯 Next steps: Run ABSA and emotion analysis modules")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()