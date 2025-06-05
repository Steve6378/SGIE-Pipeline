"""
Data Cleaning Module
Prepares scraped review data for analysis
"""

import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans and prepares review data for analysis pipeline
    """
    
    def __init__(self):
        self.required_columns = [
            'playtime_forever', 'num_games_owned', 'num_reviews',
            'votes_up', 'votes_funny', 'weighted_vote_score', 
            'comment_count', 'steam_purchase', 'review'
        ]
    
    def clean_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare review data
        
        Args:
            df: Raw review DataFrame from scraper
            
        Returns:
            Cleaned DataFrame ready for analysis
        """
        logger.info(f"Starting data cleaning on {len(df)} reviews")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Check for required columns
        missing_cols = set(self.required_columns) - set(df_clean.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove completely empty reviews
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['review'].notna()]
        df_clean = df_clean[df_clean['review'].str.strip() != '']
        removed = initial_count - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} empty reviews")
        
        # Convert timestamp if present
        if 'timestamp_created' in df_clean.columns:
            try:
                df_clean['timestamp_created'] = pd.to_datetime(df_clean['timestamp_created'])
                df_clean['review_date'] = df_clean['timestamp_created'].dt.date
                df_clean['review_hour'] = df_clean['timestamp_created'].dt.hour
            except Exception as e:
                logger.warning(f"Failed to parse timestamps: {e}")
        
        # Ensure numeric columns are numeric
        numeric_cols = [
            'playtime_forever', 'num_games_owned', 'num_reviews',
            'votes_up', 'votes_funny', 'weighted_vote_score', 'comment_count'
        ]
        
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Add review length
        df_clean['review_length'] = df_clean['review'].str.len()
        df_clean['review_words'] = df_clean['review'].str.split().str.len()
        
        # Add author_id if not present (for enhanced credibility checks)
        if 'author_id' not in df_clean.columns:
            # Create pseudo-IDs based on user stats (not perfect but helps with patterns)
            df_clean['author_id'] = (
                df_clean['num_games_owned'].astype(str) + '_' +
                df_clean['num_reviews'].astype(str) + '_' +
                df_clean.index.astype(str)
            )
        
        # Basic validation
        df_clean = df_clean[df_clean['playtime_forever'] >= 0]
        df_clean = df_clean[df_clean['weighted_vote_score'] >= 0]
        df_clean = df_clean[df_clean['weighted_vote_score'] <= 1]
        
        final_count = len(df_clean)
        logger.info(f"Data cleaning complete. {final_count} reviews retained")
        
        return df_clean
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for the dataset
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_reviews': len(df),
            'avg_playtime_hours': df['playtime_forever'].mean() / 60,
            'median_playtime_hours': df['playtime_forever'].median() / 60,
            'total_unique_reviewers': df['author_id'].nunique() if 'author_id' in df.columns else 'Unknown',
            'avg_review_length': df['review_length'].mean() if 'review_length' in df.columns else df['review'].str.len().mean(),
            'reviews_with_votes': (df['votes_up'] > 0).sum(),
            'steam_purchases': df['steam_purchase'].sum() if 'steam_purchase' in df.columns else 'Unknown',
            'date_range': {
                'earliest': df['timestamp_created'].min() if 'timestamp_created' in df.columns else 'Unknown',
                'latest': df['timestamp_created'].max() if 'timestamp_created' in df.columns else 'Unknown'
            }
        }
        
        return summary