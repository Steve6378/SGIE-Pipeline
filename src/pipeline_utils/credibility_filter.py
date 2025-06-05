"""
Credibility Detection Module
Implements enhanced y3 rules for filtering fake/uncredible reviews
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class CredibilityFilter:
    """
    Filters out potentially fake/uncredible reviews using enhanced y3 rules
    """
    
    def __init__(self, fake_rate_threshold=0.03):
        self.fake_rate_threshold = fake_rate_threshold
        self.scaler = StandardScaler()
        self.knn_model = None
        
    def prepare_features(self, df):
        """Prepare features for KNN model"""
        # Log transform weighted_vote_score
        df = df.copy()
        df['weighted_vote_score'] = np.log2(df['weighted_vote_score'].clip(lower=0.01))
        
        # Select numeric columns
        numeric_cols = [
            'playtime_forever', 'num_games_owned', 'num_reviews', 
            'votes_up', 'votes_funny', 'weighted_vote_score', 'comment_count'
        ]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(df[numeric_cols])
        
        # One-hot encode sentiment scores if available
        if 'sentiment_score' in df.columns:
            sentiment_encoded = pd.get_dummies(df['sentiment_score'])
            features = pd.concat([
                pd.DataFrame(scaled_features, columns=numeric_cols, index=df.index),
                sentiment_encoded
            ], axis=1)
        else:
            features = pd.DataFrame(scaled_features, columns=numeric_cols, index=df.index)
            
        return features
    
    def enhanced_rules(self, df):
        """
        Apply enhanced y3 rules for fake review detection
        
        Expands on original y3 rules with:
        - Time-based pattern detection (rapid-fire reviews)
        - Review velocity analysis (reviews per day)
        - Engagement ratio analysis (interaction rate vs review count)
        """
        fake_flags = pd.Series(False, index=df.index)
        
        # Original y3 rules
        rule1 = (
            (df['playtime_forever'] < 10) &
            (df['weighted_vote_score'] > 0.95) &
            (df['votes_up'] == 0)
        )
        
        rule2 = (
            (df['num_games_owned'] > 500) &
            (df['playtime_forever'] < 30)
        )
        
        rule3 = (
            (df['num_reviews'] > 50) &
            (df['votes_up'] < 1)
        )
        
        rule4 = (
            (df['weighted_vote_score'] > 0.97) &
            (df['votes_up'] == 0) &
            (df['comment_count'] == 0) &
            (df['playtime_forever'] < 15)
        )
        
        rule5 = (
            (df['num_reviews'] > np.percentile(df['num_reviews'], 98)) &
            (df['votes_up'] == 0)
        )
        
        rule6 = (
            (df['comment_count'] > 10) &
            (df['votes_funny'] > 100) &
            (df['weighted_vote_score'] < 0.1)
        )
        
        # Enhanced rules
        
        # Rule 7: Time-based patterns (rapid-fire reviews)
        if 'timestamp_created' in df.columns:
            df_sorted = df.sort_values(['author_id', 'timestamp_created']) if 'author_id' in df.columns else df.sort_values('timestamp_created')
            time_diff = pd.to_datetime(df_sorted['timestamp_created']).diff()
            
            rule7 = (time_diff < timedelta(minutes=5)) & (time_diff > timedelta(0))
            fake_flags |= rule7
            logger.info(f"Time-based rule flagged {rule7.sum()} reviews")
        
        # Rule 8: Review velocity check
        if 'author_id' in df.columns and len(df) > 1000:
            # Calculate reviews per day for each author
            author_stats = df.groupby('author_id').agg({
                'num_reviews': 'first',
                'timestamp_created': ['min', 'max', 'count']
            })
            
            author_stats.columns = ['num_reviews', 'first_review', 'last_review', 'review_count']
            author_stats['days_active'] = (
                pd.to_datetime(author_stats['last_review']) - 
                pd.to_datetime(author_stats['first_review'])
            ).dt.days.clip(lower=1)
            
            author_stats['reviews_per_day'] = author_stats['review_count'] / author_stats['days_active']
            
            # Flag authors with > 5 reviews per day and > 20 total reviews
            suspicious_authors = author_stats[
                (author_stats['num_reviews'] > 20) & 
                (author_stats['reviews_per_day'] > 5)
            ].index
            
            rule8 = df['author_id'].isin(suspicious_authors)
            fake_flags |= rule8
            logger.info(f"Velocity rule flagged {rule8.sum()} reviews")
        
        # Rule 9: Engagement ratio analysis
        if len(df) > 1000:
            # Calculate engagement metrics
            df['total_engagement'] = df['votes_up'] + df['votes_funny'] + df['comment_count']
            df['engagement_ratio'] = df['total_engagement'] / df['num_reviews'].clip(lower=1)
            
            rule9 = (
                (df['num_reviews'] > 30) &
                (df['engagement_ratio'] < 0.1)
            )
            fake_flags |= rule9
            logger.info(f"Engagement ratio rule flagged {rule9.sum()} reviews")
            
            # Clean up temporary columns
            df.drop(['total_engagement', 'engagement_ratio'], axis=1, inplace=True)
        
        # Combine all original y3 rules
        fake_flags |= (rule1 | rule2 | rule3 | rule4 | rule5 | rule6)
        
        fake_rate = fake_flags.mean()
        logger.info(f"Enhanced y3 fake rate: {fake_rate:.2%}")
        
        return fake_flags
    
    def filter_reviews(self, df, use_knn=True, k=7):
        """
        Filter out fake reviews using enhanced y3 rules and optionally KNN
        
        Args:
            df: DataFrame with review data
            use_knn: Whether to use KNN model for additional filtering
            k: Number of neighbors for KNN
            
        Returns:
            filtered_df: DataFrame with fake reviews removed
            fake_reviews_df: DataFrame containing only fake reviews
        """
        logger.info(f"Starting credibility filtering on {len(df)} reviews")
        
        # Apply rule-based detection
        fake_flags = self.enhanced_rules(df.copy())
        
        if use_knn and len(df) > 1000:
            # Prepare features for KNN
            features = self.prepare_features(df)
            
            # Train KNN on rule-based labels
            self.knn_model = KNeighborsClassifier(n_neighbors=k)
            self.knn_model.fit(features, fake_flags)
            
            # Get KNN predictions
            knn_fake_flags = self.knn_model.predict(features)
            
            # Combine rule-based and KNN predictions (OR operation)
            combined_flags = fake_flags | knn_fake_flags
            
            logger.info(f"KNN enhanced fake rate: {combined_flags.mean():.2%}")
            fake_flags = combined_flags
        
        # Separate fake and real reviews
        fake_reviews_df = df[fake_flags].copy()
        filtered_df = df[~fake_flags].copy()
        
        logger.info(f"Filtered out {len(fake_reviews_df)} fake reviews ({len(fake_reviews_df)/len(df)*100:.2f}%)")
        logger.info(f"Remaining credible reviews: {len(filtered_df)}")
        
        return filtered_df, fake_reviews_df
    
    def get_fake_review_summary(self, fake_reviews_df):
        """Generate summary statistics for fake reviews"""
        if len(fake_reviews_df) == 0:
            return "No fake reviews detected"
        
        summary = {
            'total_fake_reviews': len(fake_reviews_df),
            'avg_playtime': fake_reviews_df['playtime_forever'].mean(),
            'avg_num_games': fake_reviews_df['num_games_owned'].mean(),
            'avg_num_reviews': fake_reviews_df['num_reviews'].mean(),
            'avg_votes_up': fake_reviews_df['votes_up'].mean(),
            'most_common_playtime': fake_reviews_df['playtime_forever'].mode().iloc[0] if len(fake_reviews_df['playtime_forever'].mode()) > 0 else 0
        }
        
        return summary