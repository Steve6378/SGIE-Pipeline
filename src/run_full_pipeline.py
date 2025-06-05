"""
Full Pipeline Runner - Includes ABSA and Emotion Analysis
Extends main.py to run the complete analysis pipeline
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

from main import SteamGamesPipeline
from pipeline_utils import ABSAAnalyzer, EmotionAnalyzer

logger = logging.getLogger(__name__)


class FullSteamPipeline(SteamGamesPipeline):
    """Extended pipeline with ABSA and Emotion analysis"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__(base_dir)
        
        # Initialize additional analyzers
        self.absa_analyzer = ABSAAnalyzer(viz_dir=self.viz_dir / "absa")
        self.emotion_analyzer = EmotionAnalyzer(viz_dir=self.viz_dir / "emotions")
        
    def run_absa_analysis(self, df: pd.DataFrame, game_name: str) -> dict:
        """Run ABSA analysis on reviews"""
        logger.info(f"Starting ABSA analysis for {game_name}")
        
        try:
            # Run ABSA (both keyword and NMF)
            results = self.absa_analyzer.analyze(
                df, 
                game_name=game_name,
                run_nmf=True,
                n_topics=10
            )
            
            # Log summary
            summary = results.get('summary', {})
            logger.info(f"ABSA complete: {summary.get('total_keyword_aspects', 0)} aspects found")
            logger.info(f"Most mentioned: {summary.get('most_mentioned_aspect', 'N/A')}")
            logger.info(f"Most positive: {summary.get('most_positive_aspect', 'N/A')}")
            logger.info(f"Most negative: {summary.get('most_negative_aspect', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"ABSA analysis failed: {e}")
            return {}
    
    def run_emotion_analysis(self, df: pd.DataFrame, game_name: str, 
                           sample_size: Optional[int] = None) -> dict:
        """Run emotion analysis on reviews"""
        logger.info(f"Starting emotion analysis for {game_name}")
        
        try:
            # Check if models exist
            model_status = self.emotion_analyzer.check_models_exist()
            if not all(model_status.values()):
                logger.error("Emotion models not found. Please run setup script first.")
                return {}
            
            # Run analysis
            results = self.emotion_analyzer.analyze_reviews(
                df,
                text_column='review',
                sample_size=sample_size
            )
            
            # Create visualizations
            if results:
                self.emotion_analyzer.visualize_results(results, game_name)
                
                # Log summary
                logger.info(f"Emotion analysis complete: {results.get('total_analyzed', 0)} reviews")
                logger.info(f"Overall confidence: {results.get('overall_avg_confidence', 0):.3f}")
                
                # Log emotion distribution
                for emotion, pct in results.get('emotion_percentages', {}).items():
                    logger.info(f"  {emotion}: {pct:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {}
    
    def run_complete_pipeline(self, user_input: str, interactive: bool = True,
                            skip_scraping: bool = False, 
                            existing_data_path: Optional[str] = None,
                            emotion_sample_size: Optional[int] = None):
        """Run the complete analysis pipeline including ABSA and emotions"""
        
        # Run base pipeline (scraping, cleaning, credibility)
        pipeline_state = self.run_pipeline(
            user_input, 
            interactive, 
            skip_scraping, 
            existing_data_path
        )
        
        if not pipeline_state:
            return None
        
        # Get credible reviews
        credible_df = pipeline_state['data_files']['credible']
        game_name = pipeline_state['game_info']['name']
        game_id = pipeline_state['game_info']['id']
        
        print(f"\n{'='*60}")
        print("STARTING ADVANCED ANALYSIS")
        print(f"{'='*60}\n")
        
        # Run ABSA
        print("🔍 Running Aspect-Based Sentiment Analysis...")
        absa_results = self.run_absa_analysis(credible_df, game_name)
        
        # Run Emotion Analysis
        print("\n🎭 Running Emotion Analysis...")
        if emotion_sample_size:
            print(f"   Sampling {emotion_sample_size} reviews for emotion analysis")
        
        emotion_results = self.run_emotion_analysis(
            credible_df, 
            game_name,
            sample_size=emotion_sample_size
        )
        
        # Compile final results
        final_results = {
            **pipeline_state,
            'absa_results': absa_results,
            'emotion_results': {
                k: v for k, v in emotion_results.items() 
                if k != 'predictions_df'  # Exclude large DataFrame from JSON
            }
        }
        
        # Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.data_dir / "results" / f"{game_id}_complete_analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Saved complete results to {results_file}")
        
        # Print final summary
        self._print_analysis_summary(final_results)
        
        return final_results
    
    def _print_analysis_summary(self, results: dict):
        """Print a comprehensive summary of all analyses"""
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE - SUMMARY")
        print(f"{'='*60}\n")
        
        # Basic stats
        print(f"📊 Dataset Overview:")
        print(f"   Game: {results['game_info']['name']}")
        print(f"   Total reviews scraped: {results['raw_reviews']}")
        print(f"   Credible reviews: {results['credible_reviews']}")
        print(f"   Fake reviews filtered: {results['fake_reviews']}")
        
        # ABSA summary
        absa = results.get('absa_results', {})
        if absa:
            summary = absa.get('summary', {})
            print(f"\n🔍 Aspect Analysis:")
            print(f"   Aspects found: {summary.get('total_keyword_aspects', 0)}")
            print(f"   Most discussed: {summary.get('most_mentioned_aspect', 'N/A')}")
            print(f"   Most positive: {summary.get('most_positive_aspect', 'N/A')}")
            print(f"   Most negative: {summary.get('most_negative_aspect', 'N/A')}")
        
        # Emotion summary
        emotion = results.get('emotion_results', {})
        if emotion:
            print(f"\n🎭 Emotion Analysis:")
            print(f"   Reviews analyzed: {emotion.get('total_analyzed', 0)}")
            print(f"   Average confidence: {emotion.get('overall_avg_confidence', 0):.3f}")
            
            # Top 3 emotions
            emotions = emotion.get('emotion_percentages', {})
            if emotions:
                print(f"   Top emotions:")
                for emotion_name, pct in sorted(emotions.items(), 
                                               key=lambda x: x[1], reverse=True)[:3]:
                    print(f"     - {emotion_name}: {pct:.1f}%")
        
        print(f"\n✅ All visualizations saved to: {self.viz_dir}")
        print(f"📁 All data files saved to: {self.data_dir}")


def main():
    """Main entry point for full pipeline"""
    parser = argparse.ArgumentParser(
        description="Steam Games Full Analysis Pipeline - Complete game review analysis"
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
    parser.add_argument(
        "--emotion-sample",
        type=int,
        help="Sample size for emotion analysis (default: all reviews)"
    )
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = FullSteamPipeline()
    
    # Run complete pipeline
    result = pipeline.run_complete_pipeline(
        args.game,
        interactive=not args.non_interactive,
        skip_scraping=args.skip_scraping,
        existing_data_path=args.data_file,
        emotion_sample_size=args.emotion_sample
    )
    
    if result:
        print("\n🎉 Complete analysis finished successfully!")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()