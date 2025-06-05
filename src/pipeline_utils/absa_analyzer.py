"""
ABSA (Aspect-Based Sentiment Analysis) Module
Combines both Faster ABSA and Guided NMF approaches
"""

import pandas as pd
import numpy as np
import re
import logging
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import pipeline

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

logger = logging.getLogger(__name__)


class ABSAAnalyzer:
    """
    Combines keyword-based and NMF-based aspect sentiment analysis
    """
    
    def __init__(self, viz_dir: Optional[Path] = None):
        self.viz_dir = viz_dir or Path("outputs/visualizations/absa")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Data-driven aspects discovered using guided NMF with universal gaming seed words
        # 
        # METHODOLOGY REASONING:
        # - Pure unsupervised NMF gives unlabeled topics (Topic 0, Topic 1, etc.)
        # - Guided NMF with universal seeds ensures consistent, interpretable aspects across all games
        # - Seeds guide the algorithm but actual keywords are discovered from each game's review data
        # - This balances consistency (same aspects for all games) with data-driven accuracy
        #
        # FUTURE DEVELOPMENT: 
        # - Could extend with genre-specific seed words (RPG vs FPS vs Strategy)
        # - Could make seed words configurable per analysis
        # - Could implement hybrid approach with both guided and pure unsupervised topics
        
        self.aspect_keywords = None  # Will be populated by guided NMF discovery
        
        # Universal gaming seed words - work across all game genres
        # These guide NMF but actual keywords are discovered from the data
        self.aspect_seed_words = {
            'gameplay': [
                'gameplay', 'mechanics', 'controls', 'playing', 'fun', 'combat', 'difficulty',
                'challenge', 'progression', 'flow', 'satisfying', 'engaging', 'addictive'
            ],
            'graphics': [
                'graphics', 'visuals', 'art', 'beautiful', 'stunning', 'style', 'design',
                'textures', 'animations', 'lighting', 'detailed', 'aesthetic'
            ],
            'performance': [
                'performance', 'fps', 'lag', 'smooth', 'optimization', 'crashes', 'bugs',
                'stable', 'loading', 'technical', 'issues'
            ],
            'content': [
                'story', 'content', 'length', 'narrative', 'replay', 'variety', 'depth',
                'characters', 'plot', 'ending', 'quests'
            ],
            'audio': [
                'sound', 'music', 'audio', 'voice', 'soundtrack', 'effects', 'immersive'
            ],
            'value': [
                'price', 'worth', 'value', 'money', 'expensive', 'cheap', 'cost'
            ]
        }
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_model = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self.sentiment_model = None
    
    def discover_aspects_nmf(self, reviews: List[str], n_top_terms: int = 5) -> Dict[str, List[str]]:
        """Discover aspect keywords using guided NMF"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import NMF
            
            # TF-IDF Vectorization
            logger.info("Vectorizing reviews for aspect discovery...")
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(reviews)
            feature_names = vectorizer.get_feature_names_out()
            
            # Create guided matrix
            n_aspects = len(self.aspect_seed_words)
            H_init = np.zeros((n_aspects, len(feature_names)))
            vocab = {word: i for i, word in enumerate(feature_names)}
            
            seed_strength = 10.0
            
            for i, (aspect, seeds) in enumerate(self.aspect_seed_words.items()):
                for seed_word in seeds:
                    if seed_word in vocab:
                        H_init[i, vocab[seed_word]] = seed_strength
            
            # Run guided NMF
            logger.info(f"Running guided NMF with {n_aspects} aspects...")
            nmf_model = NMF(
                n_components=n_aspects,
                init='custom',
                solver='mu',
                beta_loss='frobenius',
                max_iter=1000,
                random_state=42,
                alpha_W=0.0,
                alpha_H=0.0
            )
            
            W_init = np.random.rand(tfidf_matrix.shape[0], n_aspects) * 0.1
            nmf_model.fit_transform(tfidf_matrix, W=W_init, H=H_init)
            
            # Extract discovered keywords
            discovered_keywords = {}
            aspect_names = list(self.aspect_seed_words.keys())
            
            for i, component in enumerate(nmf_model.components_):
                aspect_name = aspect_names[i]
                top_indices = component.argsort()[-n_top_terms:][::-1]
                discovered_keywords[aspect_name] = [feature_names[idx] for idx in top_indices]
                logger.info(f"Discovered keywords for {aspect_name}: {discovered_keywords[aspect_name]}")
            
            return discovered_keywords
            
        except ImportError:
            logger.error("sklearn not available for NMF - falling back to seed words")
            # Fallback to first 5 seed words for each aspect
            return {aspect: seeds[:5] for aspect, seeds in self.aspect_seed_words.items()}
        except Exception as e:
            logger.error(f"NMF discovery failed: {e} - falling back to seed words")
            return {aspect: seeds[:5] for aspect, seeds in self.aspect_seed_words.items()}

    def extract_aspects_keywords(self, text: str) -> List[str]:
        """Extract aspects using discovered keyword matching"""
        if self.aspect_keywords is None:
            logger.warning("Aspect keywords not discovered yet - using seed words")
            self.aspect_keywords = {aspect: seeds[:5] for aspect, seeds in self.aspect_seed_words.items()}
        
        found_aspects = set()
        text_lower = text.lower()
        
        for aspect, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if re.search(rf"\b{keyword}\b", text_lower):
                    found_aspects.add(aspect)
                    break
        
        return list(found_aspects)
    
    def analyze_sentiment_for_text(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment for a piece of text"""
        if not self.sentiment_model:
            return "NEUTRAL", 0.5
        
        try:
            # Truncate to model max length
            result = self.sentiment_model(text[:512])[0]
            label = result['label']
            score = result['score']
            return label, score
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return "NEUTRAL", 0.5
    
    def run_keyword_absa(self, df: pd.DataFrame) -> Dict:
        """Run data-driven ABSA analysis with guided NMF aspect discovery"""
        logger.info("Starting data-driven ABSA analysis")
        
        # First, discover aspects from the data if not already done
        if self.aspect_keywords is None:
            logger.info("Discovering aspects using guided NMF...")
            reviews_list = df['review'].tolist()
            self.aspect_keywords = self.discover_aspects_nmf(reviews_list)
        
        aspect_sentiments = defaultdict(list)
        aspect_confidence = defaultdict(list)
        aspect_counts = Counter()
        aspect_examples = defaultdict(list)
        all_confidence_scores = []  # Track all confidence scores for visualization
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing reviews (Keyword ABSA)"):
            review = row['review']
            aspects = self.extract_aspects_keywords(review)
            
            for aspect in aspects:
                aspect_counts[aspect] += 1
            
            if not aspects:
                continue
            
            # Analyze sentiment
            label, score = self.analyze_sentiment_for_text(review)
            
            # Store ALL confidence scores for visualization (before filtering)
            if aspects:  # Only if the review has aspects
                all_confidence_scores.append(score)
            
            # Skip low confidence predictions for final results
            if score < 0.6:
                continue
            
            for aspect in aspects:
                aspect_sentiments[aspect].append(label)
                aspect_confidence[aspect].append(score)
                
                # Store examples for each aspect
                if len(aspect_examples[aspect]) < 5:  # Keep top 5 examples
                    aspect_examples[aspect].append({
                        'review': review[:200] + '...' if len(review) > 200 else review,
                        'sentiment': label,
                        'confidence': score
                    })
        
        # Aggregate results
        summary = {}
        for aspect in aspect_sentiments:
            sentiments = aspect_sentiments[aspect]
            confidences = aspect_confidence[aspect]
            pos = sentiments.count("POSITIVE")
            neg = sentiments.count("NEGATIVE")
            total = pos + neg
            
            if total == 0:
                continue
            
            avg_conf = round(np.mean(confidences), 3)
            summary[aspect] = {
                "mentions": aspect_counts[aspect],
                "positive": pos,
                "negative": neg,
                "total_analyzed": total,
                "percent_positive": round(pos / total * 100, 2),
                "avg_confidence": avg_conf,
                "examples": aspect_examples[aspect]
            }
        
        logger.info(f"Keyword ABSA complete. Found {len(summary)} aspects")
        
        # Return both the summary and all confidence scores for visualization
        return {
            'summary': summary,
            'all_confidence_scores': all_confidence_scores
        }
    
    def run_nmf_absa(self, df: pd.DataFrame, n_topics: int = 10) -> Dict:
        """Run NMF-based topic modeling for aspect discovery"""
        logger.info("Starting NMF-based ABSA analysis")
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import NMF
            
            # Prepare text data
            reviews = df['review'].tolist()
            
            # TF-IDF Vectorization
            logger.info("Vectorizing reviews...")
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english',
                max_df=0.8,
                min_df=5
            )
            
            tfidf_matrix = vectorizer.fit_transform(reviews)
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply NMF
            logger.info(f"Running NMF with {n_topics} topics...")
            nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=300)
            nmf_features = nmf_model.fit_transform(tfidf_matrix)
            
            # Extract topics
            topics = {}
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_indices = np.argsort(topic)[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics[f"topic_{topic_idx}"] = {
                    "words": top_words,
                    "weights": [topic[i] for i in top_indices]
                }
            
            # Assign reviews to dominant topics
            topic_assignments = np.argmax(nmf_features, axis=1)
            topic_sentiments = defaultdict(list)
            
            logger.info("Analyzing sentiment for discovered topics...")
            for idx, topic_idx in enumerate(topic_assignments):
                if nmf_features[idx, topic_idx] > 0.1:  # Threshold for topic relevance
                    review = reviews[idx]
                    label, score = self.analyze_sentiment_for_text(review)
                    
                    if score > 0.6:
                        topic_key = f"topic_{topic_idx}"
                        topic_sentiments[topic_key].append({
                            'sentiment': label,
                            'confidence': score,
                            'relevance': nmf_features[idx, topic_idx]
                        })
            
            # Aggregate topic results
            nmf_summary = {}
            for topic_key, sentiments in topic_sentiments.items():
                if not sentiments:
                    continue
                
                pos = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
                neg = sum(1 for s in sentiments if s['sentiment'] == 'NEGATIVE')
                total = pos + neg
                
                if total > 0:
                    nmf_summary[topic_key] = {
                        "words": topics[topic_key]["words"][:5],
                        "mentions": total,
                        "positive": pos,
                        "negative": neg,
                        "percent_positive": round(pos / total * 100, 2),
                        "avg_relevance": round(np.mean([s['relevance'] for s in sentiments]), 3)
                    }
            
            logger.info(f"NMF ABSA complete. Found {len(nmf_summary)} topics")
            return nmf_summary
            
        except ImportError:
            logger.error("sklearn not available for NMF analysis")
            return {}
        except Exception as e:
            logger.error(f"NMF analysis failed: {e}")
            return {}
    
    def combine_results(self, keyword_results: Dict, nmf_results: Dict) -> Dict:
        """Combine keyword and NMF results"""
        combined = {
            "keyword_aspects": keyword_results,
            "discovered_topics": nmf_results,
            "summary": {
                "total_keyword_aspects": len(keyword_results),
                "total_discovered_topics": len(nmf_results),
                "most_mentioned_aspect": max(keyword_results.items(), 
                                           key=lambda x: x[1]['mentions'])[0] if keyword_results else None,
                "most_positive_aspect": max(keyword_results.items(), 
                                          key=lambda x: x[1]['percent_positive'])[0] if keyword_results else None,
                "most_negative_aspect": min(keyword_results.items(), 
                                          key=lambda x: x[1]['percent_positive'])[0] if keyword_results else None
            }
        }
        
        return combined
    
    def visualize_results(self, results: Dict, game_name: str = "Game"):
        """Create multiple visualizations for ABSA results"""
        keyword_results = results.get('keyword_aspects', {})
        nmf_results = results.get('discovered_topics', {})
        
        if not keyword_results and not nmf_results:
            logger.warning("No aspects to visualize")
            return
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Keyword Aspects Stacked Bar Chart
        if keyword_results:
            self._create_stacked_bar_chart(keyword_results, game_name, timestamp)
        
        # 2. Sentiment Distribution Pie Chart
        if keyword_results:
            self._create_sentiment_pie_chart(keyword_results, game_name, timestamp)
            
        # 3. Aspect Confidence Heatmap
        if keyword_results:
            self._create_confidence_heatmap(keyword_results, game_name, timestamp)
            
        # 4. ABSA Confidence Distribution
        if keyword_results:
            self._create_absa_confidence_distribution(results, game_name, timestamp)
            
        # 5. NMF Topics Visualization
        if nmf_results:
            self._create_nmf_topics_chart(nmf_results, game_name, timestamp)
            
        # 6. Create detailed report
        self._create_detailed_report(results, game_name)
    
    def _create_stacked_bar_chart(self, keyword_results: Dict, game_name: str, timestamp: str):
        """Create stacked bar chart for keyword aspects"""
        # Prepare data for visualization
        aspects = list(keyword_results.keys())
        positive_counts = [keyword_results[a]['positive'] for a in aspects]
        negative_counts = [keyword_results[a]['negative'] for a in aspects]
        
        # Sort by total mentions
        total_mentions = [p + n for p, n in zip(positive_counts, negative_counts)]
        sorted_indices = sorted(range(len(total_mentions)), key=lambda i: total_mentions[i], reverse=True)
        
        aspects = [aspects[i] for i in sorted_indices]
        positive_counts = [positive_counts[i] for i in sorted_indices]
        negative_counts = [negative_counts[i] for i in sorted_indices]
        total_mentions = [total_mentions[i] for i in sorted_indices]
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(aspects))
        width = 0.6
        
        # Check if we need log scale for large value ranges
        max_val = max(total_mentions) if total_mentions else 1
        min_val = min([t for t in total_mentions if t > 0]) if total_mentions else 1
        use_log_scale = max_val / min_val > 100
        
        p1 = ax.bar(x, positive_counts, width, label='Positive', color='#2ecc71')
        p2 = ax.bar(x, negative_counts, width, bottom=positive_counts, label='Negative', color='#e74c3c')
        
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Number of Reviews (log scale)', fontsize=12)
        else:
            ax.set_ylabel('Number of Reviews', fontsize=12)
        ax.set_title(f'Aspect Sentiment Analysis - {game_name}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(aspects, rotation=45, ha='right')
        ax.legend()
        
        # Add percentage labels and total counts
        for i, (pos, neg, total) in enumerate(zip(positive_counts, negative_counts, total_mentions)):
            if total > 0:
                pos_pct = pos / total * 100
                # Adjust label position based on scale
                if use_log_scale:
                    y_pos = total * 1.15  # 15% above bar for log scale
                else:
                    y_pos = total + max(total_mentions) * 0.02
                
                # Show both percentage and count for clarity
                label_text = f'{pos_pct:.0f}%\n({total})'
                ax.text(i, y_pos, label_text, 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization with clean name
        viz_file = self.viz_dir / "aspect_sentiment_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved stacked bar chart: {viz_file}")
    
    def _create_sentiment_pie_chart(self, keyword_results: Dict, game_name: str, timestamp: str):
        """Create pie chart showing overall sentiment distribution"""
        total_positive = sum(data['positive'] for data in keyword_results.values())
        total_negative = sum(data['negative'] for data in keyword_results.values())
        
        if total_positive + total_negative == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sizes = [total_positive, total_negative]
        labels = ['Positive', 'Negative']
        colors = ['#2ecc71', '#e74c3c']
        
        # Calculate percentages for better display
        total = total_positive + total_negative
        pos_pct = total_positive / total * 100
        neg_pct = total_negative / total * 100
        
        # Custom autopct function to show counts and percentages
        def autopct_format(pct):
            absolute = int(pct / 100. * total)
            return f'{pct:.1f}%\n({absolute:,})'
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct=autopct_format,
                                         startangle=90, textprops={'fontsize': 11})
        
        # Make text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add summary in title
        ax.set_title(f'Overall Sentiment Distribution - {game_name}\n'
                    f'Total Reviews Analyzed: {total:,}', 
                    fontsize=16, fontweight='bold')
        
        # Save visualization with clean name
        viz_file = self.viz_dir / "overall_sentiment_distribution.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sentiment pie chart: {viz_file}")
    
    def _create_confidence_heatmap(self, keyword_results: Dict, game_name: str, timestamp: str):
        """Create heatmap showing confidence and sentiment for each aspect"""
        aspects = list(keyword_results.keys())
        
        if len(aspects) < 2:
            return  # Skip heatmap for too few aspects
        
        # Prepare data
        sentiment_scores = []
        confidence_scores = []
        mention_counts = []
        
        for aspect in aspects:
            data = keyword_results[aspect]
            sentiment_scores.append(data['percent_positive'])
            confidence_scores.append(data['avg_confidence'] * 100)  # Convert to percentage
            mention_counts.append(data['mentions'])
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Create separate colormaps for each row
        # Row 0 (Positive %): Use diverging colormap centered at 50%
        sentiment_data = np.array([sentiment_scores])
        im1 = ax.imshow(sentiment_data, cmap='RdYlGn', aspect='auto', 
                       vmin=0, vmax=100, extent=[-0.5, len(aspects)-0.5, 0.5, -0.5])
        
        # Row 1 (Confidence %): Use grayscale
        conf_data = np.array([confidence_scores])
        # Normalize to 0-1 for grayscale
        conf_norm = conf_data / 100.0
        # Create gray color array
        gray_colors = plt.cm.gray(np.ones_like(conf_norm) * 0.9)  # Light gray
        
        # Row 2 (Mentions): Use grayscale
        mention_data = np.array([mention_counts])
        mention_norm = mention_data / np.max(mention_data)
        
        # Plot each row separately
        for j in range(len(aspects)):
            # Positive % - colored
            rect1 = plt.Rectangle((j-0.5, -0.5), 1, 1, 
                                 facecolor=plt.cm.RdYlGn(sentiment_scores[j]/100))
            ax.add_patch(rect1)
            
            # Confidence % - gray
            rect2 = plt.Rectangle((j-0.5, 0.5), 1, 1, 
                                 facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect2)
            
            # Mentions - gray
            rect3 = plt.Rectangle((j-0.5, 1.5), 1, 1, 
                                 facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect3)
        
        # Set ticks and labels
        ax.set_xlim(-0.5, len(aspects)-0.5)
        ax.set_ylim(2.5, -0.5)
        ax.set_xticks(np.arange(len(aspects)))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(aspects, rotation=45, ha='right')
        ax.set_yticklabels(['Positive %', 'Confidence %', 'Mentions'])
        
        # Add text annotations
        for i in range(3):
            for j in range(len(aspects)):
                if i == 0:  # Positive %
                    value = sentiment_scores[j]
                    # Use white text for extreme values, black for middle values
                    text_color = 'white' if value < 30 or value > 70 else 'black'
                elif i == 1:  # Confidence %
                    value = confidence_scores[j]
                    text_color = 'black'
                else:  # Mentions
                    value = mention_counts[j]
                    text_color = 'black'
                
                ax.text(j, i, f'{value:.0f}',
                       ha="center", va="center", color=text_color, fontweight='bold')
        
        ax.set_title(f'Aspect Analysis - {game_name}', fontsize=16, fontweight='bold')
        
        # Add a colorbar only for the sentiment row
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
        cbar.set_label('Positive Sentiment %', fontsize=12)
        
        plt.tight_layout()
        
        # Save visualization with clean name
        viz_file = self.viz_dir / "aspect_analysis_heatmap.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confidence heatmap: {viz_file}")
    
    def _create_absa_confidence_distribution(self, results: Dict, game_name: str, timestamp: str):
        """Create confidence distribution histogram for ABSA sentiment analysis"""
        # Get all confidence scores from the analysis
        all_confidences = results.get('all_confidence_scores', [])
        
        if not all_confidences:
            logger.warning("No confidence scores found for ABSA visualization")
            return
            
        all_confidences = np.array(all_confidences)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        ax.hist(all_confidences, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add threshold line at 0.6
        ax.axvline(0.6, color='red', linestyle='dashed', linewidth=2,
                   label='Confidence Threshold (60%)')
        
        # Add statistics
        mean_conf = np.mean(all_confidences)
        median_conf = np.median(all_confidences)
        
        ax.axvline(mean_conf, color='blue', linestyle='dashed', linewidth=2,
                   label=f'Mean: {mean_conf:.3f}')
        ax.axvline(median_conf, color='green', linestyle='dashed', linewidth=2,
                   label=f'Median: {median_conf:.3f}')
        
        # Count excluded predictions
        excluded_count = np.sum(all_confidences < 0.6)
        included_count = np.sum(all_confidences >= 0.6)
        total_count = len(all_confidences)
        
        ax.set_xlabel('Sentiment Confidence Score', fontsize=12)
        ax.set_ylabel('Number of Predictions', fontsize=12)
        ax.set_title(f'ABSA Sentiment Confidence Distribution - {game_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        # Add text box with statistics
        stats_text = f'Total predictions: {total_count:,}\nIncluded (≥60%): {included_count:,}\nExcluded (<60%): {excluded_count:,}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization with clean name
        viz_file = self.viz_dir / "absa_confidence_distribution.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ABSA confidence distribution: {viz_file}")
    
    def _create_nmf_topics_chart(self, nmf_results: Dict, game_name: str, timestamp: str):
        """Create visualization for NMF discovered topics"""
        if not nmf_results:
            return
        
        topics = list(nmf_results.keys())
        positive_counts = [nmf_results[t]['positive'] for t in topics]
        negative_counts = [nmf_results[t]['negative'] for t in topics]
        
        # Sort by total mentions
        total_mentions = [p + n for p, n in zip(positive_counts, negative_counts)]
        sorted_indices = sorted(range(len(total_mentions)), key=lambda i: total_mentions[i], reverse=True)
        
        topics = [topics[i] for i in sorted_indices]
        positive_counts = [positive_counts[i] for i in sorted_indices]
        negative_counts = [negative_counts[i] for i in sorted_indices]
        
        # Create horizontal bar chart for better topic name visibility
        fig, ax = plt.subplots(figsize=(12, max(6, len(topics) * 0.8)))
        
        y = np.arange(len(topics))
        
        p1 = ax.barh(y, positive_counts, label='Positive', color='#2ecc71')
        p2 = ax.barh(y, negative_counts, left=positive_counts, label='Negative', color='#e74c3c')
        
        ax.set_xlabel('Number of Reviews', fontsize=12)
        ax.set_title(f'Discovered Topics (NMF) - {game_name}', fontsize=16, fontweight='bold')
        ax.set_yticks(y)
        
        # Create topic labels with keywords
        topic_labels = []
        for topic in topics:
            words = nmf_results[topic]['words'][:3]  # Top 3 words
            label = f"{topic}\n({', '.join(words)})"
            topic_labels.append(label)
        
        ax.set_yticklabels(topic_labels)
        ax.legend()
        ax.invert_yaxis()  # Show highest mentions at top
        
        plt.tight_layout()
        
        # Save visualization with clean name
        viz_file = self.viz_dir / "discovered_topics_nmf.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved NMF topics chart: {viz_file}")
    
    def _create_detailed_report(self, results: Dict, game_name: str):
        """Create a detailed text report of ABSA findings"""
        report_file = self.viz_dir / "analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"ASPECT-BASED SENTIMENT ANALYSIS REPORT\n")
            f.write(f"Game: {game_name}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Keyword-based results
            f.write("KEYWORD-BASED ASPECT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            keyword_results = results.get('keyword_aspects', {})
            for aspect, data in sorted(keyword_results.items(), key=lambda x: x[1]['mentions'], reverse=True):
                f.write(f"\n{aspect.upper()}\n")
                f.write(f"  Total mentions: {data['mentions']}\n")
                f.write(f"  Analyzed: {data['total_analyzed']}\n")
                f.write(f"  Positive: {data['positive']} ({data['percent_positive']:.1f}%)\n")
                f.write(f"  Negative: {data['negative']} ({100-data['percent_positive']:.1f}%)\n")
                f.write(f"  Avg confidence: {data['avg_confidence']:.3f}\n")
                
                # Add examples if available
                if 'examples' in data and data['examples']:
                    f.write("  Example reviews:\n")
                    for ex in data['examples'][:3]:
                        f.write(f"    - [{ex['sentiment']}] {ex['review']}\n")
            
            # NMF-based results
            f.write("\n\nDISCOVERED TOPICS (NMF)\n")
            f.write("-" * 40 + "\n")
            
            nmf_results = results.get('discovered_topics', {})
            for topic, data in sorted(nmf_results.items(), key=lambda x: x[1]['mentions'], reverse=True):
                f.write(f"\n{topic.upper()}\n")
                f.write(f"  Keywords: {', '.join(data['words'])}\n")
                f.write(f"  Mentions: {data['mentions']}\n")
                f.write(f"  Positive: {data['positive']} ({data['percent_positive']:.1f}%)\n")
                f.write(f"  Negative: {data['negative']} ({100-data['percent_positive']:.1f}%)\n")
            
            # Summary
            f.write("\n\nSUMMARY\n")
            f.write("-" * 40 + "\n")
            summary = results.get('summary', {})
            f.write(f"Total keyword aspects found: {summary.get('total_keyword_aspects', 0)}\n")
            f.write(f"Total topics discovered: {summary.get('total_discovered_topics', 0)}\n")
            f.write(f"Most mentioned aspect: {summary.get('most_mentioned_aspect', 'N/A')}\n")
            f.write(f"Most positive aspect: {summary.get('most_positive_aspect', 'N/A')}\n")
            f.write(f"Most negative aspect: {summary.get('most_negative_aspect', 'N/A')}\n")
        
        logger.info(f"Saved ABSA report to {report_file}")
    
    def analyze(self, df: pd.DataFrame, game_name: str = "Game", 
               run_nmf: bool = True, n_topics: int = 10) -> Dict:
        """
        Run complete ABSA analysis
        
        Args:
            df: DataFrame with review data
            game_name: Name of the game for reporting
            run_nmf: Whether to run NMF analysis
            n_topics: Number of topics for NMF
            
        Returns:
            Combined results dictionary
        """
        logger.info(f"Starting ABSA analysis for {game_name}")
        
        # Run keyword-based analysis
        keyword_data = self.run_keyword_absa(df)
        keyword_results = keyword_data['summary']
        
        # Run NMF-based analysis if requested
        nmf_results = {}
        if run_nmf:
            nmf_results = self.run_nmf_absa(df, n_topics)
        
        # Combine results
        combined_results = self.combine_results(keyword_results, nmf_results)
        
        # Add confidence scores for visualization
        combined_results['all_confidence_scores'] = keyword_data['all_confidence_scores']
        
        # Create visualizations
        self.visualize_results(combined_results, game_name)
        
        logger.info("ABSA analysis complete")
        return combined_results