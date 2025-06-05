# Steam Games Insight Engine

A comprehensive, self-contained pipeline for analyzing Steam game reviews through multiple AI techniques including credibility detection, aspect-based sentiment analysis (ABSA), and emotion classification.

## IMPORTANT DISCLAIMER: All ideas and implementation details of features are incorporated by members of the group; only the pipeline creation and debugging process was done by Steve with support of Claude Code/Opus 4.

## 🚀 Features

- **🔍 Game Lookup**: Search Steam games by name or ID
- **🕷️ Web Scraping**: Automated Steam review collection
- **🛡️ Credibility Detection**: Enhanced fake review filtering with KNN
- **📊 ABSA Analysis**: Dual approach (keyword + topic modeling)
- **🎭 Emotion Analysis**: Ensemble classification (GPT-2, RoBERTa, DeBERTa)
- **📈 Rich Visualizations**: Publication-ready charts and reports
- **🗂️ Clean Organization**: Game-specific output directories
- **♻️ Re-run Friendly**: Clean overwrite system

## 📁 Project Structure

```
Final Pipeline/
├── src/                              # All source code
│   ├── pipeline_utils/               # Analysis modules
│   │   ├── web_scraping.py          # Steam scraping (local)
│   │   ├── web_scraper.py           # Scraper wrapper
│   │   ├── credibility_filter.py    # Enhanced fake detection
│   │   ├── absa_analyzer.py         # Sentiment analysis
│   │   ├── emotion_analyzer.py      # Emotion classification
│   │   ├── simple_predict.py        # Model prediction
│   │   └── ...                      # Other utilities
│   ├── main.py                      # CLI pipeline
│   └── setup_models.py              # Model downloader
├── models/                          # Pre-trained models (local)
│   ├── gpt2/final_model.pt         # GPT-2 emotion model
│   ├── roberta/final_model.pt      # RoBERTa emotion model
│   └── deberta/final_model.pt      # DeBERTa emotion model
├── outputs/                         # Analysis results
│   └── game_[id]/                  # Game-specific directories
│       ├── data/                   # Raw and processed data
│       └── visualizations/         # Charts and reports
├── logs/                           # Documentation and logs
│   ├── DEVELOPMENT_LOG.md          # Technical implementation
│   └── COMMUNICATIONS_LOG.md       # Design decisions
└── Steam_Games_Analysis_Pipeline.ipynb  # Interactive notebook
```

## 🛠️ Setup

### Prerequisites

**Note**: This code can only work with `Python < 3.13` due to `sentencepiece` not being supported for the most recent Python versions yet.

```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn
pip install torch transformers sklearn
pip install requests beautifulsoup4 langdetect
pip install gensim cleantext tqdm
pip install gdown  # For model downloads
```

### Model Setup

The pipeline requires pre-trained emotion models (~4.1 GB total). Run this **one-time setup**:

#### Option 1: Automated Download (Recommended)

```python
# Run in notebook cell or Python script
from src.setup_models import ModelSetup

setup = ModelSetup()
setup.download_and_extract()
```

#### Option 2: Manual Setup in Notebook

The notebook includes a built-in model setup cell that handles everything automatically:

```python
# Cell 6 in the notebook handles model download
# Just run it if you get "models not found" errors
```

#### Option 3: Command Line

```bash
cd "Final Pipeline"
python src/setup_models.py
```

### Verification

Check that models are correctly installed:

```python
from src.pipeline_utils import EmotionAnalyzer

analyzer = EmotionAnalyzer()
status = analyzer.check_models_exist()
print(status)
# Should show: {'gpt2': True, 'roberta': True, 'deberta': True}
```

## 🎮 Quick Start

### Option 1: Interactive Notebook (Recommended)

1. Open `Steam_Games_Analysis_Pipeline.ipynb`
2. Run cells sequentially
3. Change `GAME_INPUT` in Cell 3 to your desired game
4. Results display inline with interactive visualizations

### Option 2: Command Line

```bash
cd "Final Pipeline"
python src/main.py --game "Black Myth Wukong" --interactive
```

### Option 3: Python Script

```python
from src.pipeline_utils import *
from pathlib import Path

# Game lookup
lookup = SteamGameLookup()
game_info = lookup.search_games("Black Myth Wukong")[0]

# Web scraping  
from src.pipeline_utils.web_scraper import WebScraperWrapper
scraper = WebScraperWrapper()
reviews_df = scraper.scrape_reviews(game_info['id'])

# Analysis pipeline
cleaner = DataCleaner()
cleaned_df = cleaner.clean_reviews(reviews_df)

cred_filter = CredibilityFilter()
credible_df, fake_df = cred_filter.filter_reviews(cleaned_df)

# Create output directory
game_outputs = Path(f"outputs/game_{game_info['id']}")
absa_viz = game_outputs / "visualizations/absa"
emotion_viz = game_outputs / "visualizations/emotions"

# Run analyses
absa = ABSAAnalyzer(viz_dir=absa_viz)
absa_results = absa.analyze(credible_df, game_info['name'])

emotion = EmotionAnalyzer(viz_dir=emotion_viz)
emotion_results = emotion.analyze_reviews(credible_df)
```

## 📊 Analysis Outputs

### Data Files (per game)
- `outputs/game_[id]/data/raw/scraped_reviews.csv` - Original scraped data
- `outputs/game_[id]/data/cleaned/credible_reviews.csv` - Filtered reviews
- `outputs/game_[id]/data/analysis_results.json` - Complete results summary

### Visualizations (per game)
- `outputs/game_[id]/visualizations/absa/` - Sentiment analysis charts
- `outputs/game_[id]/visualizations/emotions/` - Emotion classification charts

### Reports
- Detailed text reports with insights and statistics
- Example reviews for each aspect/emotion
- Model confidence and agreement analysis

## ⚙️ Configuration

Edit the notebook's `ANALYSIS_CONFIG` section:

```python
ANALYSIS_CONFIG = {
    'overwrite_previous': True,     # Clean vs timestamped files
    'keep_backup': True,           # Backup before overwrite
    'testing_mode': True           # Use subset for faster testing
}
```

## 📈 Analysis Pipeline

1. **Game Lookup**: Validate Steam game ID or search by name
2. **Web Scraping**: Collect reviews with metadata (playtime, votes, etc.)
3. **Data Cleaning**: Standardize formats, add calculated fields
4. **Credibility Filtering**: Remove fake/spam reviews using enhanced y3 rules + KNN
5. **ABSA Analysis**: 
   - Keyword-based aspect detection (graphics, gameplay, story, etc.)
   - NMF topic modeling for discovery
   - Sentiment classification per aspect
6. **Emotion Analysis**: Ensemble prediction using 3 fine-tuned models
7. **Visualization**: Generate publication-ready charts and reports

## 🔍 Minimum Requirements

- **Reviews**: 1000 minimum (2500+ recommended for robust results)
- **Disk Space**: 5+ GB (models + data + outputs)
- **Memory**: 8+ GB RAM recommended
- **GPU**: Optional but recommended for emotion analysis

## 🛠️ Advanced Usage

### Batch Processing Multiple Games

```python
games = ["Black Myth Wukong", "Elden Ring", "Cyberpunk 2077"]

for game_name in games:
    print(f"Analyzing {game_name}...")
    # Run full pipeline for each game
    # Results saved to separate game_[id] directories
```

### Custom Aspect Keywords

```python
# Modify aspect keywords for specific game types
absa = ABSAAnalyzer()
absa.aspect_keywords['monetization'] = ['microtransaction', 'dlc', 'season pass']
```

### Sampling for Large Datasets

```python
# For games with 50k+ reviews
emotion_results = emotion_analyzer.analyze_reviews(
    credible_df, 
    sample_size=5000  # Analyze subset for performance
)
```

## 🔬 Technical Details

### Credibility Detection
- **Enhanced y3 rules**: Optimized thresholds for 2.92% fake rate
- **Time-based patterns**: Rapid-fire review detection
- **Engagement analysis**: Prolific reviewers with low engagement
- **KNN second layer**: 7-neighbor similarity detection

### ABSA Methodology
- **Keyword approach**: 10 predefined aspects with expanded vocabularies
- **NMF topic modeling**: Automated aspect discovery
- **Sentiment classification**: DistilBERT fine-tuned model
- **Confidence thresholding**: Filters low-confidence predictions

### Emotion Ensemble
- **GPT-2**: Fine-tuned on emotion classification
- **RoBERTa**: Optimized for text understanding  
- **DeBERTa**: State-of-the-art transformer architecture
- **Batch Processing**: True GPU batching for 5-10x performance improvement
- **Ensemble voting**: Weighted combination with Jensen-Shannon Divergence disagreement analysis
- **Model Uncertainty**: Real-time disagreement metrics using information theory

## 📚 Documentation

- `logs/DEVELOPMENT_LOG.md` - Detailed technical implementation
- `logs/COMMUNICATIONS_LOG.md` - Design decisions and reasoning
- Inline code documentation and docstrings

## 🐛 Troubleshooting

### Common Issues

1. **"Models not found"**: Run the model setup cell/script
2. **Import errors**: Ensure you're in the correct directory
3. **Memory errors**: Reduce sample sizes or use CPU-only mode
4. **Network timeouts**: Check internet connection for scraping
5. **Empty results**: Game might have insufficient English reviews

### Performance Tips

- **GPU Optimization**: Pipeline now optimized for high-end GPUs (RTX 4090, etc.)
- **Batch Processing**: 5-10x faster emotion analysis with true GPU batching
- **Use `testing_mode=True`** for faster iteration during development
- **Batch Size**: Automatically optimized (32) for modern GPU memory
- **Sample large datasets** (>10k reviews) if memory constrained
- **Single-Pass Architecture**: Eliminated redundant prediction runs for 3x speedup

### Getting Help

1. Check the troubleshooting section
2. Review the development log for implementation details
3. Verify model installation and file structure
4. Check output logs for specific error messages

## 🎯 Self-Contained Design

This pipeline is completely self-contained:
- ✅ No external module dependencies
- ✅ All functionality included locally
- ✅ Can be copied and run anywhere
- ✅ No system path manipulation required
- ✅ Models stored within project structure

## 📄 License

This project builds upon multiple open-source components. Please check individual model licenses for commercial use restrictions.

---

## 🚀 Ready to Analyze?

1. **Install dependencies** (see Setup section)
2. **Download models** (one-time setup)
3. **Open the notebook** or run CLI commands
4. **Enter your game** name or ID
5. **Watch the magic happen!** ✨

The pipeline will automatically scrape, clean, analyze, and visualize insights from thousands of Steam reviews in minutes.