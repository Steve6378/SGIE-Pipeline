"""
Model Setup Script
Downloads and extracts emotion analysis models from Google Drive
"""

import os
import sys
import zipfile
import tempfile
from pathlib import Path
from tqdm import tqdm
import gdown

# Global configuration constants
ARCHIVE_URL = "https://drive.google.com/file/d/1JVKU6n-yGC_s1lA68o208kw9RqlMJhb8/view?usp=sharing"
ARCHIVE_FILE_ID = "1JVKU6n-yGC_s1lA68o208kw9RqlMJhb8"

class ModelSetup:
    """Handles downloading and setting up emotion analysis models"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd() / "models"
        self.checkpoint_dir = self.base_dir
        
    def check_existing_models(self):
        """Check which model files already exist"""
        models = ["gpt2", "roberta", "deberta"]
        status = {}
        
        for model in models:
            model_dir = self.checkpoint_dir / model
            status[model] = {
                "model_file": (model_dir / "final_model.pt").exists(),
                "metrics_file": (model_dir / "final_metrics.json").exists()
            }
        
        return status
    
    def print_status(self, status):
        """Print current model status"""
        print("📁 Current model status:")
        
        for model, files in status.items():
            print(f"\n{model.upper()}:")
            
            model_status = "✅ Found" if files["model_file"] else "❌ Missing"
            print(f"  final_model.pt: {model_status}")
            
            metrics_status = "✅ Found" if files["metrics_file"] else "❌ Missing"  
            print(f"  final_metrics.json: {metrics_status}")
    
    def all_models_exist(self, status):
        """Check if all required model files exist"""
        for model, files in status.items():
            if not files["model_file"]:
                return False
        return True
    
    def download_and_extract_models(self, force=False):
        """Download and extract the model archive"""
        
        # Check existing models
        status = self.check_existing_models()
        self.print_status(status)
        
        if self.all_models_exist(status) and not force:
            print("\n✅ All models already downloaded!")
            return True
        
        print(f"\n📥 Downloading model archive from Google Drive...")
        print(f"📦 Archive size: ~4.1 GB (this may take a while)")
        
        # Create temp directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            archive_path = temp_dir / "models.zip"
            
            try:
                # Download using gdown
                print("⬇️  Starting download...")
                gdown.download(id=ARCHIVE_FILE_ID, output=str(archive_path), quiet=False)
                
                if not archive_path.exists():
                    print("❌ Download failed!")
                    return False
                
                print(f"✅ Download complete! ({archive_path.stat().st_size / 1e9:.1f} GB)")
                
                # Extract archive
                print("📦 Extracting models...")
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # List contents
                    file_list = zip_ref.namelist()
                    model_files = [f for f in file_list if f.endswith(('.pt', '.json'))]
                    
                    print(f"📄 Found {len(model_files)} model files in archive")
                    
                    # Extract with progress bar
                    for file_info in tqdm(zip_ref.infolist(), desc="Extracting"):
                        if file_info.filename.endswith(('.pt', '.json')):
                            # Extract to base directory
                            zip_ref.extract(file_info, self.base_dir)
                
                print("✅ Extraction complete!")
                
                # Fix directory structure if models are in model_distribution
                distribution_dir = self.base_dir / "models" / "model_distribution"
                if distribution_dir.exists():
                    print("📁 Reorganizing model directories...")
                    import shutil
                    
                    # Move each model directory up one level
                    for model_dir in distribution_dir.iterdir():
                        if model_dir.is_dir():
                            target_dir = self.base_dir / "models" / model_dir.name
                            if target_dir.exists():
                                shutil.rmtree(target_dir)
                            shutil.move(str(model_dir), str(target_dir))
                            print(f"  Moved {model_dir.name}/")
                    
                    # Remove empty distribution directory
                    distribution_dir.rmdir()
                    print("✅ Directory structure fixed!")
                
                # Verify extraction
                new_status = self.check_existing_models()
                if self.all_models_exist(new_status):
                    print("\n🎉 All models successfully installed!")
                    self.print_status(new_status)
                    return True
                else:
                    print("\n❌ Some models missing after extraction")
                    self.print_status(new_status)
                    return False
                    
            except Exception as e:
                print(f"❌ Error during setup: {e}")
                return False
    
    def verify_installation(self):
        """Verify models can be loaded"""
        try:
            # Try to import and check models
            sys.path.append(str(Path(__file__).parent))
            from pipeline_utils import EmotionAnalyzer
            
            analyzer = EmotionAnalyzer()
            model_status = analyzer.check_models_exist()
            
            print("\n🔍 Model verification:")
            all_good = True
            for model, exists in model_status.items():
                status = "✅" if exists else "❌"
                print(f"  {model}: {status}")
                if not exists:
                    all_good = False
            
            if all_good:
                print("\n✅ All models verified and ready to use!")
            else:
                print("\n❌ Some models failed verification")
                
            return all_good
            
        except ImportError:
            print("\n⚠️  Cannot verify models (pipeline not found)")
            print("Models appear to be installed correctly.")
            return True
        except Exception as e:
            print(f"\n❌ Verification failed: {e}")
            return False


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and setup emotion analysis models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_models.py              # Download if missing
  python setup_models.py --force      # Force re-download
  python setup_models.py --verify     # Just verify existing models
        """
    )
    
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Force re-download even if models exist"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing models, don't download"
    )
    
    args = parser.parse_args()
    
    print("🚀 Steam Games Emotion Analysis - Model Setup")
    print("=" * 60)
    
    # Check dependencies
    try:
        import gdown
    except ImportError:
        print("📦 Installing gdown for Google Drive downloads...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    # Initialize setup
    setup = ModelSetup()
    
    if args.verify:
        # Just verify
        setup.verify_installation()
    else:
        # Download and setup
        success = setup.download_and_extract_models(force=args.force)
        
        if success:
            setup.verify_installation()
            print("\n🎯 Setup complete! You can now run the emotion analysis pipeline.")
        else:
            print("\n❌ Setup failed. Please check the error messages above.")
            sys.exit(1)


if __name__ == "__main__":
    main()