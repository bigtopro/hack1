#!/usr/bin/env python3
"""
Auto-Sentiment Classification for YouTube Comments
This script monitors a specific folder in Google Drive for new comment files
and automatically runs emotion classification on them using a multilingual Hugging Face model
"""

import os
os.environ["THREADPOOLCTL_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import torch
import numpy as np
import shutil
import time
import logging
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
import gc
import psutil
from contextlib import contextmanager
import threading
import glob
import sys
from packaging import version

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def install_and_check_packages():
    logger.info("Installing required packages...")

    os.system('pip install -U transformers accelerate sentencepiece packaging')
    os.system('pip install "numpy<2.0"')
    os.system('pip install -U sympy>=1.12')

    try:
        import sympy
        logger.info(f"Using sympy {sympy.__version__}")
    except Exception as e:
        logger.error(f"Sympy import failed: {e}")

    try:
        from transformers import __version__ as transformers_version
        logger.info(f"Using transformers {transformers_version} with numpy {np.__version__}")
    except Exception:
        pass

    logger.info("✅ Required packages installed and checked")

def mount_drive():
    """Mount Google Drive and handle Colab environment"""
    logger.info("Attempting to mount Google Drive...")

    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')
        logger.info("✅ Google Drive mounted successfully")
        return True
    except ModuleNotFoundError:
        logger.warning("⚠️  Not in Google Colab environment - Drive mounting skipped")
        return False

class SentimentClassifier:
    def __init__(self,
                 model_name: str = "AnasAlokla/multilingual_go_emotions",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 16,
                 use_fp16: bool = True,
                 compile_model: bool = False):
        """Initialize tokenizer/model with optional half-precision and Torch compile.

        Args:
            model_name: HuggingFace model id for emotion classification
            device: "cuda" | "cpu" | "mps"
            batch_size: batch size for classification
            use_fp16: Cast model to float16 when running on GPU for speed & memory
            compile_model: Run `torch.compile` (PyTorch ≥2.0) for kernel fusion
        """

        # Check GPU availability - this is required for this pipeline
        if device != "cuda":
            raise RuntimeError("❌ GPU is required for this pipeline. CUDA not available.")

        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size

        # Load tokenizer / model with multiple fallback strategies
        try:
            logger.info(f"Loading tokenizer and model: {model_name}")
            # First try with standard parameters
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=False)
            # Set model max length to avoid warnings and edge cases
            self.tokenizer.model_max_length = 512
        except Exception as e:
            logger.warning(f"Standard loading failed: {e}")
            logger.info("Trying with trust_remote_code=True...")
            try:
                # If standard loading fails, try with trust_remote_code=True
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e2:
                logger.warning(f"Loading with trust_remote_code=True also failed: {e2}")
                logger.info("Trying with additional parameters...")
                try:
                    # If that also fails, try with additional parameters
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=False
                    )
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=False
                    )
                except Exception as e3:
                    logger.error(f"All loading attempts failed: {e3}")
                    logger.info("Attempting to clear cache and retry...")
                    # Clear cache and try again
                    os.system("rm -rf ~/.cache/huggingface/transformers/")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
                    except Exception as e4:
                        logger.error(f"Final loading attempt failed: {e4}")
                        raise

        # Move to device & precision
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # potentially faster
            if use_fp16:
                logger.info("Converting model to half precision (fp16)")
                self.model = self.model.half()

        self.model = self.model.to(self.device)

        # Optionally compile (PyTorch 2.x) - made optional for Colab compatibility
        if compile_model and version.parse(torch.__version__) >= version.parse("2.0") and self.device == "cuda":
            try:
                logger.info("Compiling model with torch.compile() for optimization")
                self.model = torch.compile(self.model)
                logger.info("✅ Model compiled with torch.compile()")
            except Exception as compile_err:  # pragma: no cover
                logger.warning(f"torch.compile failed: {compile_err}. Continuing without compilation.")

        logger.info(f"✅ Model loaded on {self.device} (fp16={use_fp16})")
        if self.device == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {total_mem:.2f} GB")

        # Load emotion labels dynamically from model config
        self.emotion_labels = [
            self.model.config.id2label[i]
            for i in range(self.model.config.num_labels)
        ]
        logger.info(f"Loaded {len(self.emotion_labels)} emotion labels from model config")

    def classify_batch(self, texts: List[str]) -> List[str]:
        """Classify a batch of texts and return emotion labels"""
        # Tokenize the batch
        inputs = self.tokenizer(
            texts,
            padding="longest",  # Optimal for GPU batch processing
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Apply sigmoid (multi-label scores) and get top-1 predictions
            scores = torch.sigmoid(logits)   # multi-label scores
            predicted_indices = torch.argmax(scores, dim=-1)

        # Convert indices to emotion labels
        emotions = [self.emotion_labels[idx] for idx in predicted_indices.cpu().numpy()]
        return emotions

    def classify_comments(self, comments_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify emotions for a list of comment data"""
        if not comments_data:
            return []

        # Extract texts for classification
        texts = []
        for item in comments_data:
            if isinstance(item, dict):
                texts.append(item.get("comment", str(item)))
            else:
                texts.append(str(item))

        # Process in batches
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Classifying emotions"):
            batch_texts = texts[i:i + self.batch_size]
            batch_emotions = self.classify_batch(batch_texts)

            # Add emotions to the original data
            for j, emotion in enumerate(batch_emotions):
                original_idx = i + j
                original_item = comments_data[original_idx]

                if isinstance(original_item, dict):
                    result_item = {
                        "id": original_item.get("id", original_idx),
                        "comment": original_item.get("comment", str(original_item)),
                        "emotion": emotion
                    }
                else:
                    result_item = {
                        "id": original_idx,
                        "comment": str(original_item),
                        "emotion": emotion
                    }

                results.append(result_item)

        return results

def load_comments_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load comments from a JSON file, handling both v1 (string array) and v2 (object array) formats"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")

    # Convert v1 format (string array) to v2 format (object array) if needed
    converted_data = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            # This is already v2 format or another object format
            converted_data.append(item)
        elif isinstance(item, str):
            # This is v1 format (string array), convert to object format
            converted_data.append({
                "id": idx,
                "comment": item,
                "text": item  # Also include as text for compatibility
            })
        else:
            # Handle any other format by converting to string
            converted_data.append({
                "id": idx,
                "comment": str(item),
                "text": str(item)
            })

    return converted_data

def save_sentiment_results(results: List[Dict[str, Any]], output_path: str):
    """Save sentiment classification results to a JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def get_processed_files(output_folder: str) -> set:
    """Get set of already processed files by checking output folder"""
    processed = set()
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            if filename.endswith('_sentiments.json'):
                # Extract original filename from processed filename
                original_name = filename.replace('_sentiments.json', '.json')
                processed.add(original_name)
    return processed

def monitor_and_classify(
    input_folder: str,
    output_folder: str,
    classifier: SentimentClassifier,
    check_interval: float = 5.0
):
    """Monitor input folder and process new comment files"""
    logger.info(f"🔍 Starting sentiment classification monitor")
    logger.info(f"📁 Input folder: {input_folder}")
    logger.info(f"📁 Output folder: {output_folder}")
    logger.info(f"⏱️  Check interval: {check_interval}s")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    processed_files = get_processed_files(output_folder)
    logger.info(f"✅ Previously processed files: {len(processed_files)}")

    while True:
        try:
            # Look for new JSON files in input folder
            input_files = glob.glob(os.path.join(input_folder, "*.json"))
            
            new_files = []
            for file_path in input_files:
                filename = os.path.basename(file_path)
                if filename not in processed_files:
                    new_files.append(file_path)

            if new_files:
                logger.info(f"📄 Found {len(new_files)} new files to process")
                
                for file_path in new_files:
                    filename = os.path.basename(file_path)
                    logger.info(f"📄 Processing file: {filename}")
                    
                    try:
                        # Load comments from file
                        comments_data = load_comments_from_file(file_path)
                        logger.info(f"🧠 Loaded {len(comments_data)} comments for classification")
                        
                        # Classify emotions
                        logger.info("🧠 Running emotion classification...")
                        results = classifier.classify_comments(comments_data)
                        
                        # Generate output filename
                        output_filename = filename.replace('.json', '_sentiments.json')
                        output_path = os.path.join(output_folder, output_filename)
                        
                        # Save results
                        save_sentiment_results(results, output_path)
                        logger.info(f"💾 Saved results to {output_path}")
                        
                        # Mark as processed
                        processed_files.add(filename)
                        logger.info(f"✅ Completed sentiment tagging for {filename}")

                        # Clean up GPU memory to prevent fragmentation during long runs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                    except Exception as e:
                        logger.error(f"❌ Error processing {filename}: {e}")
                        continue

            # Wait before next check
            time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Sentiment classification monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Error in monitor loop: {e}")
            time.sleep(check_interval)

def main():
    """Main function to run the sentiment classification pipeline"""
    logger.info("🚀 Starting Auto Sentiment Tagger for YouTube Comments")
    
    # Install required packages
    install_and_check_packages()
    
    # Mount Google Drive
    drive_mounted = mount_drive()
    if not drive_mounted:
        logger.error("❌ Google Drive not mounted. Exiting.")
        return
    
    # Verify GPU availability (required for this pipeline)
    if not torch.cuda.is_available():
        logger.error("❌ GPU not available. This pipeline requires CUDA.")
        return
    else:
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Initialize classifier
    classifier = SentimentClassifier(
        model_name="AnasAlokla/multilingual_go_emotions",
        device="cuda",
        batch_size=16,
        use_fp16=True,
        compile_model=False  # Disabled for Colab compatibility
    )
    
    # Define folder paths
    input_folder = "/content/drive/My Drive/youtubeComments"
    output_folder = "/content/drive/My Drive/youtubeCommentsSentiments"
    
    # Start monitoring
    monitor_and_classify(
        input_folder=input_folder,
        output_folder=output_folder,
        classifier=classifier,
        check_interval=5.0  # Check every 5 seconds as requested
    )

if __name__ == "__main__":
    main()