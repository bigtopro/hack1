#!/usr/bin/env python3
"""
Test script to run analysis directly on specified files without waiting for downloads
"""

import os
from pathlib import Path
from analysis_engine import CommentAnalysisEngine

def run_test_analysis():
    """Run analysis directly on the specified files"""
    print("Starting direct analysis test...")
    
    # Define the file paths
    embeddings_file = Path("/Users/venuvamsi/Downloads/8bMh8azh3CY_comments_embeddings.npz")
    sentiments_file = Path("/Users/venuvamsi/Downloads/8bMh8azh3CY_comments_sentiments.json")
    
    # Check if files exist
    if not embeddings_file.exists():
        print(f"Error: Embeddings file not found: {embeddings_file}")
        return
    
    if not sentiments_file.exists():
        print(f"Error: Sentiments file not found: {sentiments_file}")
        return
    
    print(f"Found embeddings file: {embeddings_file}")
    print(f"Found sentiments file: {sentiments_file}")
    
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Warning: GROQ_API_KEY not found in environment. LLM functionality will be limited.")
        response = input("Do you want to continue without LLM functionality? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create analysis engine
    engine = CommentAnalysisEngine(
        download_dir="/Users/venuvamsi/Downloads",
        api_key=api_key
    )
    
    # Run the analysis directly
    try:
        print("Starting analysis...")
        engine.process_analysis(embeddings_file, sentiments_file)
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test_analysis()