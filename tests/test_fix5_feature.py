#!/usr/bin/env python3
"""
Test script to verify the fix5.md feature implementation
"""
import json
import tempfile
import numpy as np
import sys
from pathlib import Path

# Add the analysis directory to the Python path to import analysis_engine
analysis_dir = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(analysis_dir))

from analysis_engine import CommentAnalysisEngine

def create_test_data():
    """Create test data to simulate the input files"""
    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create test sentiment data
    test_sentiment_data = [
        {"id": "1", "comment": "This is great!", "emotion": "joy"},
        {"id": "2", "comment": "I love this", "emotion": "joy"},
        {"id": "3", "comment": "Interesting point", "emotion": "interest"},
        {"id": "4", "comment": "I don't understand", "emotion": "confusion"},
        {"id": "5", "comment": "Can you explain more?", "emotion": "confusion"},
        {"id": "6", "comment": "This is amazing", "emotion": "joy"},
        {"id": "7", "comment": "Very informative", "emotion": "interest"},
        {"id": "8", "comment": "I'm confused about this", "emotion": "confusion"},
    ]
    
    sentiment_file = temp_dir / "test_video_comments_sentiments.json"
    with open(sentiment_file, 'w') as f:
        json.dump(test_sentiment_data, f)
    
    # Create test clustering data (simulated .npz file)
    # This simulates the output of the clustering process
    test_embeddings = np.random.rand(5, 1536)  # 5 comments with 1536-dim embeddings
    test_ids = np.array(["1", "3", "4", "6", "7"])  # deduplicated IDs
    test_labels = np.array([0, 1, 0, 0, 1])  # cluster assignments
    test_centroids = np.random.rand(2, 1536)  # 2 cluster centroids
    test_distances = np.array([0.1, 0.2, 0.15, 0.18, 0.22])  # distances to centroids
    
    npz_file = temp_dir / "test_clustering.npz"
    np.savez_compressed(
        npz_file,
        embeddings=test_embeddings,
        ids=test_ids,
        labels=test_labels,
        centroids=test_centroids,
        distances=test_distances
    )
    
    return temp_dir, npz_file, sentiment_file

def test_summary_stats():
    """Test the summary stats feature"""
    print("Creating test data...")
    temp_dir, npz_file, sentiment_file = create_test_data()
    
    print(f"Created test files:")
    print(f"  NPZ file: {npz_file}")
    print(f"  Sentiment file: {sentiment_file}")
    
    # Initialize the engine with a dummy API key for testing
    engine = CommentAnalysisEngine(download_dir=temp_dir, api_key="dummy_key")
    
    # Manually call process_analysis since we have the files
    print("\nProcessing analysis...")
    try:
        engine.process_analysis(npz_file, sentiment_file)
        
        # Look for the generated dashboard files
        dashboard_files = list(temp_dir.glob("analysis_dashboard_*.json"))
        if dashboard_files:
            dashboard_file = dashboard_files[0]
            print(f"\nDashboard JSON created: {dashboard_file}")
            
            # Load and inspect the dashboard JSON
            with open(dashboard_file, 'r') as f:
                dashboard_data = json.load(f)
            
            print("\nChecking for summary_stats section...")
            if 'summary_stats' in dashboard_data:
                print("✅ SUCCESS: summary_stats section found!")
                
                summary_stats = dashboard_data['summary_stats']
                
                # Check for required fields
                required_fields = ['overall_sentiment', 'deduplication_impact', 'cluster_sentiment_stats']
                for field in required_fields:
                    if field in summary_stats:
                        print(f"  ✅ {field}: Found")
                    else:
                        print(f"  ❌ {field}: Missing")
                        
                # Check for cluster_deduplication_signals
                if 'cluster_deduplication_signals' in summary_stats:
                    print("  ✅ cluster_deduplication_signals: Found")
                else:
                    print("  ❌ cluster_deduplication_signals: Missing")
                
                # Print a sample of the data
                print(f"\nSample overall_sentiment data:")
                print(json.dumps(summary_stats['overall_sentiment'], indent=2)[:500] + "...")
                
                print(f"\nSample deduplication_impact data:")
                print(json.dumps(summary_stats['deduplication_impact'], indent=2)[:500] + "...")
                
                print(f"\nSample cluster_sentiment_stats data:")
                print(json.dumps(summary_stats['cluster_sentiment_stats'][:2], indent=2) if summary_stats['cluster_sentiment_stats'] else "[]")
                
            else:
                print("❌ ERROR: summary_stats section not found in dashboard JSON")
                
        else:
            print("❌ ERROR: Dashboard JSON file not found")
            
        # Check for markdown file too
        md_files = list(temp_dir.glob("analysis_dashboard_*.md"))
        if md_files:
            print(f"\nDashboard markdown created: {md_files[0]}")
            with open(md_files[0], 'r') as f:
                content = f.read()
                if "Sentiment Snapshot" in content:
                    print("✅ SUCCESS: Sentiment Snapshot section found in markdown!")
                else:
                    print("❌ ERROR: Sentiment Snapshot section not found in markdown")
                    
    except Exception as e:
        print(f"❌ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up test directory: {temp_dir}")

if __name__ == "__main__":
    print("Testing fix5.md feature implementation...")
    test_summary_stats()
    print("\nTest completed!")