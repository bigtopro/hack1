#!/usr/bin/env python3
"""
Test script to verify the analysis engine setup
"""
import sys
import os
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False

    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False

    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False

    try:
        from groq import Groq
        print("✓ groq imported successfully")
    except ImportError as e:
        print(f"✗ groq import failed: {e}")
        return False

    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ python-dotenv import failed: {e}")
        return False

    try:
        import sys
        from pathlib import Path
        # Add the analysis directory to the Python path to import analysis_engine
        analysis_dir = Path(__file__).resolve().parent.parent / "analysis"
        sys.path.insert(0, str(analysis_dir))

        from analysis_engine import CommentAnalysisEngine
        print("✓ CommentAnalysisEngine imported successfully")
    except ImportError as e:
        print(f"✗ CommentAnalysisEngine import failed: {e}")
        return False

    return True

def main():
    print("Testing Analysis Engine Setup")
    print("=" * 40)
    
    if test_imports():
        print("\n✓ All imports successful!")
        print("\nThe analysis engine is ready to run.")
        print("\nTo start the analysis engine, use the following commands:")
        print("  source analysis_env/bin/activate")
        print("  python analysis/analysis_engine.py")
        print("\nThe engine will monitor /Users/venuvamsi/Downloads for:")
        print("  - .npz files (clustering data)")
        print("  - _sentiments.json files (sentiment analysis)")
        print("\nWhen both files are detected, it will generate an analysis report.")
        return True
    else:
        print("\n✗ Some imports failed. Please check the installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)