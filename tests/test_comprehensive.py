#!/usr/bin/env python3
"""
Comprehensive test script for analysis_engine.py
Tests the entire flow and identifies potential problems
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import tempfile
import logging

import sys
from pathlib import Path

# Add the analysis directory to the Python path to import analysis_engine
analysis_dir = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(analysis_dir))

from analysis_engine import CommentAnalysisEngine

# Set up logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_files():
    """Create test files with sample data"""
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Created temporary directory: {temp_dir}")
    
    # Create sample embeddings file (.npz)
    embeddings_file = temp_dir / "test_comments_embeddings.npz"
    sample_embeddings = np.random.rand(100, 384)  # 100 comments, 384-dimensional embeddings
    sample_ids = np.array([f"comment_{i}" for i in range(100)])
    sample_labels = np.random.randint(0, 5, size=100)  # 5 clusters
    sample_centroids = np.random.rand(5, 384)  # 5 centroids
    sample_distances = np.random.rand(100)  # distances to centroids
    
    np.savez(embeddings_file, 
             embeddings=sample_embeddings,
             ids=sample_ids,
             labels=sample_labels,
             centroids=sample_centroids,
             distances=sample_distances)
    
    # Create sample sentiments file (.json)
    sentiments_file = temp_dir / "test_comments_sentiments.json"
    sample_sentiments = []
    emotions = ["joy", "approval", "caring", "disapproval", "fear", "disappointment", "gratitude", "realization"]
    
    for i in range(100):
        sample_sentiments.append({
            "id": f"comment_{i}",
            "comment": f"This is sample comment {i} with some text content",
            "emotion": np.random.choice(emotions)
        })
    
    with open(sentiments_file, 'w') as f:
        json.dump(sample_sentiments, f, indent=2)
    
    return temp_dir, embeddings_file, sentiments_file

def test_basic_functionality():
    """Test basic functionality of the analysis engine"""
    logger.info("Testing basic functionality...")
    
    try:
        # Test initialization
        engine = CommentAnalysisEngine(download_dir="/tmp", api_key=None)
        logger.info("✓ Engine initialized successfully")
        
        # Test internal methods
        test_compute_raw_sentiment_distribution(engine)
        test_compute_cluster_weighted_sentiment_distribution(engine)
        test_calculate_entropy(engine)
        
        logger.info("✓ Basic functionality tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compute_raw_sentiment_distribution(engine):
    """Test raw sentiment distribution computation"""
    logger.info("Testing raw sentiment distribution computation...")
    
    sample_data = [
        {"emotion": "joy"},
        {"emotion": "joy"},
        {"emotion": "sadness"},
        {"emotion": "anger"},
        {"emotion": "joy"}
    ]
    
    result = engine._compute_raw_sentiment_distribution(sample_data)
    
    expected_dist = {"joy": 0.6, "sadness": 0.2, "anger": 0.2}
    expected_total = 5
    
    assert result["distribution"] == expected_dist, f"Expected {expected_dist}, got {result['distribution']}"
    assert result["total"] == expected_total, f"Expected {expected_total}, got {result['total']}"
    
    logger.info("✓ Raw sentiment distribution computation test passed")

def test_compute_cluster_weighted_sentiment_distribution(engine):
    """Test cluster-weighted sentiment distribution computation"""
    logger.info("Testing cluster-weighted sentiment distribution computation...")
    
    # Test with sample cluster sentiment map
    cluster_sentiment_map = {
        0: ["joy", "joy", "sadness"],
        1: ["joy", "joy", "joy"],
        2: ["anger", "sadness"]
    }
    
    result = engine._compute_cluster_weighted_sentiment_distribution(cluster_sentiment_map)
    
    # Expected: joy appears in 2 out of 3 clusters (0: mixed, 1: all joy, 2: mixed)
    # sadness appears in 2 out of 3 clusters
    # anger appears in 1 out of 3 clusters
    assert "joy" in result
    assert "sadness" in result
    assert "anger" in result
    
    logger.info("✓ Cluster-weighted sentiment distribution computation test passed")

def test_calculate_entropy(engine):
    """Test entropy calculation"""
    logger.info("Testing entropy calculation...")
    
    # Test with uniform distribution (max entropy)
    uniform_probs = [0.5, 0.5]
    uniform_entropy = engine._calculate_entropy(uniform_probs)
    assert uniform_entropy > 0.9, f"Uniform entropy should be high, got {uniform_entropy}"
    
    # Test with deterministic distribution (zero entropy)
    deterministic_probs = [1.0, 0.0]
    deterministic_entropy = engine._calculate_entropy(deterministic_probs)
    assert abs(deterministic_entropy) < 0.001, f"Deterministic entropy should be near zero, got {deterministic_entropy}"
    
    logger.info("✓ Entropy calculation test passed")

def test_file_processing():
    """Test the complete file processing flow"""
    logger.info("Testing complete file processing flow...")
    
    try:
        # Create test files
        temp_dir, embeddings_file, sentiments_file = create_test_files()
        logger.info(f"Created test files: {embeddings_file}, {sentiments_file}")
        
        # Create engine with API key from environment or None
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found in environment. Testing with LLM disabled.")
        
        engine = CommentAnalysisEngine(download_dir=str(temp_dir), api_key=api_key)
        
        # Test the complete processing flow
        engine.process_analysis(embeddings_file, sentiments_file)
        
        logger.info("✓ Complete file processing flow test passed")
        
        # Check if output files were created
        output_files = list(temp_dir.glob("*.json")) + list(temp_dir.glob("*.md"))
        logger.info(f"Generated output files: {[f.name for f in output_files]}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Complete file processing flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and potential problems"""
    logger.info("Testing edge cases...")
    
    try:
        # Test with empty data
        engine = CommentAnalysisEngine(api_key=None)
        
        # Test empty sentiment data
        empty_result = engine._compute_raw_sentiment_distribution([])
        assert empty_result["distribution"] == {}, f"Expected empty dict, got {empty_result['distribution']}"
        assert empty_result["total"] == 0, f"Expected 0 total, got {empty_result['total']}"
        
        # Test empty cluster sentiment map
        empty_cluster_result = engine._compute_cluster_weighted_sentiment_distribution({})
        assert empty_cluster_result == {}, f"Expected empty dict, got {empty_cluster_result}"
        
        logger.info("✓ Edge cases test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_functionality():
    """Test LLM functionality if API key is available"""
    logger.info("Testing LLM functionality...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not available, skipping LLM functionality test")
        return True
    
    try:
        engine = CommentAnalysisEngine(api_key=api_key)
        
        # Test LLM call with a simple prompt
        test_prompt = "Say 'Hello, this is a test' in one sentence."
        response = engine._call_llm(test_prompt)
        
        assert isinstance(response, str), f"Expected string response, got {type(response)}"
        assert len(response) > 0, "Expected non-empty response"
        
        logger.info("✓ LLM functionality test passed")
        return True
    except Exception as e:
        logger.error(f"✗ LLM functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("Starting comprehensive tests for analysis_engine.py")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Edge Cases", test_edge_cases),
        ("LLM Functionality", test_llm_functionality),
        ("Complete File Processing", test_file_processing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            logger.info(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.info("❌ Some tests failed")
        return False

def identify_potential_problems():
    """Identify potential problems in the code"""
    logger.info("\n" + "="*50)
    logger.info("POTENTIAL PROBLEMS IDENTIFIED")
    logger.info("="*50)
    
    problems = [
        {
            "Problem": "File path validation",
            "Description": "No validation that input files exist or are accessible before processing",
            "Severity": "Medium",
            "Location": "process_analysis method"
        },
        {
            "Problem": "Memory usage with large files", 
            "Description": "Loading entire embeddings and sentiment files into memory without size checks",
            "Severity": "High",
            "Location": "process_analysis method"
        },
        {
            "Problem": "Error handling in file operations",
            "Description": "Some file operations may fail silently or with insufficient error context",
            "Severity": "Medium",
            "Location": "Various file I/O operations"
        },
        {
            "Problem": "LLM rate limiting",
            "Description": "Rate limiting may not be sufficient if multiple instances run simultaneously",
            "Severity": "Medium",
            "Location": "_call_llm method"
        },
        {
            "Problem": "Division by zero in calculations",
            "Description": "Some calculations may divide by zero if cluster sizes are 0",
            "Severity": "Medium",
            "Location": "Various calculation methods"
        },
        {
            "Problem": "Inconsistent error handling",
            "Description": "Some methods log errors, others return error strings, inconsistent patterns",
            "Severity": "Low",
            "Location": "Various methods"
        }
    ]
    
    for i, problem in enumerate(problems, 1):
        logger.info(f"{i}. {problem['Problem']}")
        logger.info(f"   Description: {problem['Description']}")
        logger.info(f"   Severity: {problem['Severity']}")
        logger.info(f"   Location: {problem['Location']}")
        logger.info("")
    
    return problems

def suggest_fixes(problems):
    """Suggest fixes for identified problems"""
    logger.info("="*50)
    logger.info("SUGGESTED FIXES")
    logger.info("="*50)
    
    fixes = {
        "File path validation": [
            "Add file existence checks before processing",
            "Add file size validation to prevent memory issues",
            "Add proper error handling with descriptive messages"
        ],
        "Memory usage with large files": [
            "Add file size limits and streaming options",
            "Add memory usage monitoring",
            "Consider processing files in chunks for very large files"
        ],
        "Error handling in file operations": [
            "Add try-catch blocks with specific exception handling",
            "Provide more detailed error messages",
            "Add fallback mechanisms where appropriate"
        ],
        "LLM rate limiting": [
            "Add more sophisticated rate limiting with shared state",
            "Consider implementing queue-based processing",
            "Add retry logic with exponential backoff"
        ],
        "Division by zero in calculations": [
            "Add zero-division checks in all mathematical operations",
            "Use safe division functions",
            "Add validation for input parameters"
        ],
        "Inconsistent error handling": [
            "Standardize error handling patterns across the codebase",
            "Create consistent return value formats",
            "Add proper logging levels and messages"
        ]
    }
    
    for problem in problems:
        problem_name = problem["Problem"]
        logger.info(f"Fixes for '{problem_name}':")
        for i, fix in enumerate(fixes[problem_name], 1):
            logger.info(f"  {i}. {fix}")
        logger.info("")

if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Identify potential problems
    problems = identify_potential_problems()
    
    # Suggest fixes
    suggest_fixes(problems)
    
    logger.info("Testing complete!")
    sys.exit(0 if success else 1)