#!/opt/homebrew/bin/python3
"""
Test script to verify the Django API utils can handle both v1 and v2 formats
"""
import json
from pathlib import Path
import sys
import os

# Add the project root to the path so we can import the Django API utils
sys.path.insert(0, os.path.abspath('.'))

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')

# Import Django modules
import django
django.setup()

from django_api.api.utils import get_comments_from_file, get_comments_with_metadata_from_file

def test_api_utils():
    """Test that our API utility functions work correctly with both formats"""
    print("Testing Django API utility functions...")
    
    # Test with the newly generated files
    v1_file = Path("comments/SmyPTnlqhlk_comments.json")
    v2_file = Path("comments/SmyPTnlqhlk_comments_v2.json")
    
    print(f"\nTesting v1 format file: {v1_file.name}")
    comments_v1 = get_comments_from_file(v1_file)
    if comments_v1:
        print(f"  ✓ V1 function works: {len(comments_v1)} comments")
        print(f"  Sample: {comments_v1[0][:50]}...")
    else:
        print(f"  ✗ V1 function failed")
    
    print(f"\nTesting v2 format file: {v2_file.name}")
    comments_v2 = get_comments_with_metadata_from_file(v2_file)
    if comments_v2:
        print(f"  ✓ V2 function works: {len(comments_v2)} comments")
        sample = comments_v2[0]
        if isinstance(sample, dict):
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Sample text: {sample.get('text', '')[:50]}...")
            print(f"  Sample timestamp: {sample.get('publishedAt', 'N/A')}")
            print(f"  Sample likeCount: {sample.get('likeCount', 'N/A')}")
            print(f"  Sample parentId: {sample.get('parentId', 'N/A')}")
    else:
        print(f"  ✗ V2 function failed")
    
    # Test that v2 function can also handle v1 format (backward compatibility)
    print(f"\nTesting v2 function with v1 format file (backward compatibility):")
    comments_v2_from_v1 = get_comments_with_metadata_from_file(v1_file)
    if comments_v2_from_v1:
        print(f"  ✓ V2 function handles v1 format: {len(comments_v2_from_v1)} comments")
        sample = comments_v2_from_v1[0]
        if isinstance(sample, dict):
            print(f"  Converted to object with keys: {list(sample.keys())}")
    else:
        print(f"  ✗ V2 function failed with v1 format")
    
    print("\nAPI utility test completed!")

if __name__ == "__main__":
    test_api_utils()