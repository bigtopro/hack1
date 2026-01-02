#!/usr/bin/env python3
"""
Test script to verify the new YouTube comments metadata features
"""
import json
from pathlib import Path
from django_api.api.utils import get_comments_from_file, get_comments_with_metadata_from_file

def test_format_compatibility():
    """Test that our new functions can handle both v1 and v2 formats"""
    print("Testing format compatibility...")
    
    # Look for existing comment files
    comments_dir = Path("comments")
    comment_files = list(comments_dir.glob("*_comments.json"))
    
    if not comment_files:
        print("No comment files found in 'comments' directory")
        return
    
    print(f"Found {len(comment_files)} comment files")
    
    for file_path in comment_files:
        print(f"\nTesting file: {file_path.name}")
        
        # Test v1 format function
        comments_v1 = get_comments_from_file(file_path)
        if comments_v1:
            print(f"  ✓ V1 function works: {len(comments_v1)} comments")
            if len(comments_v1) > 0:
                print(f"    Sample: {comments_v1[0][:50]}...")
        else:
            print(f"  ✗ V1 function failed")
        
        # Test v2 format function
        comments_v2 = get_comments_with_metadata_from_file(file_path)
        if comments_v2:
            print(f"  ✓ V2 function works: {len(comments_v2)} comments")
            if len(comments_v2) > 0:
                sample = comments_v2[0]
                if isinstance(sample, dict):
                    print(f"    Sample keys: {list(sample.keys())}")
                    print(f"    Sample text: {sample.get('text', sample.get('comment', ''))[:50]}...")
                else:
                    print(f"    Sample: {str(sample)[:50]}...")
        else:
            print(f"  ✗ V2 function failed")
    
    print("\nFormat compatibility test completed!")

if __name__ == "__main__":
    import sys
    import os
    # Add the project root to the path so we can import the Django API utils
    sys.path.insert(0, os.path.abspath('.'))
    
    # Set up Django settings
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')
    
    # Import Django modules
    import django
    django.setup()
    
    test_format_compatibility()