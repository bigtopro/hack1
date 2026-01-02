#!/usr/bin/env python3
"""
Simple test to verify JSON structure handling
"""
import json
from pathlib import Path

def test_json_structure():
    """Test that we can handle both v1 and v2 JSON structures"""
    print("Testing JSON structure handling...")
    
    # Look for existing comment files
    comments_dir = Path("comments")
    comment_files = list(comments_dir.glob("*_comments.json"))
    
    if not comment_files:
        print("No comment files found in 'comments' directory")
        return
    
    print(f"Found {len(comment_files)} comment files")
    
    for file_path in comment_files[:2]:  # Test first 2 files
        print(f"\nTesting file: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  ✓ File loaded successfully: {len(data)} items")
            
            if data:
                first_item = data[0]
                print(f"  First item type: {type(first_item)}")
                
                if isinstance(first_item, str):
                    print("  Format: v1 (string array)")
                    print(f"  Sample: {first_item[:50]}...")
                elif isinstance(first_item, dict):
                    print("  Format: v2 (object array) or mixed")
                    print(f"  Keys: {list(first_item.keys())}")
                    text_content = first_item.get('text', first_item.get('comment', 'N/A'))
                    print(f"  Sample text: {text_content[:50]}...")
                else:
                    print(f"  Unknown format: {type(first_item)}")
            
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
    
    print("\nJSON structure test completed!")

def simulate_v2_format():
    """Simulate what the v2 format would look like"""
    print("\nSimulating v2 format structure:")
    
    v2_sample = [
        {
            "id": "comment_123",
            "text": "This is a sample comment",
            "publishedAt": "2023-01-01T12:00:00Z",
            "likeCount": 5,
            "parentId": None
        },
        {
            "id": "comment_456", 
            "text": "This is a reply comment",
            "publishedAt": "2023-01-01T12:05:00Z",
            "likeCount": 10,
            "parentId": "comment_123"
        }
    ]
    
    print(json.dumps(v2_sample, indent=2))

if __name__ == "__main__":
    test_json_structure()
    simulate_v2_format()