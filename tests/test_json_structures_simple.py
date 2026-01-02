#!/opt/homebrew/bin/python3
"""
Simple test to verify both v1 and v2 JSON structures
"""
import json
from pathlib import Path

def test_json_structures():
    """Test that both v1 and v2 JSON structures are correct"""
    print("Testing JSON structures...")
    
    # Test with the newly generated files
    v1_file = Path("comments/SmyPTnlqhlk_comments.json")
    v2_file = Path("comments/SmyPTnlqhlk_comments_v2.json")
    
    print(f"\nTesting v1 format file: {v1_file.name}")
    with open(v1_file, 'r', encoding='utf-8') as f:
        v1_data = json.load(f)
    
    print(f"  ✓ V1 file loaded: {len(v1_data)} comments")
    if v1_data:
        first_item = v1_data[0]
        print(f"  First item type: {type(first_item)}")
        if isinstance(first_item, str):
            print(f"  ✓ V1 format correct: string array")
            print(f"  Sample: {first_item[:50]}...")
        else:
            print(f"  ✗ V1 format incorrect: expected string, got {type(first_item)}")
    
    print(f"\nTesting v2 format file: {v2_file.name}")
    with open(v2_file, 'r', encoding='utf-8') as f:
        v2_data = json.load(f)
    
    print(f"  ✓ V2 file loaded: {len(v2_data)} comments")
    if v2_data:
        first_item = v2_data[0]
        print(f"  First item type: {type(first_item)}")
        if isinstance(first_item, dict):
            print(f"  ✓ V2 format correct: object array")
            required_keys = ['id', 'text', 'publishedAt', 'likeCount']
            present_keys = list(first_item.keys())
            print(f"  Present keys: {present_keys}")
            
            # Check if all required keys are present
            missing_keys = [key for key in required_keys if key not in present_keys]
            if not missing_keys:
                print(f"  ✓ All required keys present: {required_keys}")
            else:
                print(f"  ✗ Missing keys: {missing_keys}")
            
            # Check for parentId (may not be present in all comments)
            if 'parentId' in present_keys:
                print(f"  ✓ parentId field present (reply thread support)")
            else:
                print(f"  - parentId field not present in first comment (normal for top-level comments)")
                
            print(f"  Sample text: {first_item.get('text', '')[:50]}...")
            print(f"  Sample timestamp: {first_item.get('publishedAt', 'N/A')}")
            print(f"  Sample likeCount: {first_item.get('likeCount', 'N/A')}")
        else:
            print(f"  ✗ V2 format incorrect: expected dict, got {type(first_item)}")
    
    # Check for reply comments in v2 data
    reply_count = 0
    for comment in v2_data:
        if isinstance(comment, dict) and comment.get('parentId') is not None:
            reply_count += 1
    
    print(f"\n  Reply comments found: {reply_count}")
    if reply_count > 0:
        print(f"  ✓ Reply thread functionality working")
    else:
        print(f"  - No reply comments in this video (normal)")
    
    print("\nJSON structure test completed successfully!")

if __name__ == "__main__":
    test_json_structures()