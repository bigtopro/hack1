"""
Utility functions for YouTube comments extraction
"""
import subprocess
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, List


def extract_video_id_from_url(url_or_id: str) -> Optional[str]:
    """Extract video ID from YouTube URL or return as-is if already an ID"""
    if not url_or_id:
        return None
    
    # If it's already a video ID (11 characters, alphanumeric)
    if len(url_or_id) == 11 and url_or_id.replace('-', '').replace('_', '').isalnum():
        return url_or_id
    
    # Try to extract from various YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/channel\/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    return url_or_id  # Return as-is if no pattern matches


def extract_comments(video_id: str, java_project_dir: Path, comments_dir: Path) -> Dict:
    """
    Call the Java YouTubeCommentsExtractor to extract comments
    Returns dict with status and file path for both v1 and v2
    """
    comments_dir.mkdir(parents=True, exist_ok=True)
    output_file = comments_dir / f"{video_id}_comments.json"
    output_v2_file = comments_dir / f"{video_id}_comments_v2.json"

    # Check if both files already exist
    if output_file.exists() and output_v2_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
            with open(output_v2_file, 'r', encoding='utf-8') as f:
                comments_v2 = json.load(f)
                return {
                    'success': True,
                    'file_path': str(output_file),
                    'file_path_v2': str(output_v2_file),
                    'comment_count': len(comments),
                    'comment_count_v2': len(comments_v2),
                    'output': 'Files already exist, skipping extraction.',
                    'cached': True
                }
        except:
            # Files exist but corrupted, re-extract
            pass

    # Change to Java project directory
    original_cwd = os.getcwd()
    try:
        os.chdir(java_project_dir)

        # Build the Maven command
        cmd = [
            'mvn', 'exec:java',
            '-Dexec.mainClass=com.lucy.YouTubeCommentsExtractor',
            f'-Dexec.args={video_id}'
        ]

        # Run the Java extractor
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            return {
                'success': False,
                'error': result.stderr,
                'output': result.stdout
            }

        # Check if both files were created
        if output_file.exists() and output_v2_file.exists():
            # Read and return comment counts
            with open(output_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
            with open(output_v2_file, 'r', encoding='utf-8') as f:
                comments_v2 = json.load(f)
                return {
                    'success': True,
                    'file_path': str(output_file),
                    'file_path_v2': str(output_v2_file),
                    'comment_count': len(comments),
                    'comment_count_v2': len(comments_v2),
                    'output': result.stdout
                }
        else:
            return {
                'success': False,
                'error': 'Comment files were not created properly',
                'output': result.stdout
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Extraction timed out after 10 minutes'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        os.chdir(original_cwd)


def get_comments_from_file(file_path: Path) -> Optional[List[str]]:
    """Read comments from JSON file, handling both v1 (string array) and v2 (object array) formats"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if this is v2 format (list of objects with 'text' field)
        if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'text' in data[0]:
            # This is v2 format, extract just the text
            return [item['text'] for item in data]
        else:
            # This is v1 format (list of strings)
            return data
    except Exception as e:
        return None


def get_comments_with_metadata_from_file(file_path: Path) -> Optional[List[Dict]]:
    """Read comments with metadata from JSON file, handling both v1 and v2 formats"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if this is v1 format (list of strings)
        if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            # Convert v1 format to v2 format with default metadata
            return [{'id': f'comment_{i}', 'text': text, 'publishedAt': None, 'likeCount': 0, 'parentId': None}
                    for i, text in enumerate(data)]
        else:
            # This is already v2 format (list of objects)
            return data
    except Exception as e:
        return None


def list_available_comments(comments_dir: Path) -> List[Dict]:
    """List all available comment files (both v1 and v2)"""
    if not comments_dir.exists():
        return []

    files = []
    # Get both v1 and v2 files
    for json_file in comments_dir.glob("*_comments.json"):
        if "_comments_v2.json" not in str(json_file):  # Skip v2 files in this loop
            video_id = json_file.stem.replace('_comments', '')
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    comments = json.load(f)
                    files.append({
                        'video_id': video_id,
                        'file_name': json_file.name,
                        'comment_count': len(comments),
                        'file_path': str(json_file),
                        'version': 'v1'
                    })
            except:
                pass

    # Also include v2 files
    for json_file in comments_dir.glob("*_comments_v2.json"):
        video_id = json_file.stem.replace('_comments_v2', '')
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
                files.append({
                    'video_id': video_id,
                    'file_name': json_file.name,
                    'comment_count': len(comments),
                    'file_path': str(json_file),
                    'version': 'v2'
                })
        except:
            pass

    return files


