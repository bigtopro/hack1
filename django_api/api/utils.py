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
    Returns dict with status and file path
    """
    comments_dir.mkdir(parents=True, exist_ok=True)
    output_file = comments_dir / f"{video_id}_comments.json"
    
    # Check if file already exists
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
                return {
                    'success': True,
                    'file_path': str(output_file),
                    'comment_count': len(comments),
                    'output': 'File already exists, skipping extraction.',
                    'cached': True
                }
        except:
            # File exists but corrupted, re-extract
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
        
        # Check if file was created
        if output_file.exists():
            # Read and return comment count
            with open(output_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
                return {
                    'success': True,
                    'file_path': str(output_file),
                    'comment_count': len(comments),
                    'output': result.stdout
                }
        else:
            return {
                'success': False,
                'error': 'Comment file was not created',
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
    """Read comments from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None


def list_available_comments(comments_dir: Path) -> List[Dict]:
    """List all available comment files"""
    if not comments_dir.exists():
        return []
    
    files = []
    for json_file in comments_dir.glob("*_comments.json"):
        video_id = json_file.stem.replace('_comments', '')
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
                files.append({
                    'video_id': video_id,
                    'file_name': json_file.name,
                    'comment_count': len(comments),
                    'file_path': str(json_file)
                })
        except:
            pass
    
    return files


