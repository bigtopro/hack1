"""
Sentiment analysis utilities for processing 27 emotions
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import json


# 27 emotions mapping (based on common emotion models)
EMOTIONS_27 = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise'
]


def load_npz_results(npz_path: Path) -> Dict:
    """
    Load clustering results from .npz file
    
    Args:
        npz_path: Path to .npz file
    
    Returns:
        Dict with embeddings, ids, labels, centroids, distances
    """
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'embeddings': data['embeddings'],
        'ids': data['ids'].tolist() if hasattr(data['ids'], 'tolist') else data['ids'],
        'labels': data['labels'],
        'centroids': data['centroids'],
        'distances': data['distances']
    }


def load_sentiment_results(sentiment_path: Path) -> Dict:
    """
    Load sentiment analysis results (27 emotions) from file
    
    Expected format: JSON file with emotion predictions per comment
    {
        "comment_id": [emotion_scores or emotion_label],
        ...
    }
    
    Args:
        sentiment_path: Path to sentiment results file
    
    Returns:
        Dict mapping comment IDs to emotions
    """
    if not sentiment_path.exists():
        return {}
    
    with open(sentiment_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def map_emotions_to_comments(comments: List[str], comment_ids: List, sentiment_data: Dict) -> List[Dict]:
    """
    Map 27 emotions to comments
    
    Args:
        comments: List of comment texts
        comment_ids: List of comment IDs
        sentiment_data: Dict mapping comment IDs to emotion data
    
    Returns:
        List of dicts with comment, id, and emotions
    """
    result = []
    
    for idx, (comment, comment_id) in enumerate(zip(comments, comment_ids)):
        emotion_info = sentiment_data.get(str(comment_id), sentiment_data.get(idx, {}))
        
        # Handle different formats of emotion data
        if isinstance(emotion_info, dict):
            # If it's a dict with emotion scores
            emotions = emotion_info
        elif isinstance(emotion_info, list):
            # If it's a list of [emotion, score] pairs
            emotions = {emotion: score for emotion, score in emotion_info}
        elif isinstance(emotion_info, str):
            # If it's a single emotion label
            emotions = {emotion_info: 1.0}
        else:
            emotions = {}
        
        result.append({
            'comment_id': comment_id,
            'comment': comment,
            'emotions': emotions,
            'primary_emotion': max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
        })
    
    return result


def group_comments_by_emotion(comments_with_emotions: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group comments by their primary emotion
    
    Args:
        comments_with_emotions: List of comment dicts with emotions
    
    Returns:
        Dict mapping emotion names to lists of comments
    """
    grouped = {emotion: [] for emotion in EMOTIONS_27}
    
    for comment_data in comments_with_emotions:
        primary = comment_data.get('primary_emotion')
        if primary and primary in grouped:
            grouped[primary].append(comment_data)
        # Also add to all emotions that have non-zero scores
        for emotion, score in comment_data.get('emotions', {}).items():
            if emotion in grouped and score > 0:
                if comment_data not in grouped[emotion]:
                    grouped[emotion].append(comment_data)
    
    # Remove empty emotions
    return {k: v for k, v in grouped.items() if v}


def analyze_emotion_reasons(comments_by_emotion: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
    """
    Analyze reasons for each emotion by extracting common themes
    
    Args:
        comments_by_emotion: Dict mapping emotions to comment lists
    
    Returns:
        Dict mapping emotions to lists of reason keywords/themes
    """
    # Simple keyword extraction (can be enhanced with NLP)
    emotion_reasons = {}
    
    for emotion, comments in comments_by_emotion.items():
        if not comments:
            continue
        
        # Extract common words/phrases (simplified - can use better NLP)
        all_text = ' '.join([c['comment'] for c in comments[:50]])  # Sample first 50
        words = all_text.lower().split()
        
        # Filter common stop words (simplified list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Get most common words as "reasons"
        common_words = [word for word, count in Counter(meaningful_words).most_common(10)]
        
        emotion_reasons[emotion] = common_words
    
    return emotion_reasons


def combine_clusters_and_emotions(cluster_labels: np.ndarray, comments_with_emotions: List[Dict]) -> Dict:
    """
    Combine cluster information with emotion data
    
    Args:
        cluster_labels: Array of cluster labels for each comment
        comments_with_emotions: List of comment dicts with emotions
    
    Returns:
        Dict with cluster analysis including emotions
    """
    clusters = {}
    
    for idx, (label, comment_data) in enumerate(zip(cluster_labels, comments_with_emotions)):
        cluster_id = int(label)
        
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                'cluster_id': cluster_id,
                'comments': [],
                'emotions': {},
                'emotion_distribution': {}
            }
        
        clusters[cluster_id]['comments'].append(comment_data)
        
        # Aggregate emotions in this cluster
        for emotion, score in comment_data.get('emotions', {}).items():
            if emotion not in clusters[cluster_id]['emotions']:
                clusters[cluster_id]['emotions'][emotion] = []
            clusters[cluster_id]['emotions'][emotion].append(score)
    
    # Calculate emotion distribution per cluster
    for cluster_id, cluster_data in clusters.items():
        emotion_counts = {}
        for comment in cluster_data['comments']:
            primary = comment.get('primary_emotion')
            if primary:
                emotion_counts[primary] = emotion_counts.get(primary, 0) + 1
        
        total = len(cluster_data['comments'])
        cluster_data['emotion_distribution'] = {
            emotion: count / total
            for emotion, count in emotion_counts.items()
        }
    
    return clusters

