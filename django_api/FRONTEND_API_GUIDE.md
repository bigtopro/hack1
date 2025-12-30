# Frontend API Integration Guide

## Dashboard Endpoint

**GET** `/api/dashboard/{video_id}/`

Returns the complete analysis dashboard JSON directly (not wrapped).

### Example Request
```javascript
fetch('http://localhost:8000/api/dashboard/8bMh8azh3CY/')
  .then(res => res.json())
  .then(data => {
    // data contains the full dashboard structure
    console.log(data.meta);           // Video metadata
    console.log(data.summary_stats);   // Overall sentiment stats
    console.log(data.emotions);        // Array of emotions with percentages
    console.log(data.focus_topics);    // Array of discussion topics/clusters
  });
```

### Response Structure

The endpoint returns the dashboard JSON directly:

```json
{
  "meta": {
    "video_id": "8bMh8azh3CY",
    "total_comments": 68413,
    "embedded_comments": 68413,
    "analysis_timestamp": "2025-12-31T00:20:10.945532Z",
    "model": "llama-3.1-8b-instant"
  },
  "summary_stats": {
    "overall_sentiment": {
      "raw": {
        "neutral": 64.2,
        "admiration": 7.7,
        "curiosity": 3.8,
        ...
      },
      "topic_level": {
        "neutral": 94.0,
        "admiration": 2.0,
        ...
      },
      "dominant_raw": "neutral",
      "dominant_topic_level": "neutral"
    },
    "theme_sentiment_stats": [
      {
        "cluster_id": 7,
        "label": "Theme 7",
        "comment_count": 1720,
        "sentiment_breakdown": {...},
        "dominant_sentiment": "neutral",
        "sentiment_entropy": 2.73,
        "confidence": "low"
      },
      ...
    ]
  },
  "emotions": [
    {
      "emotion": "neutral",
      "percentage": 64.2,
      "confidence": "medium",
      "summary": "...",
      "reasons": []
    },
    {
      "emotion": "admiration",
      "percentage": 7.7,
      "confidence": "medium",
      "summary": "...",
      "reasons": []
    },
    ...
  ],
  "focus_topics": [
    {
      "cluster_id": 3,
      "label": "Topic",
      "comment_count": 3376,
      "percentage": 4.93,
      "sentiment_distribution": {
        "neutral": 71,
        "curiosity": 7,
        ...
      },
      "entropy": 1.84,
      "confidence": "low",
      "summary": "...",
      "sample_comments": {
        "centroid": ["comment 1", "comment 2", ...],
        "edge_cases": ["comment 1", "comment 2", ...]
      }
    },
    ...
  ]
}
```

### Key Data for Frontend

1. **Overall Sentiment**: `data.summary_stats.overall_sentiment.raw` - Percentages for each emotion
2. **Emotions Array**: `data.emotions` - Each emotion with percentage, confidence, summary, reasons
3. **Focus Topics**: `data.focus_topics` - Discussion clusters with sentiment distribution
4. **Theme Stats**: `data.summary_stats.theme_sentiment_stats` - Per-cluster sentiment breakdowns

### Error Response

```json
{
  "error": "Analysis dashboard not found for video {video_id}",
  "message": "No analysis dashboard files found. Run analysis first.",
  "video_id": "{video_id}"
}
```

## Complete Endpoint List

1. **POST** `/api/extract/` - Extract comments
2. **GET** `/api/comments/` - List all comment files
3. **GET** `/api/comments/{video_id}/` - Get comments for video
4. **GET** `/api/embedding/{video_id}/status/` - Check embedding status
5. **POST** `/api/embedding/{video_id}/download/` - Download results
6. **GET** `/api/sentiment/{video_id}/` - Get sentiment analysis
7. **GET** `/api/dashboard/{video_id}/` - Get analysis dashboard ⭐
8. **POST** `/api/analyze/{video_id}/` - Trigger analysis engine

## Frontend Integration Example

```javascript
// Fetch dashboard data
async function loadDashboard(videoId) {
  try {
    const response = await fetch(`http://localhost:8000/api/dashboard/${videoId}/`);
    if (!response.ok) {
      throw new Error('Dashboard not found');
    }
    const dashboard = await response.json();
    
    // Use the data
    console.log('Total comments:', dashboard.meta.total_comments);
    console.log('Emotions:', dashboard.emotions);
    console.log('Topics:', dashboard.focus_topics);
    
    // Update your UI
    updateEmotionChart(dashboard.emotions);
    updateTopicList(dashboard.focus_topics);
    updateSentimentStats(dashboard.summary_stats);
    
    return dashboard;
  } catch (error) {
    console.error('Error loading dashboard:', error);
  }
}
```

