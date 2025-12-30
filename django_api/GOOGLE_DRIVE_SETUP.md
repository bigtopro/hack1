# Google Drive Integration Setup

## Overview

The API now includes Google Drive integration to:
1. Upload extracted comments JSON to Drive (triggers Colab processing)
2. Check if embedding/clustering is complete
3. Download processed results (.npz files)
4. Analyze sentiment with 27 emotions

## Setup Instructions

### 1. Install Dependencies

```bash
cd django_api
pip install -r requirements.txt
```

### 2. Google Drive API Setup

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Create a new project** (or use existing)
3. **Enable Google Drive API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"
4. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop app" as application type
   - Download the credentials JSON file
5. **Save credentials**:
   - Place the downloaded file as `credentials.json` in the `hack1/` directory
   - Or set `GOOGLE_DRIVE_CREDENTIALS_PATH` in `.env`

### 3. Environment Variables

Add to your `.env` file (optional, defaults provided):

```env
GOOGLE_DRIVE_CREDENTIALS_PATH=/path/to/credentials.json
GOOGLE_DRIVE_TOKEN_PATH=/path/to/token.pickle
GOOGLE_DRIVE_COMMENTS_FOLDER=youtubeComments
GOOGLE_DRIVE_EMBED_FOLDER=youtubeComments/embed
```

### 4. First-Time Authentication

On first use, the API will:
1. Open a browser window for Google OAuth
2. Ask you to sign in and authorize
3. Save the token to `token.pickle` for future use

## New API Endpoints

### 1. Check Embedding Status
**GET** `/api/embedding/{video_id}/status/`

Check if Colab has processed the comments file.

**Response:**
```json
{
  "video_id": "rE_530gL0cs",
  "processed": true,
  "npz_file": {
    "file_id": "abc123...",
    "file_name": "rE_530gL0cs_comments_embeddings.npz",
    "modified_time": "2025-01-01T12:00:00Z",
    "exists": true
  },
  "summary_file": {...}
}
```

### 2. Download Results
**POST** `/api/embedding/{video_id}/download/`

Download clustering results from Google Drive.

**Response:**
```json
{
  "video_id": "rE_530gL0cs",
  "status": "downloaded",
  "file_path": "/path/to/results/rE_530gL0cs_comments_embeddings.npz",
  "message": "Results downloaded successfully"
}
```

### 3. Sentiment Analysis
**GET** `/api/sentiment/{video_id}/`

Get sentiment analysis with 27 emotions.

**Query Parameters:**
- `cluster_id`: Filter by cluster (optional)
- `emotion`: Filter by emotion (optional)

**Response:**
```json
{
  "video_id": "rE_530gL0cs",
  "total_comments": 10875,
  "emotions": ["admiration", "amusement", "anger", ...],
  "comments_by_emotion": {
    "anger": {
      "count": 234,
      "comments": [...]
    },
    ...
  },
  "emotion_reasons": {
    "anger": ["hate", "disgusting", "terrible", ...],
    ...
  },
  "clusters_with_emotions": {
    "0": {
      "cluster_id": 0,
      "comments": [...],
      "emotion_distribution": {
        "anger": 0.45,
        "disappointment": 0.30,
        ...
      }
    }
  },
  "has_sentiment_data": true
}
```

## Workflow

1. **Extract comments** (automatically uploads to Drive):
```bash
POST /api/extract/
{"video_id_or_url": "rE_530gL0cs"}
```

2. **Check if processed** (poll every few minutes):
```bash
GET /api/embedding/rE_530gL0cs/status/
```

3. **Download results when ready**:
```bash
POST /api/embedding/rE_530gL0cs/download/
```

4. **Get sentiment analysis**:
```bash
GET /api/sentiment/rE_530gL0cs/
```

## Notes

- The extract endpoint now automatically uploads to Drive if credentials are configured
- Colab script monitors `/content/drive/My Drive/youtubeComments/` for new files
- Processed files are saved to `/content/drive/My Drive/youtubeComments/embed/`
- Sentiment data should be saved as `{video_id}_sentiment.json` in the embed folder

