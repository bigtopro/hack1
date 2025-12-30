# Django REST API for YouTube Comments Extractor

This Django REST API provides a simple, fast interface to extract YouTube comments using the existing Java-based extractor backend.

## Architecture

### System Overview

```
┌─────────────────┐
│   Frontend      │
│   (Your App)    │
└────────┬────────┘
         │ HTTP REST API
         ▼
┌─────────────────────────────────────┐
│   Django REST API                   │
│   (django_api/)                     │
│                                     │
│   ┌─────────────────────────────┐  │
│   │  API Endpoints              │  │
│   │  - POST /api/extract/       │  │
│   │  - GET /api/comments/       │  │
│   │  - GET /api/comments/{id}/  │  │
│   └───────────┬─────────────────┘  │
│               │                     │
│   ┌───────────▼─────────────────┐  │
│   │  Utils Layer                 │  │
│   │  - URL parsing               │  │
│   │  - Comment extraction        │  │
│   │  - File management           │  │
│   └───────────┬─────────────────┘  │
└───────────────┼─────────────────────┘
                │ Subprocess call
                ▼
┌─────────────────────────────────────┐
│   Java Backend                      │
│   (hack1/src/main/java/)            │
│                                     │
│   ┌─────────────────────────────┐  │
│   │  YouTubeCommentsExtractor   │  │
│   │  - YouTube Data API v3      │  │
│   │  - Multi-threaded extraction │  │
│   │  - API key rotation        │  │
│   └───────────┬─────────────────┘  │
└───────────────┼─────────────────────┘
                │ Writes JSON
                ▼
┌─────────────────────────────────────┐
│   File System                       │
│   hack1/comments/                   │
│   {video_id}_comments.json         │
└───────────┬─────────────────────────┘
            │ Uploads to Drive
            ▼
┌─────────────────────────────────────┐
│   Google Drive                       │
│   youtubeComments/                   │
│   ├── {video_id}_comments.json      │
│   └── embed/                         │
│       ├── {video_id}_embeddings.npz │
│       └── {video_id}_sentiment.json │
└─────────────────────────────────────┘
```

### Component Details

1. **Django REST API Layer**
   - Handles HTTP requests/responses
   - Validates input (video IDs/URLs)
   - Manages file I/O
   - Returns JSON responses

2. **Java Backend Integration**
   - Django calls Java extractor via Maven subprocess
   - Java extractor uses YouTube Data API v3
   - Multi-threaded for performance
   - Handles API key rotation and rate limiting

3. **Data Flow**
   - Frontend sends video ID/URL → Django API
   - Django parses video ID → Calls Java extractor
   - Java extracts comments → Saves to JSON file
   - Django reads file → Returns to frontend

### Design Decisions

- **Synchronous Processing**: Simple, fast, returns when done (no polling needed)
- **No Database**: Comments stored as JSON files (simple, portable)
- **Direct Integration**: Django directly calls Java via subprocess (no message queues)
- **Stateless API**: Each request is independent

## Setup

1. **Create and activate a virtual environment:**
```bash
cd django_api
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run migrations:**
```bash
python manage.py migrate
```

4. **Start the development server:**
```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/`

## Google Drive Integration Setup

The API can automatically upload extracted comments to Google Drive and download processed results. This enables integration with Colab notebooks for AI processing (embeddings, clustering, sentiment analysis).

### Prerequisites

1. **Google Cloud Project**: Create or use an existing project at [Google Cloud Console](https://console.cloud.google.com/)
2. **Enable Google Drive API**: 
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API" and enable it
3. **OAuth Consent Screen**:
   - Go to "APIs & Services" > "OAuth consent screen"
   - Choose "External" and complete the setup
   - Add yourself as a test user (your Google account email)

### Getting Credentials

1. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop app" as application type
   - Download the JSON file

2. **Save credentials.json**:
   - Rename the downloaded file to `credentials.json`
   - Place it in the `hack1/` directory (same level as `django_api/` folder)
   - Path: `/Users/alba/Desktop/hack/hack1/credentials.json`

3. **First-Time Authentication**:
   - When you first use the API with Drive, a browser window will open
   - Sign in with your Google account and authorize the app
   - A `token.json` file will be created automatically in `hack1/` folder
   - You won't need to authenticate again until the token expires

### Drive Folder Structure

The API expects this structure in your Google Drive:

```
My Drive/
└── youtubeComments/
    ├── {video_id}_comments.json          ← Uploaded by Django API
    └── embed/
        ├── {video_id}_comments_embeddings.npz      ← Created by Colab
        ├── {video_id}_comments_embeddings.json     ← Created by Colab
        └── {video_id}_sentiment.json                ← Created by Colab
```

Folders are created automatically if they don't exist.

### Optional: Without Google Drive

The API works without Google Drive credentials. If `credentials.json` is not found:
- Comments are still extracted and saved locally
- Drive upload is skipped (with a message in the response)
- All other endpoints work normally

### Configuration

The following paths are configured in `config/settings.py`:

```python
GOOGLE_DRIVE_CREDENTIALS_PATH = 'credentials.json'  # OAuth credentials
GOOGLE_DRIVE_TOKEN_PATH = 'token.json'              # Saved auth token
GOOGLE_DRIVE_COMMENTS_FOLDER = 'youtubeComments'     # Upload folder
GOOGLE_DRIVE_EMBED_FOLDER = 'youtubeComments/embed'  # Results folder
```

These files should be placed in the `hack1/` directory (same level as `django_api/`).

### Security Note

- `credentials.json` and `token.json` contain sensitive OAuth credentials
- Keep them secure and don't share them publicly
- These files are already excluded from version control

## API Endpoints

### 1. Extract Comments
**POST** `/api/extract/`

Extract comments from a YouTube video. This endpoint runs synchronously and returns when extraction is complete.

**Request Body:**
```json
{
  "video_id_or_url": "dQw4w9WgXcQ"
}
```

**Request Examples:**
```bash
# Using video ID
curl -X POST http://localhost:8000/api/extract/ \
  -H "Content-Type: application/json" \
  -d '{"video_id_or_url": "dQw4w9WgXcQ"}'

# Using YouTube URL
curl -X POST http://localhost:8000/api/extract/ \
  -H "Content-Type: application/json" \
  -d '{"video_id_or_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

**Response (Success with Drive):**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "status": "success",
  "comment_count": 1234,
  "file_path": "/Users/alba/Desktop/hack/hack1/comments/dQw4w9WgXcQ_comments.json",
  "message": "Successfully extracted 1234 comments",
  "drive_upload": {
    "uploaded": true,
    "file_id": "1abc123...",
    "file_url": "https://drive.google.com/file/d/1abc123.../view",
    "message": "File uploaded to Google Drive successfully"
  }
}
```

**Response (Success without Drive):**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "status": "success",
  "comment_count": 1234,
  "file_path": "/Users/alba/Desktop/hack/hack1/comments/dQw4w9WgXcQ_comments.json",
  "message": "Successfully extracted 1234 comments",
  "drive_upload": {
    "uploaded": false,
    "message": "Google Drive credentials not found. Extraction completed, but file not uploaded to Drive."
  }
}
```

**Response (Error):**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "status": "error",
  "error": "Error message here",
  "output": "Java output..."
}
```

**Notes:**
- Extraction typically takes 2-5 minutes depending on comment count
- The request blocks until extraction completes
- Comments are saved to `hack1/comments/{video_id}_comments.json`

### 2. List All Comment Files
**GET** `/api/comments/`

List all available comment files that have been extracted.

**Request:**
```bash
curl http://localhost:8000/api/comments/
```

**Response:**
```json
{
  "count": 2,
  "files": [
    {
      "video_id": "dQw4w9WgXcQ",
      "file_name": "dQw4w9WgXcQ_comments.json",
      "comment_count": 1234,
      "file_path": "/Users/name/Desktop/hack/hack1/comments/dQw4w9WgXcQ_comments.json"
    },
    {
      "video_id": "rE_530gL0cs",
      "file_name": "rE_530gL0cs_comments.json",
      "comment_count": 10875,
      "file_path": "/Users/name/Desktop/hack/hack1/comments/rE_530gL0cs_comments.json"
    }
  ]
}
```

### 3. Get Comments for a Video
**GET** `/api/comments/{video_id}/`

Retrieve all comments for a specific video.

**Request:**
```bash
curl http://localhost:8000/api/comments/dQw4w9WgXcQ/
```

**Response:**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "comment_count": 1234,
  "comments": [
    "This is a great video!",
    "Love this content",
    "Amazing work!",
    ...
  ]
}
```

**Error Response (404):**
```json
{
  "error": "Comments not found for video dQw4w9WgXcQ"
}
```

### 4. Check Embedding Status
**GET** `/api/embedding/{video_id}/status/`

Check if embedding/clustering processing has been completed for a video. This checks Google Drive for processed files.

**Request:**
```bash
curl http://localhost:8000/api/embedding/rE_530gL0cs/status/
```

**Response (Processed):**
```json
{
  "video_id": "rE_530gL0cs",
  "npz_file_exists": true,
  "npz_file_info": {
    "id": "1abc123...",
    "name": "rE_530gL0cs_comments_embeddings.npz",
    "webViewLink": "https://drive.google.com/file/d/.../view"
  },
  "json_summary_exists": true,
  "json_summary_info": {...},
  "status": "processed",
  "message": "Embedding status for rE_530gL0cs: processed"
}
```

**Response (Not Processed):**
```json
{
  "video_id": "rE_530gL0cs",
  "npz_file_exists": false,
  "npz_file_info": null,
  "json_summary_exists": false,
  "json_summary_info": null,
  "status": "not_processed",
  "message": "Embedding status for rE_530gL0cs: not_processed"
}
```

**Response (No Credentials):**
```json
{
  "video_id": "rE_530gL0cs",
  "processed": false,
  "message": "Google Drive credentials not configured. Cannot check Drive status.",
  "note": "To enable Drive integration, add credentials.json file. See GOOGLE_DRIVE_SETUP.md"
}
```

### 5. Download Results
**POST** `/api/embedding/{video_id}/download/`

Download processed embedding/clustering results from Google Drive to local storage.

**Request:**
```bash
curl -X POST http://localhost:8000/api/embedding/rE_530gL0cs/download/
```

**Response (Success):**
```json
{
  "video_id": "rE_530gL0cs",
  "status": "downloaded",
  "file_path": "/Users/alba/Desktop/hack/hack1/results/rE_530gL0cs_comments_embeddings.npz",
  "message": "Results downloaded successfully"
}
```

**Response (Not Found):**
```json
{
  "error": "Embedding results not found for video rE_530gL0cs",
  "message": "File not found in Google Drive. Colab may still be processing."
}
```

### 6. Sentiment Analysis
**GET** `/api/sentiment/{video_id}/`

Get sentiment analysis results with 27 emotions, optionally filtered by cluster or emotion.

**Query Parameters:**
- `cluster_id` (optional): Filter by cluster ID
- `emotion` (optional): Filter by emotion name

**Request:**
```bash
# Get all sentiment data
curl http://localhost:8000/api/sentiment/rE_530gL0cs/

# Filter by emotion
curl "http://localhost:8000/api/sentiment/rE_530gL0cs/?emotion=anger"

# Filter by cluster
curl "http://localhost:8000/api/sentiment/rE_530gL0cs/?cluster_id=0"
```

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
    }
  },
  "emotion_reasons": {...},
  "clusters_with_emotions": {...},
  "has_sentiment_data": true
}
```

**Note:** Requires sentiment data file (`{video_id}_sentiment.json`) to be downloaded locally first.

## Project Structure

```
hack1/
├── django_api/              # Django REST API
│   ├── api/
│   │   ├── views.py         # API endpoint handlers
│   │   ├── serializers.py   # Request/response validation
│   │   ├── urls.py          # URL routing
│   │   ├── utils.py         # Business logic (extraction, file I/O)
│   │   ├── drive_utils.py   # Google Drive integration
│   │   └── sentiment_utils.py # Sentiment analysis utilities
│   ├── config/
│   │   ├── settings.py      # Django configuration
│   │   └── urls.py          # Root URL configuration
│   ├── manage.py
│   └── requirements.txt
├── src/main/java/           # Java backend
│   └── com/lucy/
│       ├── YouTubeCommentsExtractor.java
│       └── YoutubeCommentScraper.java
├── comments/               # Extracted comments (JSON files)
│   ├── {video_id}_comments.json
│   └── ...
├── results/                # Downloaded processed results
│   ├── {video_id}_comments_embeddings.npz
│   └── {video_id}_sentiment.json
├── credentials.json        # Google Drive OAuth credentials
├── token.json              # Google Drive auth token
└── .env                    # YouTube API keys
```

## Requirements

- **Python 3.7+**
- **Django 4.2.7**
- **Django REST Framework 3.14.0**
- **Google API Python Client** (for Drive integration)
- **numpy** (for sentiment analysis)
- **Maven** (for running Java extractor)
- **Java 8+** (for the YouTube extractor)
- **YouTube Data API v3 keys** (in `.env` file)

## Configuration

### Environment Variables

The API reads YouTube API keys from the `.env` file in the parent `hack1/` directory:

```env
YOUTUBE_API_KEY_1=your_api_key_here
YOUTUBE_API_KEY_2=your_api_key_here
# ... up to YOUTUBE_API_KEY_20
```

### Paths

- **Comments Directory**: `hack1/comments/`
- **Java Project**: `hack1/` (parent directory)
- **API Keys**: `hack1/.env`

## Usage Examples

### Complete Workflow

1. **Extract comments from a video:**
```bash
curl -X POST http://localhost:8000/api/extract/ \
  -H "Content-Type: application/json" \
  -d '{"video_id_or_url": "https://www.youtube.com/watch?v=rE_530gL0cs"}'
```

2. **List all extracted comment files:**
```bash
curl http://localhost:8000/api/comments/
```

3. **Get comments for a specific video:**
```bash
curl http://localhost:8000/api/comments/rE_530gL0cs/
```

## Testing

### Quick Test

1. **Start the server:**
```bash
cd django_api
python manage.py runserver
```

2. **Test extract endpoint:**
```bash
curl -X POST http://localhost:8000/api/extract/ \
  -H "Content-Type: application/json" \
  -d '{"video_id_or_url": "https://www.youtube.com/watch?v=rE_530gL0cs"}'
```

Expected: Returns success with comment count and optional Drive upload info.

3. **Test list comments:**
```bash
curl http://localhost:8000/api/comments/
```

Expected: Returns list of all extracted comment files.

4. **Test get comments:**
```bash
curl http://localhost:8000/api/comments/rE_530gL0cs/
```

Expected: Returns all comments for the video.

### Testing with Google Drive

If you have `credentials.json` configured:

1. **Test embedding status:**
```bash
curl http://localhost:8000/api/embedding/rE_530gL0cs/status/
```

2. **Test download results (after Colab processing):**
```bash
curl -X POST http://localhost:8000/api/embedding/rE_530gL0cs/download/
```

3. **Test sentiment analysis:**
```bash
curl http://localhost:8000/api/sentiment/rE_530gL0cs/
```

### Expected Responses

- **200 OK**: Successful request
- **400 Bad Request**: Invalid input (e.g., invalid video ID)
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Google Drive not configured (for Drive endpoints)

For detailed test results and expected responses, see `TEST_RESULTS.md`.

## Notes

- **Synchronous Processing**: The extract endpoint blocks until completion (typically 2-5 minutes)
- **File Storage**: Comments are saved as JSON files in `hack1/comments/`
- **API Key Management**: The Java backend handles API key rotation automatically
- **Error Handling**: Errors from the Java extractor are returned in the API response
- **URL Parsing**: The API accepts both video IDs and full YouTube URLs

## Error Handling

The API returns appropriate HTTP status codes:

- **200 OK**: Success
- **400 Bad Request**: Invalid input (e.g., invalid video ID)
- **404 Not Found**: Comments file not found
- **500 Internal Server Error**: Extraction failed

Error responses include a descriptive `error` field with details about what went wrong.
