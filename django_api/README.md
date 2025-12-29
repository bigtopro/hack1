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

**Response (Success):**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "status": "success",
  "comment_count": 1234,
  "file_path": "/Users/alba/Desktop/hack/hack1/comments/dQw4w9WgXcQ_comments.json",
  "message": "Successfully extracted 1234 comments"
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

## Project Structure

```
hack1/
├── django_api/              # Django REST API
│   ├── api/
│   │   ├── views.py         # API endpoint handlers
│   │   ├── serializers.py   # Request/response validation
│   │   ├── urls.py          # URL routing
│   │   └── utils.py         # Business logic (extraction, file I/O)
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
└── .env                    # YouTube API keys
```

## Requirements

- **Python 3.7+**
- **Django 4.2.7**
- **Django REST Framework 3.14.0**
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
