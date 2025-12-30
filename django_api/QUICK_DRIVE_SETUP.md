# Quick Google Drive Setup Guide

## What You Need

You need a `credentials.json` file from Google Cloud Console. This file allows the API to upload files to your Google Drive.

## Step-by-Step Instructions

### 1. Go to Google Cloud Console
Visit: https://console.cloud.google.com/

### 2. Create or Select a Project
- If you don't have a project, click "Create Project"
- Give it a name (e.g., "YouTube Comments API")
- Click "Create"

### 3. Enable Google Drive API
- In the left sidebar, go to **"APIs & Services"** > **"Library"**
- Search for **"Google Drive API"**
- Click on it and press **"Enable"**

### 4. Configure OAuth Consent Screen
- Go to **"APIs & Services"** > **"OAuth consent screen"**
- Choose **"External"** (unless you have a Google Workspace)
- Click **"Create"**
- Fill in:
  - **App name**: "YouTube Comments API" (or any name)
  - **User support email**: Your email
  - **Developer contact information**: Your email
- Click **"Save and Continue"**
- On **"Scopes"** page, click **"Save and Continue"** (default scopes are fine)
- On **"Test users"** page, click **"+ ADD USERS"** and add your Google account email
- Click **"Save and Continue"**
- Click **"Back to Dashboard"**

### 5. Create OAuth Credentials
- Go to **"APIs & Services"** > **"Credentials"**
- Click **"+ CREATE CREDENTIALS"** at the top
- Select **"OAuth client ID"**
- Choose **"Desktop app"** as the application type
- Give it a name (e.g., "Django API Client")
- Click **"Create"**

### 6. Download credentials.json
- A popup will show your Client ID and Client Secret
- Click the **"DOWNLOAD JSON"** button (top right of the popup)
- This downloads a file like `client_secret_xxxxx.json`

### 7. Rename and Place the File
```bash
# Rename the downloaded file to credentials.json
mv ~/Downloads/client_secret_*.json /Users/alba/Desktop/hack/hack1/credentials.json
```

Or manually:
- Rename the downloaded file to `credentials.json`
- Move it to: `/Users/alba/Desktop/hack/hack1/credentials.json`
  (Same folder as your `pom.xml` and `django_api` folder)

### 8. First-Time Authentication
When you first use the API with Drive:
1. The API will open a browser window
2. Sign in with your Google account
3. Click **"Allow"** to grant permissions
4. A `token.json` file will be created automatically in `hack1/` folder
5. You won't need to authenticate again (until token expires)

## Verify It Works

After placing `credentials.json`, test the extract endpoint:

```bash
curl -X POST http://localhost:8000/api/extract/ \
  -H "Content-Type: application/json" \
  -d '{"video_id_or_url": "https://www.youtube.com/watch?v=rE_530gL0cs"}'
```

You should see in the response:
```json
{
  "drive_upload": {
    "uploaded": true,
    "file_id": "...",
    "file_url": "..."
  }
}
```

## File Structure

After setup, your `hack1/` folder should have:
```
hack1/
├── credentials.json    ← You add this
├── token.json          ← Auto-created on first use
├── django_api/
├── pom.xml
└── ...
```

## Troubleshooting

**"Credentials not found"**
- Make sure `credentials.json` is in `/Users/alba/Desktop/hack/hack1/`
- Check the filename is exactly `credentials.json` (not `credentials.json.json`)

**"Access denied" or "Permission error"**
- Make sure you added yourself as a test user in OAuth consent screen
- Make sure you clicked "Allow" when the browser opened

**"Token expired"**
- Delete `token.json` and run the API again (it will re-authenticate)

