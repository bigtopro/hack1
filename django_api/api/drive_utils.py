"""
Google Drive utilities for uploading and downloading files
"""
import os
import json
import io
from pathlib import Path
from typing import Dict, Optional, List
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError
import pickle


# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Google Drive folder paths
DRIVE_COMMENTS_DIR = "youtubeComments"
DRIVE_EMBED_DIR = "youtubeComments/embed"


def get_drive_service(credentials_path: Optional[str] = None, token_path: Optional[str] = None):
    """
    Get authenticated Google Drive service
    
    Args:
        credentials_path: Path to credentials.json (OAuth client config)
        token_path: Path to store/load token.pickle
    
    Returns:
        Google Drive service object
    """
    creds = None
    
    # Default paths
    if not credentials_path:
        credentials_path = os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH', 'credentials.json')
    if not token_path:
        token_path = os.getenv('GOOGLE_DRIVE_TOKEN_PATH', 'token.json')
    
    # Convert Path objects to strings
    if isinstance(credentials_path, Path):
        credentials_path = str(credentials_path)
    if isinstance(token_path, Path):
        token_path = str(token_path)
    
    # Load existing token
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Google Drive credentials not found at {credentials_path}. "
                    "Please download credentials.json from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)


def find_or_create_folder(service, folder_name: str, parent_id: Optional[str] = None) -> str:
    """
    Find or create a folder in Google Drive
    
    Args:
        service: Google Drive service
        folder_name: Name of the folder
        parent_id: Parent folder ID (None for root)
    
    Returns:
        Folder ID
    """
    # Search for existing folder
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    else:
        query += " and 'root' in parents"
    
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    
    if items:
        return items[0]['id']
    
    # Create folder if it doesn't exist
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        folder_metadata['parents'] = [parent_id]
    
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')


def upload_file_to_drive(service, file_path: Path, drive_folder_path: str, file_name: Optional[str] = None) -> Dict:
    """
    Upload a file to Google Drive
    
    Args:
        service: Google Drive service
        file_path: Local file path
        drive_folder_path: Drive folder path (e.g., "youtubeComments")
        file_name: Optional custom file name
    
    Returns:
        Dict with file_id and file_url
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Navigate/create folder structure
    folder_ids = []
    current_parent = None
    
    for folder_name in drive_folder_path.split('/'):
        folder_id = find_or_create_folder(service, folder_name, current_parent)
        folder_ids.append(folder_id)
        current_parent = folder_id
    
    # File metadata
    file_name = file_name or file_path.name
    file_metadata = {
        'name': file_name,
        'parents': [current_parent] if current_parent else []
    }
    
    # Upload file
    media = MediaFileUpload(str(file_path), resumable=True)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()
    
    return {
        'file_id': file.get('id'),
        'file_url': file.get('webViewLink'),
        'file_name': file_name
    }


def check_file_exists_in_drive(service, file_name: str, drive_folder_path: str) -> Optional[Dict]:
    """
    Check if a file exists in Google Drive folder
    
    Args:
        service: Google Drive service
        file_name: Name of the file to check
        drive_folder_path: Drive folder path
    
    Returns:
        Dict with file info if exists, None otherwise
    """
    # Navigate to folder
    folder_ids = []
    current_parent = None
    
    for folder_name in drive_folder_path.split('/'):
        folder_id = find_or_create_folder(service, folder_name, current_parent)
        folder_ids.append(folder_id)
        current_parent = folder_id
    
    # Search for file
    query = f"name='{file_name}' and '{current_parent}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, modifiedTime)").execute()
    items = results.get('files', [])
    
    if items:
        return {
            'file_id': items[0]['id'],
            'file_name': items[0]['name'],
            'modified_time': items[0].get('modifiedTime'),
            'exists': True
        }
    
    return None


def download_file_from_drive(service, file_id: str, output_path: Path) -> bool:
    """
    Download a file from Google Drive
    
    Args:
        service: Google Drive service
        file_id: Google Drive file ID
        output_path: Local path to save file
    
    Returns:
        True if successful
    """
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(fh.getvalue())
        
        return True
    except HttpError as e:
        print(f"Error downloading file: {e}")
        return False


def list_files_in_drive_folder(service, drive_folder_path: str, file_extension: Optional[str] = None) -> List[Dict]:
    """
    List files in a Google Drive folder
    
    Args:
        service: Google Drive service
        drive_folder_path: Drive folder path
        file_extension: Optional file extension filter (e.g., '.npz')
    
    Returns:
        List of file info dicts
    """
    # Navigate to folder
    folder_ids = []
    current_parent = None
    
    for folder_name in drive_folder_path.split('/'):
        folder_id = find_or_create_folder(service, folder_name, current_parent)
        folder_ids.append(folder_id)
        current_parent = folder_id
    
    # List files
    query = f"'{current_parent}' in parents and trashed=false"
    if file_extension:
        # Note: Drive doesn't support extension filtering directly, we'll filter after
        pass
    
    results = service.files().list(q=query, fields="files(id, name, modifiedTime, mimeType)").execute()
    items = results.get('files', [])
    
    # Filter by extension if specified
    if file_extension:
        items = [item for item in items if item['name'].endswith(file_extension)]
    
    return [
        {
            'file_id': item['id'],
            'file_name': item['name'],
            'modified_time': item.get('modifiedTime'),
            'mime_type': item.get('mimeType')
        }
        for item in items
    ]

