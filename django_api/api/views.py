"""
API views for YouTube comments extraction
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from pathlib import Path
import os
from .utils import (
    extract_video_id_from_url,
    get_comments_from_file,
    list_available_comments,
    extract_comments
)
from .serializers import (
    ExtractCommentsSerializer, 
    CommentFileSerializer,
    EmbeddingStatusSerializer,
    DownloadResultsSerializer,
    SentimentAnalysisSerializer
)
from .drive_utils import (
    get_drive_service,
    upload_file_to_drive,
    check_file_exists_in_drive,
    download_file_from_drive
)
from .sentiment_utils import (
    load_npz_results,
    load_sentiment_results,
    map_emotions_to_comments,
    group_comments_by_emotion,
    analyze_emotion_reasons,
    combine_clusters_and_emotions,
    EMOTIONS_27
)


@api_view(['POST'])
def extract_comments_view(request):
    """
    Extract comments from a YouTube video (synchronous)
    
    POST /api/extract/
    Body: {"video_id_or_url": "dQw4w9WgXcQ"}
    
    Returns when extraction is complete.
    """
    serializer = ExtractCommentsSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    video_id_or_url = serializer.validated_data['video_id_or_url']
    video_id = extract_video_id_from_url(video_id_or_url)
    
    if not video_id:
        return Response(
            {'error': 'Invalid video ID or URL'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Extract comments synchronously (fast, direct)
    result = extract_comments(
        video_id=video_id,
        java_project_dir=settings.JAVA_PROJECT_DIR,
        comments_dir=settings.COMMENTS_DIR
    )
    
    if result['success']:
        # Upload to Google Drive if credentials are available (optional)
        drive_upload_result = None
        try:
            credentials_path = Path(settings.GOOGLE_DRIVE_CREDENTIALS_PATH)
            if credentials_path.exists():
                service = get_drive_service(
                    credentials_path=str(credentials_path),
                    token_path=str(settings.GOOGLE_DRIVE_TOKEN_PATH)
                )
                drive_result = upload_file_to_drive(
                    service=service,
                    file_path=Path(result['file_path']),
                    drive_folder_path=settings.GOOGLE_DRIVE_COMMENTS_FOLDER
                )
                drive_upload_result = {
                    'uploaded': True,
                    'file_id': drive_result['file_id'],
                    'file_url': drive_result['file_url'],
                    'message': 'File uploaded to Google Drive successfully'
                }
            else:
                drive_upload_result = {
                    'uploaded': False,
                    'message': 'Google Drive credentials not found. Extraction completed, but file not uploaded to Drive.',
                    'note': 'To enable Drive upload, add credentials.json file. See GOOGLE_DRIVE_SETUP.md'
                }
        except Exception as e:
            drive_upload_result = {
                'uploaded': False,
                'error': str(e),
                'message': 'Failed to upload to Google Drive, but extraction completed successfully'
            }
        
        response_data = {
            'video_id': video_id,
            'status': 'success',
            'comment_count': result['comment_count'],
            'file_path': result['file_path'],
            'message': f'Successfully extracted {result["comment_count"]} comments'
        }
        
        if drive_upload_result:
            response_data['drive_upload'] = drive_upload_result
        
        return Response(response_data, status=status.HTTP_200_OK)
    else:
        return Response({
            'video_id': video_id,
            'status': 'error',
            'error': result.get('error', 'Unknown error'),
            'output': result.get('output', '')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_comments_view(request, video_id):
    """
    Get comments for a specific video
    
    GET /api/comments/{video_id}/
    """
    comments_file = settings.COMMENTS_DIR / f"{video_id}_comments.json"
    
    if not comments_file.exists():
        return Response(
            {'error': f'Comments not found for video {video_id}'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    comments = get_comments_from_file(comments_file)
    
    if comments is None:
        return Response(
            {'error': 'Failed to read comments file'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    return Response({
        'video_id': video_id,
        'comment_count': len(comments),
        'comments': comments
    })


@api_view(['GET'])
def list_comments_view(request):
    """
    List all available comment files
    
    GET /api/comments/
    """
    files = list_available_comments(settings.COMMENTS_DIR)
    serializer = CommentFileSerializer(files, many=True)
    return Response({
        'count': len(files),
        'files': serializer.data
    })


@api_view(['GET'])
def check_embedding_status_view(request, video_id):
    """
    Check if embedding/clustering has been processed for a video
    
    GET /api/embedding/{video_id}/status/
    """
    try:
        credentials_path = Path(settings.GOOGLE_DRIVE_CREDENTIALS_PATH)
        if not credentials_path.exists():
            return Response({
                'video_id': video_id,
                'processed': False,
                'message': 'Google Drive credentials not configured. Cannot check Drive status.',
                'note': 'To enable Drive integration, add credentials.json file. See GOOGLE_DRIVE_SETUP.md',
                'npz_file': None,
                'summary_file': None
            }, status=status.HTTP_200_OK)
        
        service = get_drive_service(
            credentials_path=str(credentials_path),
            token_path=str(settings.GOOGLE_DRIVE_TOKEN_PATH)
        )
        
        # Check for .npz file in embed folder
        npz_file_name = f"{video_id}_comments_embeddings.npz"
        npz_file = check_file_exists_in_drive(
            service=service,
            file_name=npz_file_name,
            drive_folder_path=settings.GOOGLE_DRIVE_EMBED_FOLDER
        )
        
        # Check for JSON summary file
        json_file_name = f"{video_id}_comments_embeddings.json"
        json_file = check_file_exists_in_drive(
            service=service,
            file_name=json_file_name,
            drive_folder_path=settings.GOOGLE_DRIVE_EMBED_FOLDER
        )
        
        status_message = "processed" if npz_file else "not_processed"
        
        serializer = EmbeddingStatusSerializer(data={
            'video_id': video_id,
            'npz_file_exists': bool(npz_file),
            'npz_file_info': npz_file,
            'json_summary_exists': bool(json_file),
            'json_summary_info': json_file,
            'status': status_message,
            'message': f'Embedding status for {video_id}: {status_message}'
        })
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def download_results_view(request, video_id):
    """
    Download clustering results from Google Drive
    
    POST /api/embedding/{video_id}/download/
    """
    try:
        credentials_path = Path(settings.GOOGLE_DRIVE_CREDENTIALS_PATH)
        if not credentials_path.exists():
            return Response({
                'error': 'Google Drive credentials not configured',
                'message': 'Cannot download from Drive without credentials',
                'note': 'To enable Drive integration, add credentials.json file. See GOOGLE_DRIVE_SETUP.md'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        service = get_drive_service(
            credentials_path=str(credentials_path),
            token_path=str(settings.GOOGLE_DRIVE_TOKEN_PATH)
        )
        
        # Check for .npz file
        npz_file_name = f"{video_id}_comments_embeddings.npz"
        npz_file = check_file_exists_in_drive(
            service=service,
            file_name=npz_file_name,
            drive_folder_path=settings.GOOGLE_DRIVE_EMBED_FOLDER
        )
        
        if not npz_file:
            return Response({
                'error': f'Embedding results not found for video {video_id}',
                'message': 'File not found in Google Drive. Colab may still be processing.'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Download file
        local_path = settings.RESULTS_DIR / npz_file_name
        success = download_file_from_drive(
            service=service,
            file_id=npz_file['file_id'],
            output_path=local_path
        )
        
        if success:
            return Response({
                'video_id': video_id,
                'status': 'downloaded',
                'file_path': str(local_path),
                'message': 'Results downloaded successfully'
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'error': 'Failed to download file'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def sentiment_analysis_view(request, video_id):
    """
    Get sentiment analysis (27 emotions) for comments
    
    GET /api/sentiment/{video_id}/
    
    Query params:
    - cluster_id: Optional, filter by cluster
    - emotion: Optional, filter by emotion
    """
    try:
        # Load comments
        comments_file = settings.COMMENTS_DIR / f"{video_id}_comments.json"
        if not comments_file.exists():
            return Response({
                'error': f'Comments not found for video {video_id}'
            }, status=status.HTTP_404_NOT_FOUND)
        
        comments = get_comments_from_file(comments_file)
        if comments is None:
            return Response({
                'error': 'Failed to read comments file'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Load clustering results if available
        npz_file = settings.RESULTS_DIR / f"{video_id}_comments_embeddings.npz"
        cluster_data = None
        if npz_file.exists():
            cluster_data = load_npz_results(npz_file)
        
        # Load sentiment results (expected to be in results folder or Drive)
        sentiment_file = settings.RESULTS_DIR / f"{video_id}_sentiment.json"
        sentiment_data = {}
        if sentiment_file.exists():
            sentiment_data = load_sentiment_results(sentiment_file)
        else:
            # Try to download from Drive if available
            try:
                if os.path.exists(settings.GOOGLE_DRIVE_CREDENTIALS_PATH):
                    service = get_drive_service(
                        credentials_path=str(settings.GOOGLE_DRIVE_CREDENTIALS_PATH),
                        token_path=str(settings.GOOGLE_DRIVE_TOKEN_PATH)
                    )
                    sentiment_drive_file = check_file_exists_in_drive(
                        service=service,
                        file_name=f"{video_id}_sentiment.json",
                        drive_folder_path=settings.GOOGLE_DRIVE_EMBED_FOLDER
                    )
                    if sentiment_drive_file:
                        download_file_from_drive(
                            service=service,
                            file_id=sentiment_drive_file['file_id'],
                            output_path=sentiment_file
                        )
                        sentiment_data = load_sentiment_results(sentiment_file)
            except:
                pass
        
        # Map emotions to comments
        comment_ids = list(range(len(comments)))
        comments_with_emotions = map_emotions_to_comments(
            comments=comments,
            comment_ids=comment_ids,
            sentiment_data=sentiment_data
        )
        
        # Group by emotion
        comments_by_emotion = group_comments_by_emotion(comments_with_emotions)
        
        # Analyze emotion reasons
        emotion_reasons = analyze_emotion_reasons(comments_by_emotion)
        
        # Combine with clusters if available
        clusters_with_emotions = None
        if cluster_data:
            clusters_with_emotions = combine_clusters_and_emotions(
                cluster_labels=cluster_data['labels'],
                comments_with_emotions=comments_with_emotions
            )
        
        # Filter by query params
        cluster_id = request.query_params.get('cluster_id')
        emotion_filter = request.query_params.get('emotion')
        
        if cluster_id and clusters_with_emotions:
            clusters_with_emotions = {
                k: v for k, v in clusters_with_emotions.items() 
                if k == int(cluster_id)
            }
        
        if emotion_filter and emotion_filter in comments_by_emotion:
            comments_by_emotion = {emotion_filter: comments_by_emotion[emotion_filter]}
        
        return Response({
            'video_id': video_id,
            'total_comments': len(comments),
            'emotions': EMOTIONS_27,
            'comments_by_emotion': {
                emotion: {
                    'count': len(comments_list),
                    'comments': comments_list[:10]  # Limit to 10 per emotion
                }
                for emotion, comments_list in comments_by_emotion.items()
            },
            'emotion_reasons': emotion_reasons,
            'clusters_with_emotions': clusters_with_emotions,
            'has_sentiment_data': len(sentiment_data) > 0
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

