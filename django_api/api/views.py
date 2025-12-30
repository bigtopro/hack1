"""
API views for YouTube comments extraction
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from pathlib import Path
import os
import json
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



@api_view(["GET"])
def get_analysis_dashboard_view(request, video_id):
    """
    Get the analysis dashboard JSON for a specific video

    GET /api/dashboard/{video_id}/
    """
    try:
        # Look for the dashboard JSON file in the Downloads directory
        # The analysis engine saves files with timestamp in the name
        import glob
        import os
        from django.conf import settings
        from pathlib import Path
        
        # Search for dashboard files in the Downloads directory
        download_dir = Path("/Users/venuvamsi/Downloads")  # Default path from analysis_engine.py
        dashboard_files = list(download_dir.glob(f"analysis_dashboard_{video_id}_*.json"))
        
        # If not found in Downloads, also check the results directory
        if not dashboard_files:
            results_dir = settings.RESULTS_DIR
            dashboard_files = list(results_dir.glob(f"analysis_dashboard_{video_id}_*.json"))
        
        # If still not found, look for any dashboard file with the video_id pattern
        if not dashboard_files:
            dashboard_files = list(download_dir.glob(f"*analysis_dashboard*{video_id}*.json"))
            if not dashboard_files:
                dashboard_files = list(settings.RESULTS_DIR.glob(f"*analysis_dashboard*{video_id}*.json"))
        
        # If still not found, look for the most recent dashboard file
        if not dashboard_files:
            all_dashboard_files = list(download_dir.glob("analysis_dashboard_*.json")) + \
                                 list(settings.RESULTS_DIR.glob("analysis_dashboard_*.json"))
            if all_dashboard_files:
                # Get the most recent dashboard file
                import os
                dashboard_files = [max(all_dashboard_files, key=os.path.getctime)]
        
        if dashboard_files:
            dashboard_file = dashboard_files[0]  # Take the first (or most recent) file
            with open(dashboard_file, "r") as f:
                dashboard_data = json.load(f)
            
            return Response({
                "video_id": video_id,
                "dashboard_data": dashboard_data,
                "file_path": str(dashboard_file),
                "status": "success"
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                "error": f"Analysis dashboard not found for video {video_id}",
                "message": "The analysis may not have completed yet or the dashboard file is missing.",
                "video_id": video_id
            }, status=status.HTTP_404_NOT_FOUND)
    
    except Exception as e:
        return Response({
            "error": str(e),
            "video_id": video_id
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
def trigger_analysis_view(request, video_id):
    """
    Trigger the analysis engine for a specific video

    POST /api/analyze/{video_id}/
    """
    try:
        # Import the analysis engine
        import sys
        from pathlib import Path
        
        # Add the parent directory to the Python path to import analysis_engine
        parent_dir = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from analysis_engine import CommentAnalysisEngine
        
        # Find the required files for analysis
        # Look for .npz and _sentiments.json files
        download_dir = Path("/Users/venuvamsi/Downloads")  # Default path from analysis_engine.py
        results_dir = settings.RESULTS_DIR
        
        # Look for clustering file (.npz)
        clustering_files = list(download_dir.glob(f"{video_id}*.npz")) + \
                          list(results_dir.glob(f"{video_id}*.npz"))
        if not clustering_files:
            clustering_files = list(download_dir.glob(f"*{video_id}*.npz")) + \
                              list(results_dir.glob(f"*{video_id}*.npz"))
        
        # Look for sentiment file (_sentiments.json)
        sentiment_files = list(download_dir.glob(f"{video_id}*_sentiments.json")) + \
                         list(results_dir.glob(f"{video_id}*_sentiments.json"))
        if not sentiment_files:
            sentiment_files = list(download_dir.glob(f"*{video_id}*_sentiments.json")) + \
                             list(results_dir.glob(f"*{video_id}*_sentiments.json"))
        
        if not clustering_files:
            return Response({
                "error": f"Clustering file (.npz) not found for video {video_id}",
                "message": "Run clustering and embedding first to generate the .npz file",
                "video_id": video_id
            }, status=status.HTTP_404_NOT_FOUND)
        
        if not sentiment_files:
            return Response({
                "error": f"Sentiment file (_sentiments.json) not found for video {video_id}",
                "message": "Run sentiment analysis first to generate the _sentiments.json file",
                "video_id": video_id
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Get the first available files
        clustering_file = clustering_files[0]
        sentiment_file = sentiment_files[0]
        
        # Initialize the analysis engine
        api_key = os.getenv("GROQ_API_KEY")
        engine = CommentAnalysisEngine(download_dir=str(download_dir), api_key=api_key)
        
        # Run the analysis
        engine.process_analysis(clustering_file, sentiment_file)
        
        return Response({
            "video_id": video_id,
            "status": "analysis_completed",
            "message": "Analysis completed successfully",
            "clustering_file": str(clustering_file),
            "sentiment_file": str(sentiment_file)
        }, status=status.HTTP_200_OK)
    
    except ImportError as e:
        return Response({
            "error": f"Failed to import analysis engine: {str(e)}",
            "message": "Make sure analysis_engine.py is available in the project root",
            "video_id": video_id
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    except Exception as e:
        return Response({
            "error": str(e),
            "video_id": video_id
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

