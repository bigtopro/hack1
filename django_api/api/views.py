"""
API views for YouTube comments extraction
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from pathlib import Path
from .utils import (
    extract_video_id_from_url,
    get_comments_from_file,
    list_available_comments,
    extract_comments
)
from .serializers import (
    ExtractCommentsSerializer, 
    CommentFileSerializer
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
        return Response({
            'video_id': video_id,
            'status': 'success',
            'comment_count': result['comment_count'],
            'file_path': result['file_path'],
            'message': f'Successfully extracted {result["comment_count"]} comments'
        }, status=status.HTTP_200_OK)
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

