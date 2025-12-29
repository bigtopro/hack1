"""
Serializers for API requests/responses
"""
from rest_framework import serializers


class ExtractCommentsSerializer(serializers.Serializer):
    video_id_or_url = serializers.CharField(
        required=True, 
        help_text="YouTube video ID or URL"
    )


class CommentFileSerializer(serializers.Serializer):
    video_id = serializers.CharField()
    file_name = serializers.CharField()
    comment_count = serializers.IntegerField()
    file_path = serializers.CharField()

