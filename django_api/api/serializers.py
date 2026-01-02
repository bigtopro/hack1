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


class EmbeddingStatusSerializer(serializers.Serializer):
    video_id = serializers.CharField()
    npz_file_exists = serializers.BooleanField()
    npz_file_info = serializers.DictField(required=False, allow_null=True)
    json_summary_exists = serializers.BooleanField()
    json_summary_info = serializers.DictField(required=False, allow_null=True)
    status = serializers.CharField()
    message = serializers.CharField()


class DownloadResultsSerializer(serializers.Serializer):
    video_id = serializers.CharField()
    status = serializers.CharField()
    file_path = serializers.CharField()
    message = serializers.CharField()


class SentimentAnalysisSerializer(serializers.Serializer):
    video_id = serializers.CharField()
    total_comments = serializers.IntegerField()
    emotions = serializers.ListField(child=serializers.CharField())
    comments_by_emotion = serializers.DictField()
    emotion_reasons = serializers.DictField()
    clusters_with_emotions = serializers.DictField()
    has_sentiment_data = serializers.BooleanField()


class CommentWithMetadataSerializer(serializers.Serializer):
    id = serializers.CharField()
    text = serializers.CharField()
    publishedAt = serializers.CharField(required=False, allow_null=True)
    likeCount = serializers.IntegerField()
    parentId = serializers.CharField(required=False, allow_null=True)

