"""
URL routing for API endpoints
"""
from django.urls import path
from . import views

urlpatterns = [
    path('extract/', views.extract_comments_view, name='extract-comments'),
    path('comments/', views.list_comments_view, name='list-comments'),
    path('comments/<str:video_id>/', views.get_comments_view, name='get-comments'),
    path('embedding/<str:video_id>/status/', views.check_embedding_status_view, name='check-embedding-status'),
    path('embedding/<str:video_id>/download/', views.download_results_view, name='download-results'),
    path('sentiment/<str:video_id>/', views.sentiment_analysis_view, name='sentiment-analysis'),
    path('dashboard/<str:video_id>/', views.get_analysis_dashboard_view, name='get-analysis-dashboard'),
    path('analyze/<str:video_id>/', views.trigger_analysis_view, name='trigger-analysis'),
]

