"""
URL routing for API endpoints
"""
from django.urls import path
from . import views

urlpatterns = [
    path('extract/', views.extract_comments_view, name='extract-comments'),
    path('comments/', views.list_comments_view, name='list-comments'),
    path('comments/<str:video_id>/', views.get_comments_view, name='get-comments'),
]

