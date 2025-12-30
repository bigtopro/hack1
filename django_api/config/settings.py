"""
Django settings for youtube comments API project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in parent directory (hack1)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / '.env')

# BASE_DIR is django_api directory
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-insecure-change-me-in-production')
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# No database needed - using file-based storage
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Paths - django_api is inside hack1, so parent is hack1
# BASE_DIR.parent = hack1 directory
JAVA_PROJECT_DIR = BASE_DIR.parent
COMMENTS_DIR = BASE_DIR.parent / 'comments'

# Google Drive settings
GOOGLE_DRIVE_CREDENTIALS_PATH = os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH', BASE_DIR.parent / 'credentials.json')
GOOGLE_DRIVE_TOKEN_PATH = os.getenv('GOOGLE_DRIVE_TOKEN_PATH', BASE_DIR.parent / 'token.json')
GOOGLE_DRIVE_COMMENTS_FOLDER = os.getenv('GOOGLE_DRIVE_COMMENTS_FOLDER', 'youtubeComments')
GOOGLE_DRIVE_EMBED_FOLDER = os.getenv('GOOGLE_DRIVE_EMBED_FOLDER', 'youtubeComments/embed')

# Local storage for downloaded results
RESULTS_DIR = BASE_DIR.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ],
}

