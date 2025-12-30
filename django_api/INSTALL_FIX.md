# Installation Fix for Python 3.13

## Issue
numpy 1.24.3 is not compatible with Python 3.13.7

## Solution

Updated `requirements.txt` to use `numpy>=1.26.0` which is compatible with Python 3.13.

## Installation Steps

```bash
cd django_api

# Install without numpy first (if it fails)
pip3 install Django==4.2.7 djangorestframework==3.14.0 python-dotenv==1.0.0 requests==2.31.0 google-api-python-client==2.108.0 google-auth-httplib2==0.1.1 google-auth-oauthlib==1.1.0

# Then install numpy separately (will get compatible version)
pip3 install "numpy>=1.26.0"

# Or install all at once
pip3 install -r requirements.txt
```

## Alternative: Skip numpy for now

If you don't need sentiment analysis immediately, you can comment out numpy:

```bash
# numpy>=1.26.0  # Comment this out if not needed yet
```

The API will still work for extraction and Drive upload, but sentiment analysis will need numpy when you're ready to use it.

