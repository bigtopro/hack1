#!/bin/bash

# Script to run both Django API and React frontend

echo "Starting fullstack application..."

# Function to start Django API server
start_django() {
    echo "Starting Django API server..."
    cd django_api
    source ../analysis_env/bin/activate
    python manage.py runserver 8000 &
    DJANGO_PID=$!
    cd ..
    echo "Django API server started with PID $DJANGO_PID"
}

# Function to start React frontend
start_frontend() {
    echo "Starting React frontend..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    echo "React frontend started with PID $FRONTEND_PID"
}

# Start Django API
start_django

# Wait a moment for Django to start
sleep 3

# Start React frontend
start_frontend

echo "Both servers are running!"
echo "Django API: http://localhost:8000"
echo "React Frontend: http://localhost:5173 (or as shown in the frontend terminal)"

# Keep the script running
wait $DJANGO_PID $FRONTEND_PID