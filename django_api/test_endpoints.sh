#!/bin/bash
# Test script for new API endpoints

BASE_URL="http://localhost:8000/api"
VIDEO_ID="rE_530gL0cs"

echo "🧪 Testing Django API Endpoints"
echo "================================"
echo ""

echo "1️⃣  Testing Extract Endpoint (with Drive upload)"
echo "POST $BASE_URL/extract/"
curl -s -X POST "$BASE_URL/extract/" \
  -H "Content-Type: application/json" \
  -d "{\"video_id_or_url\": \"$VIDEO_ID\"}" | python3 -m json.tool
echo ""
echo ""

echo "2️⃣  Testing Check Embedding Status"
echo "GET $BASE_URL/embedding/$VIDEO_ID/status/"
curl -s "$BASE_URL/embedding/$VIDEO_ID/status/" | python3 -m json.tool
echo ""
echo ""

echo "3️⃣  Testing Download Results"
echo "POST $BASE_URL/embedding/$VIDEO_ID/download/"
curl -s -X POST "$BASE_URL/embedding/$VIDEO_ID/download/" | python3 -m json.tool
echo ""
echo ""

echo "4️⃣  Testing Sentiment Analysis"
echo "GET $BASE_URL/sentiment/$VIDEO_ID/"
curl -s "$BASE_URL/sentiment/$VIDEO_ID/" | python3 -m json.tool | head -50
echo ""
echo ""

echo "5️⃣  Testing Sentiment with Emotion Filter"
echo "GET $BASE_URL/sentiment/$VIDEO_ID/?emotion=anger"
curl -s "$BASE_URL/sentiment/$VIDEO_ID/?emotion=anger" | python3 -m json.tool | head -30
echo ""
echo ""

echo "✅ Testing complete!"

