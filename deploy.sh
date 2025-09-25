#!/bin/bash

# EDIT THIS: Replace with your actual Google Cloud project ID
PROJECT_ID="allocation-api-bhati-2025"

echo "🚀 Deploying Allocation API to Google Cloud Run..."

# Set project
gcloud config set project $PROJECT_ID

# Build container
echo "📦 Building container..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/allocation-api

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy allocation-api \
  --image gcr.io/$PROJECT_ID/allocation-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --set-env-vars SUPABASE_URL=https://nctovqenidcfbbuceuib.supabase.co \
  --set-env-vars SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5jdG92cWVuaWRjZmJidWNldWliIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc2NjA0ODIsImV4cCI6MjA3MzIzNjQ4Mn0.Tse_lidVCbwLfXXzo5nXPmWU5HDSBJZDqfDimMxjf3I \
  --set-env-vars MODEL_PATH=models/allocation_model.pkl

echo "✅ Deployment complete!"
echo ""
echo "🌐 Your API is now live!"
echo "📋 Test your endpoints:"

# Get the service URL
SERVICE_URL=$(gcloud run services describe allocation-api --region=us-central1 --format="value(status.url)")

echo "   Health Check: curl $SERVICE_URL/health"
echo "   Run Allocation: curl -X POST $SERVICE_URL/run-allocation -H 'Content-Type: application/json' -d '{"internship_id": "INT001"}'"
echo "   Get Results: curl $SERVICE_URL/get-results/INT001"
echo ""
echo "🔗 Use this URL in your frontend: $SERVICE_URL"
