# Comment Analysis Engine

This project monitors the Downloads folder for clustering (.npz) and sentiment analysis (.json) files, then generates comprehensive insights using LLM-powered analysis.

## Setup

### 1. Create Virtual Environment

Run the setup script to create a virtual environment with all required dependencies:

```bash
./setup.sh
```

This will:
- Create a virtual environment named `analysis_env`
- Install all required packages
- Set up the environment for running the analysis engine

### 2. Configure API Key

Copy the `.env` template and add your Groq API key:

```bash
cp .env .env.local
```

Then edit `.env.local` and add your API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Analysis Engine

### 1. Activate Virtual Environment

```bash
source analysis_env/bin/activate
```

### 2. Run the Analysis Engine

```bash
python analysis_engine.py
```

The engine will start monitoring `/Users/venuvamsi/Downloads` (or your configured directory) for:
- `.npz` files containing clustering data
- `_sentiments.json` files containing sentiment analysis

### 3. Testing with Files

To test the engine:
1. Place a `.npz` clustering file and a `_sentiments.json` file in your Downloads folder
2. The engine will automatically detect them and start the analysis
3. Once complete, a markdown report will be generated in the Downloads folder

## Required File Formats

### Clustering File (.npz)
Must contain:
- `embeddings`: Array of embeddings
- `ids`: Deduplicated comment IDs
- `labels`: K-means cluster labels
- `centroids`: Cluster centroids
- `distances`: Distances to centroids

### Sentiment File (_sentiments.json)
JSON array of objects with:
```json
[
  {
    "id": 1,
    "comment": "Sample comment text",
    "emotion": "joy"
  }
]
```

## Analysis Report Structure

The generated report includes:
1. Overall Viewer Sentiment (volume-based)
2. Key Emotional Drivers
3. Selected Topic Analysis
4. Engagement Diagnostics
5. Actionable Insights
6. Recommendations

## Dashboard Outputs

The system now generates three synchronized outputs:

1. **Full Analysis Report** (`analysis_report_*.md`)
   - Deep narrative analysis
   - Detailed insights and explanations

2. **Dashboard JSON** (`analysis_dashboard_*.json`)
   - Structured, UI-ready data
   - Canonical analytics payload
   - Emotion-first structure with confidence markers
   - Focus on emotional drivers and reasons

3. **Dashboard Markdown** (`analysis_dashboard_*.md`)
   - Human-readable dashboard preview
   - Mirrors UI content
   - For review and QA

All three outputs are generated from the same analysis run, ensuring consistency.

## Key Changes from Previous Versions

- **Emotion-First Analysis**: The system now prioritizes understanding why viewers feel certain emotions rather than focusing on deduplication
- **Cluster-Based Insights**: Analysis focuses on semantic clusters to understand different reasons behind emotions
- **Improved Confidence Scoring**: Confidence levels now incorporate entropy calculations for more accurate assessments
- **Simplified Dashboard**: Dashboard structure is now emotion-centric with clearer organization
- **Better Performance**: Removed deduplication step for faster processing while maintaining insight quality

## Stopping the Engine

Press `Ctrl+C` to stop the monitoring process.