# Comment Analysis Engine

This project monitors the Downloads folder for clustering (.npz) and sentiment analysis (.json) files, then generates comprehensive insights using LLM-powered analysis.

## Project Structure

The project is organized into the following directories:

- `analysis/` - Core analysis scripts and engines
- `docs/` - Documentation and specification files
- `tests/` - Test scripts and utilities
- `scripts/` - Shell scripts for setup and execution
- `django_api/` - Django backend API
- `frontend/` - React frontend application
- `src/` - Java source code for YouTube comment extraction
- `comments/` - Generated comment files
- `analysis results/` - Analysis output files

## Setup

### 1. Create Virtual Environment

Run the setup script to create a virtual environment with all required dependencies:

```bash
./scripts/setup.sh
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
python analysis/analysis_engine.py
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

## Frontend Integration

The project now includes a modern React-based frontend for visualizing the analysis results. The frontend is located in the `frontend/` directory.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`.

### Frontend Features

- Modern dark-themed UI with glassmorphism effects
- Real-time visualization of comment analysis results
- Interactive charts showing cluster distribution
- Discussion clusters organized by themes
- Common ideas and themes extraction
- Opportunities and signals identification

### Backend Integration

To connect the frontend to the backend analysis engine:

1. The backend generates JSON dashboard files in the Downloads directory
2. The frontend needs to be updated to fetch these files from the appropriate endpoint
3. Modify the `handleAnalyze` function in `frontend/src/App.tsx` to make API calls to the backend

### Building for Production

To create a production build of the frontend:
```bash
npm run build
```

## Running the Fullstack Application

To run both the Django API backend and React frontend together:

1. Make sure you have both the Django API and frontend dependencies installed:
   ```bash
   # For Django API (in django_api directory)
   pip install -r requirements.txt

   # For React frontend (in frontend directory)
   npm install
   ```

2. Run the fullstack application:
   ```bash
   ./scripts/run_fullstack.sh
   ```

This will start both servers:
- Django API at `http://localhost:8000`
- React frontend at `http://localhost:5173` (or as shown in the terminal)

## Stopping the Engine

Press `Ctrl+C` to stop the monitoring process.