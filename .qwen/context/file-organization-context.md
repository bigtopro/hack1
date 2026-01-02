# File Organization Context Analysis

## Feature Description

The codebase contains a comprehensive YouTube comment analysis system that processes comments through multiple stages: embedding generation, sentiment analysis, clustering, and insight generation. The current structure has many files in the root directory that need to be organized into logical folders based on their functionality to improve maintainability and reduce clutter.

## Architectural Overview

The system consists of several interconnected components:

1. **Comment Embedding Pipeline**: Generates semantic embeddings from YouTube comments using transformer models
2. **Sentiment Analysis Pipeline**: Classifies emotions in comments using multilingual models
3. **Clustering Engine**: Groups similar comments using K-means clustering with silhouette-based optimization
4. **Analysis Engine**: Combines clustering and sentiment data to generate insights using LLMs
5. **Frontend Application**: React-based UI for visualizing analysis results
6. **Django API**: Backend API for serving analysis results
7. **Testing & Utilities**: Various test files and utility scripts

The data flow follows this pattern: Raw comments → Embeddings → Sentiment Classification → Clustering → Analysis → Dashboard/Reports

## Relevant External APIs, Models, or Schemas Involved

### Models
- `intfloat/multilingual-e5-small` for comment embeddings
- `AnasAlokla/multilingual_go_emotions` for sentiment classification
- Llama 3.1 8B Instant for analysis via Groq API

### File Formats
- `.npz` files containing embeddings, IDs, labels, centroids, and distances
- JSON files for sentiment data with comment ID, text, and emotion
- Markdown reports for analysis results
- JSON dashboard files for frontend consumption

### APIs
- Groq API for LLM analysis
- Google Drive API for file monitoring (in Colab environment)
- YouTube Data API v3 for comment scraping (Java component)

## Ranked List of Files

### Must Change
- `analysis_engine.py` → Move to `src/analysis/`
- `auto_embed_comments_final.py` → Move to `src/embedding/`
- `sentiment_classifier.py` → Move to `src/sentiment/`
- `pom.xml` → Move to `src/youtube-scraper/`
- `videoList.txt` → Move to `config/`

### Likely Impacted
- `run_analysis.sh` → Update paths and move to `scripts/`
- `run_fullstack.sh` → Update paths and move to `scripts/`
- `setup.sh` → Update paths and move to `scripts/`
- `requirements.txt` → Move to root or `backend/`
- `README.md` → Update documentation to reflect new structure

### Review for Assumptions
- All test files (`test_*.py`) → Move to `tests/`
- All specification files (`*Spec.md`) → Move to `docs/specs/`
- `EMBED_COMMENTS_NOTEBOOK_EXPLAINED.md` → Move to `docs/`
- `analysis results/` → Move to `data/output/`
- `comments/` → Move to `data/input/`
- `django_api/` → Already in appropriate location
- `frontend/` → Already in appropriate location
- `analysis_env/` → Move to `env/` or keep at root
- `src/` → Already in appropriate location but may need reorganization

## Data Schema Implications

The file reorganization will not change any data schemas or structures. All internal data contracts remain the same:
- NPZ files continue to contain embeddings, IDs, labels, centroids, and distances
- Sentiment JSON files maintain the same structure with id, comment, and emotion fields
- Dashboard JSON files maintain the same structure for frontend consumption

The only change is the physical location of files, which may require updates to:
- File paths in shell scripts
- Relative imports in Python files
- Documentation references

## Downstream Risks and Consistency Concerns

1. **Shell Script Updates**: All shell scripts will need updated paths to executables
2. **Python Import Paths**: Relative imports may need adjustment if directory structure changes significantly
3. **Configuration Files**: Environment variables and configuration files may reference specific paths
4. **Documentation**: README and other documentation files reference specific file locations
5. **Git Tracking**: Some directories like analysis results may need to be added to .gitignore

## Open Questions, Unknowns, or Assumptions

1. **Virtual Environment Location**: Should `analysis_env/` be moved to a different location or kept at root?
2. **Data Directory Structure**: Should input/output data be organized differently beyond just moving directories?
3. **Cross-Platform Compatibility**: Will path changes affect cross-platform compatibility?
4. **Docker/Deployment**: Are there deployment scripts or Docker configurations that assume current structure?
5. **IDE/Editor Configs**: Do VSCode settings or other editor configurations reference specific file paths?

The analysis assumes that the primary goal is to organize files by functionality while maintaining all existing behavior and interfaces. No functional changes are intended, only structural reorganization.