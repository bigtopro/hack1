import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
from pathlib import Path
import time
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import logging

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without loading .env
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required libraries for LLM integration
try:
    from groq import Groq
except ImportError as e:
    logger.warning(
        "Groq library not installed. "
        "Run: pip install groq"
    )
    Groq = None

# Rate limiting constants
MAX_CALLS_PER_MIN = 20  # leave buffer
MIN_DELAY_SEC = 3.2  # 60/20 = 3 sec, +0.2 buffer

class CommentAnalysisEngine:
    """
    Comment Insight & Audience Analysis Engine
    """
    
    def __init__(self, download_dir: str = "/Users/venuvamsi/Downloads",
                 api_key: Optional[str] = None):
        self.download_dir = Path(download_dir)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")  # Changed from OPENROUTER_API_KEY
        self.processed_files = set()

        # Rate limiting
        self._last_llm_call = 0.0

        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Define system prompts
        self.global_system_prompt = """You are an analytical assistant specializing in audience psychology and content analysis.

You must:
- Base your analysis ONLY on the data provided.
- Never invent statistics, percentages, or trends.
- Never assume time-based changes unless explicitly stated.
- Focus on explaining causes, patterns, and implications.
- Be concise, structured, and concrete.
- Avoid generic advice and clichés.

You are not a chatbot.
You are producing insights for creators and analysts who will act on your output."""
        
        self.sentiment_summary_prompt = """You are analyzing aggregated viewer sentiment data.

Your task:
- Interpret the overall emotional state of the audience.
- Compare raw sentiment percentages with deduplicated (semantic) sentiment percentages.
- Explain what this difference implies about emotional intensity vs emotional diversity.

Rules:
- Do NOT restate the numbers verbatim.
- Do NOT speculate beyond the data.
- Focus on what dominates, what is niche, and what is structurally important.

Output format:
- 1 short paragraph: overall emotional tone
- 3–5 bullet points: key takeaways"""
        
        self.emotion_cluster_reason_prompt = """You are analyzing viewer comments that all express the same emotion: {EMOTION}.

The comments are grouped into distinct semantic clusters.
Each cluster represents a different underlying reason for this emotion.

Your task:
- Identify the main reason behind each cluster.
- Explain how these reasons differ from one another.
- Focus on causes, not solutions.

Rules:
- Do NOT merge clusters unless they are clearly the same reason.
- Do NOT generalize across clusters.
- Treat each cluster as a separate emotional driver.

Output format:
- Section title: "Reasons for {EMOTION}"
- For each cluster:
  - Short label for the reason
  - 2–3 sentence explanation grounded in the comments"""
        
        self.cluster_sentiment_mix_prompt = """You are analyzing a single discussion topic derived from viewer comments.

This topic has:
- A known emotional distribution
- A measurable level of emotional consistency or polarization

Your task:
- Explain what this topic is about.
- Describe how viewers emotionally respond to it.
- Interpret whether the topic is emotionally consistent or emotionally mixed.

Rules:
- Explicitly connect emotions to aspects of the topic.
- If emotions are mixed, explain why.
- Do NOT make recommendations.

Output format:
- Topic summary (2–3 sentences)
- Emotional interpretation (2–3 sentences)"""
        
        self.engagement_diagnostics_prompt = """You are evaluating viewer engagement quality and risk signals.

Your task:
- Identify which topics are emotionally stable.
- Identify which topics are emotionally polarizing.
- Highlight potential engagement risks or strengths.

Rules:
- Do NOT assume growth or decline.
- Focus on emotional structure, not popularity.
- Avoid alarmist language.

Output format:
- 3–5 bullet points:
  - What is emotionally stable
  - What is emotionally polarizing
  - Why this matters for engagement"""
        
        self.actionable_insights_prompt = """You are synthesizing insights from a full viewer comment analysis.

Your task:
- Translate emotional and semantic patterns into actionable insights.
- Focus on content strategy, communication, and audience understanding.

Rules:
- Base every recommendation on earlier analysis.
- Avoid generic advice.
- Be specific and concrete.

Output format:
- 3 recommended actions
- 2 things that should not be changed
- 1 high-risk / high-reward experiment idea"""

    def _truncate_prompt(self, text: str, max_chars: int = 3000) -> str:
        """
        Conservative truncation.
        ~4 chars ≈ 1 token → 3000 chars ≈ 750 tokens
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[TRUNCATED]"

    def _initialize_llm(self):
        """Initialize the LLM with Groq"""
        if not self.api_key:
            logger.warning("No GROQ_API_KEY provided. LLM disabled.")
            return None

        if Groq is None:
            logger.warning("Groq is not available. LLM functionality will be limited.")
            return None

        try:
            return Groq(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            return None

    def _call_llm(self, user_prompt: str) -> str:
        """Helper method to call LLM with system and user messages"""
        if not self.llm:
            return "LLM not available"

        # Apply truncation before calling LLM
        user_prompt = self._truncate_prompt(user_prompt)

        # Rate limiting
        now = time.time()
        elapsed = now - self._last_llm_call
        if elapsed < MIN_DELAY_SEC:
            time.sleep(MIN_DELAY_SEC - elapsed)

        # Retry with exponential backoff
        for attempt in range(3):
            try:
                response = self.llm.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": self.global_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300  # 🔒 HARD LIMIT
                )

                # Update last call time
                self._last_llm_call = time.time()

                return response.choices[0].message.content
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM call failed (attempt {attempt+1}), retrying in {wait}s: {e}")
                if attempt < 2:  # Don't sleep after the last attempt
                    time.sleep(wait)

        return "LLM failed after retries"

    def monitor_and_analyze(self):
        """Monitor the download directory and start analysis when both files are present"""
        logger.info(f"Starting to monitor {self.download_dir} for analysis files...")

        while True:
            # Look for .npz and _sentiments.json files
            npz_files = list(self.download_dir.glob("*.npz"))
            sentiment_files = list(self.download_dir.glob("*_sentiments.json"))

            # Filter out files that are currently being written to (by checking if they're locked or recently modified)
            ready_npz_files = [f for f in npz_files if self._is_file_ready(f)]
            ready_sentiment_files = [f for f in sentiment_files if self._is_file_ready(f)]

            if ready_npz_files and ready_sentiment_files:
                # Get the most recent files
                latest_npz = max(ready_npz_files, key=lambda x: x.stat().st_mtime)
                latest_sentiment = max(ready_sentiment_files, key=lambda x: x.stat().st_mtime)

                # Create a unique identifier for this file pair to avoid reprocessing
                file_pair_id = self._generate_file_pair_id(latest_npz, latest_sentiment)

                # Check if we've already processed this pair using a marker file
                marker_file = self.download_dir / f".{file_pair_id}_processed"
                if marker_file.exists():
                    time.sleep(10)  # Wait before checking again
                    continue

                logger.info(f"Found new file pair ready for analysis: {latest_npz.name} and {latest_sentiment.name}")

                # Process the files
                try:
                    self.process_analysis(latest_npz, latest_sentiment)

                    # Create marker file to indicate processing is complete
                    marker_file.touch()
                    logger.info(f"Analysis completed for {file_pair_id}, marker file created: {marker_file}")
                except Exception as e:
                    logger.error(f"Error processing files: {e}")

            time.sleep(10)  # Wait before checking again

    def _is_file_ready(self, file_path: Path) -> bool:
        """Check if a file is ready for processing (not being written to)"""
        try:
            # Check if file size has been stable for a short period
            initial_size = file_path.stat().st_size
            time.sleep(0.5)  # Brief pause
            final_size = file_path.stat().st_size
            return initial_size == final_size
        except (OSError, FileNotFoundError):
            return False

    def _generate_file_pair_id(self, npz_path: Path, sentiment_path: Path) -> str:
        """Generate a unique ID for a file pair based on content hashes and timestamps"""
        npz_hash = self._get_file_hash(npz_path)
        sentiment_hash = self._get_file_hash(sentiment_path)
        npz_time = str(int(npz_path.stat().st_mtime))
        sentiment_time = str(int(sentiment_path.stat().st_mtime))

        # Combine hashes and timestamps to create unique identifier
        combined = f"{npz_hash[:8]}_{sentiment_hash[:8]}_{npz_time}_{sentiment_time}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file content"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            # If we can't read the file, return a hash of the path and timestamp
            return hashlib.md5(f"{file_path}_{file_path.stat().st_mtime}".encode()).hexdigest()

    def process_analysis(self, npz_path: Path, sentiment_path: Path):
        """Process the clustering and sentiment files to generate insights"""
        logger.info("Loading clustering and sentiment data...")

        # Track the current sentiment file for video ID extraction
        self._current_sentiment_file = sentiment_path

        # Load clustering data
        cluster_data = np.load(npz_path)
        embeddings = cluster_data['embeddings']
        ids = cluster_data['ids']  # deduplicated comment IDs
        labels = cluster_data['labels']  # k-means cluster id per embedding
        centroids = cluster_data['centroids']  # cluster centroids
        distances = cluster_data['distances']  # distance of each embedding to its centroid

        # Load sentiment data (contains all original comments, may include duplicates)
        with open(sentiment_path, 'r') as f:
            sentiment_data = json.load(f)

        # Step 6.1: Create sentiment lookup table from sentiment JSON
        sentiment_by_id = {item['id']: item['emotion'] for item in sentiment_data}

        # Step 6.2: Perform deduplication-aware join
        # Only IDs present in .npz["ids"] are used for cluster-based analysis
        valid_ids = set(ids)  # These are the deduplicated semantic comment IDs

        # Create filtered sentiment data that only includes deduplicated comments
        filtered_sentiment_data = []
        for item in sentiment_data:
            if item['id'] in valid_ids:
                filtered_sentiment_data.append(item)

        # Create mapping from cluster ID to sentiment (for semantic analysis)
        cluster_sentiment_map = defaultdict(list)
        cluster_comment_map = defaultdict(list)

        # Build an index for O(1) ID lookup to avoid O(N²) behavior
        # Store as instance variable to fix scope issue
        self.id_to_index = {id_: idx for idx, id_ in enumerate(ids)}
        # Also store distances as instance variable to fix scope issue
        self.distances = distances

        for item in filtered_sentiment_data:
            comment_id = item['id']
            # Find the index of this comment ID in the clustering data using the pre-built index
            cluster_idx = self.id_to_index.get(comment_id)
            if cluster_idx is not None:
                cluster_label = labels[cluster_idx]

                # Map cluster to sentiments and comments
                cluster_sentiment_map[cluster_label].append(item['emotion'])
                cluster_comment_map[cluster_label].append(item)

        # Step 6.3: Compute dual sentiment views
        # Raw Viewer Sentiment (from all sentiment JSON entries)
        self.raw_sentiment_distribution = self._compute_raw_sentiment_distribution(sentiment_data)

        # Semantic Sentiment (from deduplicated IDs only)
        self.semantic_sentiment_distribution = self._compute_semantic_sentiment_distribution(filtered_sentiment_data)

        logger.info("Starting analysis...")

        # Step 1: Global Sentiment Distribution
        # Compute percentages for both raw and semantic views
        raw_sentiment_dist = self.raw_sentiment_distribution
        semantic_sentiment_dist = self.semantic_sentiment_distribution

        # Generate LLM summary for global sentiment
        global_sentiment_summary = self._generate_global_sentiment_summary(
            raw_sentiment_dist, semantic_sentiment_dist
        )

        # Step 2: Emotion -> Cluster -> Reason Analysis
        emotion_cluster_analysis = self._analyze_emotion_clusters(
            cluster_sentiment_map, cluster_comment_map, labels, centroids
        )

        # Step 3: Cluster -> Sentiment Mix Analysis
        cluster_sentiment_analysis = self._analyze_cluster_sentiment_mix(
            cluster_sentiment_map, cluster_comment_map, centroids
        )

        # Generate engagement diagnostics
        engagement_diagnostics = self._generate_engagement_diagnostics(
            cluster_sentiment_map, cluster_comment_map
        )

        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(
            global_sentiment_summary, emotion_cluster_analysis,
            cluster_sentiment_analysis, engagement_diagnostics
        )

        # Create final report
        report = self._create_final_report(
            global_sentiment_summary,
            emotion_cluster_analysis,
            cluster_sentiment_analysis,
            engagement_diagnostics,
            actionable_insights
        )

        # Create and save dashboard JSON and markdown
        dashboard_json = self._build_dashboard_json(
            raw_sentiment_dist, semantic_sentiment_dist,
            cluster_sentiment_analysis, emotion_cluster_analysis,
            engagement_diagnostics, actionable_insights
        )

        dashboard_json_path = self.download_dir / f"analysis_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(dashboard_json_path, 'w') as f:
            json.dump(dashboard_json, f, indent=2)

        dashboard_md = self._render_dashboard_md(dashboard_json)
        dashboard_md_path = self.download_dir / f"analysis_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(dashboard_md_path, 'w') as f:
            f.write(dashboard_md)

        # Save report
        report_path = self.download_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Analysis report saved to {report_path}")
        logger.info(f"Dashboard JSON saved to {dashboard_json_path}")
        logger.info(f"Dashboard markdown saved to {dashboard_md_path}")

    def _compute_raw_sentiment_distribution(self, sentiment_data: List[Dict]) -> Dict[str, float]:
        """Compute raw sentiment distribution from all sentiment data"""
        emotions = [item['emotion'] for item in sentiment_data]
        total = len(emotions)
        emotion_counts = Counter(emotions)
        return {emotion: count/total for emotion, count in emotion_counts.items()}

    def _compute_semantic_sentiment_distribution(self, filtered_sentiment_data: List[Dict]) -> Dict[str, float]:
        """Compute semantic sentiment distribution from deduplicated data"""
        emotions = [item['emotion'] for item in filtered_sentiment_data]
        total = len(emotions)
        emotion_counts = Counter(emotions)
        return {emotion: count/total for emotion, count in emotion_counts.items()}

    def _generate_global_sentiment_summary(self, raw_dist: Dict[str, float],
                                         semantic_dist: Dict[str, float]) -> str:
        """Generate LLM summary of global sentiment"""
        if not self.llm:
            return "LLM not available for global sentiment summary"

        # Format the distributions for the prompt
        raw_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in raw_dist.items()])
        semantic_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in semantic_dist.items()])

        prompt = f"""
Raw sentiment distribution: {raw_str}
Semantic (deduplicated) sentiment distribution: {semantic_str}

{self.sentiment_summary_prompt}
"""

        try:
            response = self._call_llm(prompt)
            return str(response)
        except Exception as e:
            logger.error(f"Error generating global sentiment summary: {e}")
            return f"Error generating summary: {e}"

    def _analyze_emotion_clusters(self, cluster_sentiment_map: Dict,
                                cluster_comment_map: Dict,
                                labels: np.ndarray,
                                centroids: np.ndarray) -> Dict:  # Changed distances to centroids
        """Analyze each emotion by cluster to understand reasons"""
        if not self.llm:
            return {"error": "LLM not available for emotion cluster analysis"}

        results = {}

        # Get unique emotions
        all_emotions = set()
        for emotions in cluster_sentiment_map.values():
            all_emotions.update(emotions)

        for emotion in all_emotions:
            # Find clusters that have this emotion
            emotion_clusters = {}
            for cluster_id, emotions in cluster_sentiment_map.items():
                emotion_count = emotions.count(emotion)
                if emotion_count > 0:
                    emotion_clusters[cluster_id] = emotion_count

            if not emotion_clusters:
                continue

            # For each cluster with this emotion, get representative comments
            cluster_analysis = {}
            for cluster_id, count in emotion_clusters.items():
                # Get comments for this cluster that have this emotion
                cluster_items_with_distances = []
                for item in cluster_comment_map.get(cluster_id, []):
                    if item['emotion'] == emotion:
                        # Find the distance for this specific comment
                        comment_id = item['id']
                        cluster_idx = self.id_to_index.get(comment_id)  # Use instance variable
                        if cluster_idx is not None:
                            distance = self.distances[cluster_idx]  # Use instance variable
                            cluster_items_with_distances.append((item['comment'], distance))

                # Sort by ascending distance to centroid (closest first) to prioritize core ideas
                cluster_items_with_distances.sort(key=lambda x: x[1])

                # Take up to 10 comments, prioritizing those closer to centroid
                cluster_comments = [comment for comment, _ in cluster_items_with_distances[:10]]

                cluster_analysis[cluster_id] = {
                    'size': count,
                    'comments': cluster_comments
                }

            # Generate LLM analysis for this emotion
            # Limit to top 3 clusters to control token usage
            limited_cluster_analysis = dict(list(cluster_analysis.items())[:3])

            prompt = f"""
Emotion: {emotion}
Clusters with this emotion:
"""
            for cluster_id, data in limited_cluster_analysis.items():
                prompt += f"\nCluster {cluster_id} (size: {data['size']}):\n"
                for i, comment in enumerate(data['comments'][:5]):  # Show first 5 comments
                    prompt += f"  - {comment}\n"

            prompt += f"\n{self.emotion_cluster_reason_prompt.format(EMOTION=emotion)}"

            try:
                response = self._call_llm(prompt)  # Use new calling pattern
                results[emotion] = str(response)
            except Exception as e:
                logger.error(f"Error analyzing emotion {emotion}: {e}")
                results[emotion] = f"Error analyzing emotion {emotion}: {e}"

        return results

    def _analyze_cluster_sentiment_mix(self, cluster_sentiment_map: Dict,
                                     cluster_comment_map: Dict,
                                     centroids: np.ndarray) -> Dict:
        """Analyze sentiment mix within each cluster"""
        if not self.llm:
            return {"error": "LLM not available for cluster sentiment analysis"}

        results = {}

        for cluster_id, emotions in cluster_sentiment_map.items():
            # Calculate sentiment distribution for this cluster
            total_emotions = len(emotions)
            emotion_counts = Counter(emotions)
            emotion_dist = {emotion: count/total_emotions for emotion, count in emotion_counts.items()}

            # Calculate entropy (measure of emotional consistency)
            entropy = self._calculate_entropy(emotion_dist.values())

            # Get sample comments for this cluster with distance information
            cluster_items_with_distances = []
            for item in cluster_comment_map.get(cluster_id, []):
                comment_id = item['id']
                cluster_idx = self.id_to_index.get(comment_id)  # Use instance variable
                if cluster_idx is not None:
                    distance = self.distances[cluster_idx]  # Use instance variable
                    cluster_items_with_distances.append((item['comment'], distance))

            # Sort by ascending distance to centroid (closest first) to prioritize core ideas
            cluster_items_with_distances.sort(key=lambda x: x[1])

            # Take first 5 closest comments as samples
            sample_comments = [comment for comment, _ in cluster_items_with_distances[:5]]

            # Prepare prompt for LLM
            # Format emotion distribution as percentages, not raw dict
            emotion_dist_str = "\n".join([f"- {emotion}: {percentage*100:.0f}%" for emotion, percentage in emotion_dist.items()])

            prompt = f"""
Cluster ID: {cluster_id}
Emotion distribution:
{emotion_dist_str}
Sentiment entropy (0=consistent, higher=polarizing): {entropy:.2f}
Sample comments:
"""
            for comment in sample_comments:
                prompt += f"  - {comment}\n"

            prompt += f"\n{self.cluster_sentiment_mix_prompt}"

            try:
                response = self._call_llm(prompt)  # Use new calling pattern
                results[cluster_id] = {
                    'summary': str(response),
                    'emotion_distribution': emotion_dist,
                    'entropy': entropy
                }
            except Exception as e:
                logger.error(f"Error analyzing cluster {cluster_id}: {e}")
                results[cluster_id] = {
                    'summary': f"Error analyzing cluster {cluster_id}: {e}",
                    'emotion_distribution': emotion_dist,
                    'entropy': entropy
                }

        return results

    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate entropy of a probability distribution"""
        entropy = 0.0
        for p in probabilities:
            if p > 0:  # Avoid log(0)
                entropy -= p * math.log2(p)
        return entropy

    def _generate_engagement_diagnostics(self, cluster_sentiment_map: Dict,
                                       cluster_comment_map: Dict) -> str:
        """Generate engagement and risk diagnostics"""
        if not self.llm:
            return "LLM not available for engagement diagnostics"

        # Calculate entropy for each cluster to identify polarizing topics
        cluster_entropies = {}
        for cluster_id, emotions in cluster_sentiment_map.items():
            total_emotions = len(emotions)
            emotion_counts = Counter(emotions)
            emotion_dist = {emotion: count/total_emotions for emotion, count in emotion_counts.items()}
            entropy = self._calculate_entropy(emotion_dist.values())
            cluster_entropies[cluster_id] = entropy

        # Identify stable and polarizing clusters using relative thresholds based on distribution
        if cluster_entropies:
            entropies = list(cluster_entropies.values())
            entropies_sorted = sorted(entropies)

            # Define thresholds based on distribution: bottom 30% = stable, top 30% = polarizing
            stable_threshold_idx = max(0, int(len(entropies_sorted) * 0.3))
            polarizing_threshold_idx = max(0, int(len(entropies_sorted) * 0.7))

            stable_threshold = entropies_sorted[stable_threshold_idx] if entropies_sorted else 0
            polarizing_threshold = entropies_sorted[polarizing_threshold_idx] if entropies_sorted else 0

            stable_clusters = {cid: entropy for cid, entropy in cluster_entropies.items() if entropy <= stable_threshold}
            polarizing_clusters = {cid: entropy for cid, entropy in cluster_entropies.items() if entropy >= polarizing_threshold}
        else:
            stable_clusters = {}
            polarizing_clusters = {}

        # Prepare prompt for LLM
        prompt = f"""
Stable clusters (emotionally consistent topics):
{list(stable_clusters.keys()) if stable_clusters else 'None'}

Polarizing clusters (emotionally diverse topics):
{list(polarizing_clusters.keys()) if polarizing_clusters else 'None'}

{self.engagement_diagnostics_prompt}
"""

        try:
            response = self._call_llm(prompt)  # Use new calling pattern
            return str(response)
        except Exception as e:
            logger.error(f"Error generating engagement diagnostics: {e}")
            return f"Error generating diagnostics: {e}"

    def _generate_actionable_insights(self, global_summary: str,
                                    emotion_analysis: Dict,
                                    cluster_analysis: Dict,
                                    engagement_diagnostics: str) -> str:
        """Generate actionable insights from all analysis"""
        if not self.llm:
            return "LLM not available for actionable insights"

        # Create structured summaries instead of raw JSON to reduce token usage
        # Limit to top 3 emotions and top 5 clusters to control token usage
        emotion_summary = ""
        for emotion, analysis in list(emotion_analysis.items())[:3]:
            emotion_summary += f"- {emotion}: {analysis[:300]}...\n"  # Truncate individual analyses

        cluster_summary = ""
        for cluster_id, analysis in list(cluster_analysis.items())[:5]:
            cluster_summary += f"- Cluster {cluster_id}: {analysis.get('summary', '')[:200]}...\n"

        prompt = f"""
Global Sentiment Summary:
{global_summary}

Key Emotional Drivers Summary:
{emotion_summary}

Topic-Level Emotional Impact Summary:
{cluster_summary}

Engagement Diagnostics:
{engagement_diagnostics}

{self.actionable_insights_prompt}
"""

        try:
            response = self._call_llm(prompt)  # Use new calling pattern
            return str(response)
        except Exception as e:
            logger.error(f"Error generating actionable insights: {e}")
            return f"Error generating insights: {e}"

    def _create_final_report(self, global_summary: str, 
                           emotion_analysis: Dict, 
                           cluster_analysis: Dict, 
                           engagement_diagnostics: str, 
                           actionable_insights: str) -> str:
        """Create the final analysis report in markdown format"""
        report = f"""# Comment Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## About This Analysis
This report presents two key perspectives on viewer sentiment:
- **Raw Sentiment**: Based on all comments, reflecting emotional intensity/volume
- **Semantic Sentiment**: Based on deduplicated comments, reflecting emotional diversity of reasons

## 1. Overall Viewer Sentiment
{global_summary}

## 2. Key Emotional Drivers
"""
        for emotion, analysis in emotion_analysis.items():
            report += f"""
### Reasons for {emotion.capitalize()}
{analysis}
"""

        report += f"""
## 3. Topic-Level Emotional Impact
"""
        for cluster_id, analysis in cluster_analysis.items():
            report += f"""
### Cluster {cluster_id}
{analysis['summary']}
"""

        report += f"""
## 4. Engagement Diagnostics
{engagement_diagnostics}

## 5. Actionable Insights
{actionable_insights}

## 6. What to Do Next
Based on the analysis above, here are the recommended next steps:
- Review the most emotionally polarizing topics to understand potential risks
- Focus content strategy on topics that generate positive emotions
- Address concerns raised in negative emotion clusters
"""
        return report

    def _build_dashboard_json(self, raw_sentiment_dist, semantic_sentiment_dist,
                             cluster_sentiment_analysis, emotion_cluster_analysis,
                             engagement_diagnostics, actionable_insights):
        """Build the dashboard JSON from analysis results"""
        # Extract video ID from the file name
        video_id = "unknown"
        if hasattr(self, '_current_sentiment_file'):
            import re
            match = re.search(r'([A-Za-z0-9_-]+)_comments_sentiments\.json', self._current_sentiment_file.name)
            if match:
                video_id = match.group(1)

        # Calculate total comments
        total_comments = sum(semantic_sentiment_dist.values()) if semantic_sentiment_dist else 0
        unique_comments = len(self.id_to_index) if hasattr(self, 'id_to_index') else 0

        # Build cluster distribution
        cluster_distribution = []
        for cluster_id, analysis in cluster_sentiment_analysis.items():
            cluster_info = analysis.get('emotion_distribution', {})
            if cluster_info:
                cluster_size = sum(cluster_info.values())
                dominant_sentiment = max(cluster_info, key=cluster_info.get) if cluster_info else "unknown"
                confidence = self._confidence_label(cluster_size, analysis.get('entropy', 0))

                cluster_distribution.append({
                    "cluster_id": int(cluster_id),
                    "label": f"Cluster {cluster_id}",
                    "comment_count": int(cluster_size),
                    "percentage": round(cluster_size * 100, 2),
                    "dominant_sentiment": dominant_sentiment,
                    "confidence": confidence
                })

        # Build sentiment overview
        sentiment_overview = {
            "distribution": {k: int(v * 100) for k, v in raw_sentiment_dist.items()},
            "dominant_sentiment": max(raw_sentiment_dist, key=raw_sentiment_dist.get) if raw_sentiment_dist else "unknown",
            "sentiment_entropy": self._calculate_entropy(list(raw_sentiment_dist.values()))
        }

        # Build clusters data
        clusters = []
        for cluster_id, analysis in cluster_sentiment_analysis.items():
            cluster_info = analysis.get('emotion_distribution', {})
            entropy = analysis.get('entropy', 0)
            cluster_size = sum(cluster_info.values()) if cluster_info else 0
            confidence = self._confidence_label(cluster_size, entropy)

            # Get sample comments for this cluster
            sample_comments = self._get_sample_comments_for_cluster(cluster_id)

            clusters.append({
                "cluster_id": int(cluster_id),
                "label": f"Cluster {cluster_id}",
                "comment_count": int(cluster_size),
                "percentage": round(cluster_size * 100, 2),
                "sentiment_distribution": {k: int(v * 100) for k, v in cluster_info.items()},
                "entropy": round(entropy, 2),
                "confidence": confidence,
                "summary": analysis.get('summary', ''),
                "sample_comments": sample_comments
            })

        # Build themes from emotion cluster analysis
        themes = []
        for emotion, analysis in emotion_cluster_analysis.items():
            themes.append({
                "theme": emotion,
                "why": analysis[:200],  # Truncate for brevity
                "implication": "Analysis of emotional drivers",
                "recommended_action": f"Address {emotion} related feedback",
                "confidence": "medium"
            })

        # Build opportunities and risks from engagement diagnostics and actionable insights
        opportunities = []
        risks = []

        # Parse actionable insights for opportunities and risks
        if "3 recommended actions" in actionable_insights:
            lines = actionable_insights.split('\n')
            for line in lines:
                if line.strip().startswith('- ') and 'opportunity' in line.lower():
                    opportunities.append({
                        "type": "content_opportunity",
                        "signal": line.strip('- '),
                        "action": "Implement recommendation",
                        "confidence": "medium"
                    })

        if "potential risks" in engagement_diagnostics.lower():
            risks.append({
                "type": "engagement_risk",
                "signal": engagement_diagnostics[:200],
                "impact": "Reduced engagement",
                "confidence": "medium"
            })

        dashboard_json = {
            "meta": {
                "video_id": video_id,
                "total_comments": int(total_comments),
                "unique_comments": unique_comments,
                "analysis_timestamp": datetime.now().isoformat() + "Z",
                "model": "llama-3.1-8b-instant"
            },
            "audience_overview": {
                "cluster_distribution": cluster_distribution
            },
            "sentiment_overview": sentiment_overview,
            "clusters": clusters,
            "themes": themes,
            "opportunities": opportunities,
            "risks": risks
        }

        return dashboard_json

    def _confidence_label(self, cluster_size, entropy):
        """Determine confidence level based on cluster size and entropy"""
        if cluster_size >= 250 and entropy < 1.0:
            return "high"
        if cluster_size >= 120 and entropy < 1.5:
            return "medium"
        return "low"

    def _get_sample_comments_for_cluster(self, cluster_id):
        """Get sample comments for a cluster with core and edge cases"""
        if not hasattr(self, 'cluster_comment_map'):
            return {"centroid": [], "edge_cases": []}

        cluster_comments = self.cluster_comment_map.get(cluster_id, [])
        if not cluster_comments:
            return {"centroid": [], "edge_cases": []}

        # Sort by distance to centroid (closest first)
        cluster_items_with_distances = []
        for item in cluster_comments:
            comment_id = item['id']
            cluster_idx = self.id_to_index.get(comment_id)
            if cluster_idx is not None:
                distance = self.distances[cluster_idx]
                cluster_items_with_distances.append((item['comment'], distance))

        cluster_items_with_distances.sort(key=lambda x: x[1])

        # Determine sample size based on cluster size
        cluster_size = len(cluster_items_with_distances)
        if cluster_size >= 200:
            n_samples = 12
        elif cluster_size >= 100:
            n_samples = 10
        else:
            n_samples = 8

        n_core = int(n_samples * 0.7)
        n_edge = n_samples - n_core

        core = [comment for comment, _ in cluster_items_with_distances[:n_core]]
        edge = [comment for comment, _ in cluster_items_with_distances[-n_edge:]]

        return {
            "centroid": core,
            "edge_cases": edge
        }

    def _render_dashboard_md(self, dashboard_json):
        """Render dashboard JSON to markdown format"""
        meta = dashboard_json.get('meta', {})
        audience_overview = dashboard_json.get('audience_overview', {})
        sentiment_overview = dashboard_json.get('sentiment_overview', {})
        clusters = dashboard_json.get('clusters', [])
        themes = dashboard_json.get('themes', [])
        opportunities = dashboard_json.get('opportunities', [])
        risks = dashboard_json.get('risks', [])

        md = f"""# Audience Insights Dashboard

## Overview
- Total comments: {meta.get('total_comments', 0):,}
- Unique comments (deduplicated): {meta.get('unique_comments', 0):,}
- Dominant sentiment: {sentiment_overview.get('dominant_sentiment', 'Unknown')}
- Analysis confidence: {self._get_overall_confidence(clusters)}

---

## Audience Signals at a Glance
| Theme | % of Comments | Dominant Emotion | Confidence |
|-----|--------------|------------------|------------|
"""

        for cluster in audience_overview.get('cluster_distribution', []):
            md += f"| {cluster['label']} | {cluster['percentage']:.1f}% | {cluster['dominant_sentiment']} | {cluster['confidence'].title()} |\n"

        md += "\n---\n\n## Discussion Clusters\n\n"

        for cluster in clusters:
            md += f"### {cluster['label']} ({cluster['confidence'].title()} confidence)\n"
            md += f"**Summary:** {cluster['summary'][:200]}...\n\n"

            sample_comments = cluster.get('sample_comments', {})
            if sample_comments.get('centroid'):
                md += "**Core comments:**\n"
                for comment in sample_comments['centroid'][:3]:  # Show first 3
                    md += f"- {comment}\n"
                md += "\n"

            if sample_comments.get('edge_cases'):
                md += "**Edge cases:**\n"
                for comment in sample_comments['edge_cases'][:2]:  # Show first 2
                    md += f"- {comment}\n"
                md += "\n"

        md += "---\n\n## Common Ideas & Themes\n"
        for theme in themes[:5]:  # Show top 5 themes
            md += f"- {theme['theme'].title()}: {theme['why'][:100]}...\n"

        md += "\n---\n\n## Opportunities\n"
        for opportunity in opportunities:
            md += f"- 🚀 {opportunity['action']} ({opportunity['confidence'].title()} confidence)\n"

        md += "\n---\n\n## Risks\n"
        for risk in risks:
            md += f"- ⚠️ {risk['signal'][:100]}... ({risk['confidence'].title()} confidence)\n"

        return md

    def _get_overall_confidence(self, clusters):
        """Determine overall confidence based on cluster confidences"""
        if not clusters:
            return "Low"

        high_conf = sum(1 for c in clusters if c['confidence'] == 'high')
        medium_conf = sum(1 for c in clusters if c['confidence'] == 'medium')

        if high_conf >= len(clusters) * 0.7:
            return "High"
        elif high_conf + medium_conf >= len(clusters) * 0.7:
            return "Medium"
        else:
            return "Low"

def main():
    """Main function to run the analysis engine"""
    # Load API key from environment
    api_key = os.getenv("GROQ_API_KEY")  # Changed from OPENROUTER_API_KEY

    # Load download directory from environment, default to /Users/venuvamsi/Downloads
    download_dir = os.getenv("DOWNLOAD_DIR", "/Users/venuvamsi/Downloads")

    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment. LLM functionality will be limited.")
        response = input("Do you want to continue without LLM functionality? (y/n): ")
        if response.lower() != 'y':
            return

    # Create analysis engine
    engine = CommentAnalysisEngine(download_dir=download_dir, api_key=api_key)

    # Start monitoring
    try:
        engine.monitor_and_analyze()
    except KeyboardInterrupt:
        logger.info("Analysis engine stopped by user.")
    except Exception as e:
        logger.error(f"Error running analysis engine: {e}")

if __name__ == "__main__":
    main()