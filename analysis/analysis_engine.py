import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
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
        self.global_system_prompt = """You are an analyst helping explain audience feedback on online videos.

Your role is to:
- identify clear reasons behind audience emotions
- explain differences between reasons
- stay grounded in the provided comments only

Rules:
- Do NOT invent facts or statistics
- Do NOT generalize beyond the comments shown
- Do NOT give solutions unless explicitly asked
- Prefer concise, structured explanations
- If multiple distinct reasons exist, separate them clearly

You are explaining patterns, not summarizing text."""
        
        self.sentiment_summary_prompt = """You are analyzing aggregated viewer sentiment data.

Your task:
- Interpret the overall emotional state of the audience.
- Compare raw sentiment percentages with cluster-weighted (topic-level) sentiment percentages.
- Explain what this difference implies about emotional intensity vs emotional diversity across topics.

Rules:
- Do NOT restate the numbers verbatim.
- Do NOT speculate beyond the data.
- Focus on what dominates, what is niche, and what is structurally important.

Output format:
- 1 short paragraph: overall emotional tone
- 3–5 bullet points: key takeaways"""
        
        self.emotion_cluster_reason_prompt = """You are analyzing viewer comments that all express the emotion: {EMOTION}.

The comments come from different semantic clusters.
Each cluster represents a different underlying reason.

Your task:
- Identify the main reasons why viewers feel this emotion
- Explain how these reasons differ
- Focus on causes, not cluster structure

Do NOT label clusters.
Do NOT mention cluster IDs.
Do NOT output JSON.
Use short paragraphs with clear headings.

Frame your explanation using one or more of these lenses where relevant:
- expectation vs outcome
- clarity vs confusion
- effort vs reward
- inclusion vs exclusion
- fairness vs luck

End your explanation with one sentence describing
why this emotion matters for viewer engagement or perception."""
        
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

        # Single attempt only - no retries for JSON parsing
        try:
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": self.global_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=450  # 🔒 HARD LIMIT - raised from 300 to reduce truncation
            )

            # Update last call time
            self._last_llm_call = time.time()

            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return "LLM call failed"

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
        ids = cluster_data['ids']  # comment IDs for mapping to clusters
        labels = cluster_data['labels']  # k-means cluster id per embedding
        centroids = cluster_data['centroids']  # cluster centroids
        distances = cluster_data['distances']  # distance of each embedding to its centroid

        # Load sentiment data (contains all original comments, may include duplicates)
        with open(sentiment_path, 'r') as f:
            sentiment_data = json.load(f)

        # Ensure sentiment data is in the expected format with 'id' and 'comment' fields
        processed_sentiment_data = []
        for idx, item in enumerate(sentiment_data):
            if isinstance(item, dict) and 'id' in item and 'comment' in item:
                # Already in expected format
                processed_sentiment_data.append(item)
            elif isinstance(item, dict) and 'comment' in item:
                # Add default id if missing
                item_copy = item.copy()
                if 'id' not in item_copy:
                    item_copy['id'] = idx
                processed_sentiment_data.append(item_copy)
            else:
                # Unexpected format, skip or handle appropriately
                logger.warning(f"Unexpected sentiment data format at index {idx}: {item}")
                continue
        sentiment_data = processed_sentiment_data

        # Create mapping from cluster ID to sentiment (for semantic analysis)
        cluster_sentiment_map = defaultdict(list)
        cluster_comment_map = defaultdict(list)

        # Build an index for O(1) ID lookup to avoid O(N²) behavior
        # Store as instance variable to fix scope issue
        self.id_to_index = {id_: idx for idx, id_ in enumerate(ids)}
        # Also store distances as instance variable to fix scope issue
        self.distances = distances
        # Store cluster_comment_map as instance variable for later use
        self.cluster_comment_map = cluster_comment_map

        # Map all sentiment data to clusters based on IDs
        for item in sentiment_data:
            comment_id = item['id']
            # Find the index of this comment ID in the clustering data using the pre-built index
            cluster_idx = self.id_to_index.get(comment_id)
            if cluster_idx is not None:
                cluster_label = labels[cluster_idx]

                # Map cluster to sentiments and comments
                cluster_sentiment_map[cluster_label].append(item['emotion'])
                cluster_comment_map[cluster_label].append(item)

        # Step 6.3: Compute emotion statistics (No LLM)
        # Raw Viewer Sentiment (from all sentiment JSON entries)
        self.raw_sentiment_data = self._compute_raw_sentiment_distribution(sentiment_data)


        logger.info("Starting emotion-first analysis...")

        # Step 1: Global Emotion Statistics (No LLM)
        raw_sentiment_dist = self.raw_sentiment_data["distribution"]

        # Step 2: Select emotions worth explaining
        eligible_emotions = self._select_eligible_emotions(raw_sentiment_dist, cluster_sentiment_map)
        logger.info(f"Selected {len(eligible_emotions)} emotions for detailed analysis: {eligible_emotions}")

        # Step 3: For each selected emotion, analyze clusters that express it
        emotion_cluster_analysis = self._analyze_emotion_clusters(
            cluster_sentiment_map, cluster_comment_map, labels, centroids, eligible_emotions
        )

        # Step 4: Select clusters for limited analysis (top 8 by size or entropy)
        selected_cluster_ids = self._select_clusters_for_analysis(cluster_sentiment_map)
        logger.info(f"Selected {len(selected_cluster_ids)} clusters for detailed analysis")

        # Step 5: Analyze sentiment mix for selected clusters only
        cluster_sentiment_analysis = self._analyze_cluster_sentiment_mix(
            cluster_sentiment_map, cluster_comment_map, centroids, selected_cluster_ids
        )

        # Generate engagement diagnostics
        engagement_diagnostics = self._generate_engagement_diagnostics(
            cluster_sentiment_map, cluster_comment_map
        )

        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(
            raw_sentiment_dist, emotion_cluster_analysis,
            cluster_sentiment_analysis, engagement_diagnostics
        )

        # Create final report
        report = self._create_final_report(
            raw_sentiment_dist,
            emotion_cluster_analysis,
            cluster_sentiment_analysis,
            engagement_diagnostics,
            actionable_insights
        )

        # Calculate cluster-weighted sentiment distribution for dashboard
        cluster_weighted_sentiment_dist = self._compute_cluster_weighted_sentiment_distribution(cluster_sentiment_map)

        # Create and save dashboard JSON and markdown
        dashboard_json = self._build_dashboard_json(
            raw_sentiment_dist, cluster_weighted_sentiment_dist,
            cluster_sentiment_analysis, emotion_cluster_analysis,
            engagement_diagnostics, actionable_insights,
            cluster_sentiment_map
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

    def _compute_raw_sentiment_distribution(self, sentiment_data: List[Dict]) -> Dict[str, Any]:
        """Compute raw sentiment distribution from all sentiment data"""
        emotions = [item['emotion'] for item in sentiment_data]
        total = len(emotions)
        emotion_counts = Counter(emotions)
        return {
            "distribution": {emotion: count/total for emotion, count in emotion_counts.items()},
            "total": total
        }

    def _compute_cluster_weighted_sentiment_distribution(self, cluster_sentiment_map: Dict) -> Dict[str, float]:
        """Compute cluster-weighted sentiment distribution (topic-level emotional diversity)"""
        # Count how many clusters express each emotion (not how many comments)
        dominant_emotions = []
        for cluster_id, emotions in cluster_sentiment_map.items():
            if emotions:  # If cluster has emotions
                # Find the most common emotion in this cluster
                emotion_counts = Counter(emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
                dominant_emotions.append(dominant_emotion)

        # Count how many clusters express each emotion
        total_clusters = len(dominant_emotions)
        emotion_counts = Counter(dominant_emotions)
        return {emotion: count/total_clusters if total_clusters > 0 else 0.0
                for emotion, count in emotion_counts.items()}

    def _generate_global_sentiment_summary(self, raw_dist: Dict[str, float],
                                         cluster_weighted_dist: Dict[str, float]) -> str:
        """Generate LLM summary of global sentiment"""
        if not self.llm:
            return "LLM not available for global sentiment summary"

        # Format the distributions for the prompt
        raw_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in raw_dist.items()])
        cluster_weighted_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in cluster_weighted_dist.items()])

        prompt = f"""
Raw sentiment distribution (volume): {raw_str}
Cluster-weighted sentiment distribution (topic-level diversity): {cluster_weighted_str}

{self.sentiment_summary_prompt}
"""

        try:
            response = self._call_llm(prompt)
            return str(response)
        except Exception as e:
            logger.warning(f"Error generating global sentiment summary: {e}")
            return f"Error generating summary: {e}"

    def _analyze_emotion_clusters(self, cluster_sentiment_map: Dict,
                                cluster_comment_map: Dict,
                                labels: np.ndarray,
                                centroids: np.ndarray,
                                eligible_emotions: List[str] = None) -> Dict:  # Changed distances to centroids
        """Analyze each emotion by cluster to understand reasons"""
        if not self.llm:
            return {"error": "LLM not available for emotion cluster analysis"}

        results = {}

        # Use eligible emotions if provided, otherwise get all emotions
        emotions_to_analyze = eligible_emotions if eligible_emotions else []
        if not eligible_emotions:
            # Get unique emotions
            all_emotions = set()
            for emotions in cluster_sentiment_map.values():
                all_emotions.update(emotions)
            emotions_to_analyze = list(all_emotions)

        # Get the raw sentiment distribution from the instance variable
        raw_sentiment_dist = self.raw_sentiment_data["distribution"] if hasattr(self, 'raw_sentiment_data') and self.raw_sentiment_data else {}

        for emotion in emotions_to_analyze:
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

            # Sort clusters by size (within this emotion) and select top 3-5
            sorted_clusters = sorted(emotion_clusters.items(), key=lambda x: x[1], reverse=True)
            selected_clusters = dict(sorted_clusters[:5])  # Select top 5 clusters max

            # Build comment blocks for prompt (without cluster labels)
            comment_blocks = []
            for cluster_id in selected_clusters.keys():
                cluster_data = cluster_analysis[cluster_id]
                comment_block = f"Comments ({cluster_data['size']} total):\n"
                for comment in cluster_data['comments'][:8]:  # Limit to 8 comments max
                    comment_block += f"- {comment}\n"
                comment_blocks.append(comment_block.strip())

            # Build the prompt with the new format
            comment_blocks_text = "\n\n".join(comment_blocks)

            # Add contrast-based reasoning - select high-purity and mixed clusters
            high_purity_cluster = None
            mixed_cluster = None

            # Find clusters with different emotional purity
            for cluster_id in selected_clusters.keys():
                cluster_emotions = cluster_sentiment_map.get(cluster_id, [])
                if cluster_emotions:
                    emotion_counts = Counter(cluster_emotions)
                    emotion_dist = {e: count/len(cluster_emotions) for e, count in emotion_counts.items()}
                    entropy = self._calculate_entropy(emotion_dist.values())

                    # Track the highest purity cluster for this emotion
                    current_emotion_purity = emotion_dist.get(emotion, 0)
                    if current_emotion_purity > 0.7:
                        if not high_purity_cluster:
                            high_purity_cluster = cluster_id
                        else:
                            # Compare with existing high_purity_cluster
                            existing_emotion_purity = Counter(cluster_sentiment_map[high_purity_cluster]).get(emotion, 0) / len(cluster_sentiment_map[high_purity_cluster])
                            if current_emotion_purity > existing_emotion_purity:
                                high_purity_cluster = cluster_id
                    elif entropy > 1.5:
                        if not mixed_cluster:
                            mixed_cluster = cluster_id
                        else:
                            # Compare with existing mixed_cluster
                            existing_mixed_entropy = self._calculate_entropy(
                                {e: count/len(cluster_sentiment_map[mixed_cluster]) for e, count in
                                 Counter(cluster_sentiment_map[mixed_cluster]).items()}.values())
                            if entropy > existing_mixed_entropy:
                                mixed_cluster = cluster_id

            contrast_info = ""
            if high_purity_cluster and mixed_cluster:
                high_purity_emotions = cluster_sentiment_map.get(high_purity_cluster, [])
                mixed_emotions = cluster_sentiment_map.get(mixed_cluster, [])

                contrast_info = f"""
Compare these two groups:
- One where the emotion is consistent (with {len(high_purity_emotions)} comments)
- One where emotions are mixed (with {len(mixed_emotions)} comments)

Explain what differentiates these viewers.
"""

            # Add special instructions for neutral emotion
            neutral_instruction = ""
            if emotion == "neutral" or raw_sentiment_dist.get("neutral", 0) >= 0.40:
                neutral_instruction = """
Neutral reactions are prominent.
Focus on what this implies about:
- emotional engagement
- memorability
- passive vs active viewing
"""

            prompt = f"""
{comment_blocks_text}

---

{contrast_info}

{neutral_instruction}

{self.emotion_cluster_reason_prompt.format(EMOTION=emotion)}
"""

            try:
                response = self._call_llm(prompt)  # Use new calling pattern
                # Store raw LLM output with confidence
                confidence = self._calculate_emotion_confidence(emotion, raw_sentiment_dist.get(emotion, 0), len(selected_clusters))
                results[emotion] = {
                    "analysis_text": str(response),
                    "confidence": confidence,
                    "confidence_reason": self._get_confidence_reason(emotion, raw_sentiment_dist.get(emotion, 0), len(selected_clusters))
                }
            except Exception as e:
                logger.error(f"Error analyzing emotion {emotion}: {e}")
                # Store fallback text with low confidence
                results[emotion] = {
                    "analysis_text": "The comments show mixed signals without a dominant, clearly articulated reason.",
                    "confidence": "low",
                    "confidence_reason": "LLM call failed"
                }

        return results

    def _analyze_cluster_sentiment_mix(self, cluster_sentiment_map: Dict,
                                     cluster_comment_map: Dict,
                                     centroids: np.ndarray,
                                     selected_cluster_ids: List[int] = None) -> Dict:
        """Analyze sentiment mix within each cluster"""
        if not self.llm:
            return {"error": "LLM not available for cluster sentiment analysis"}

        results = {}

        # Only analyze selected clusters if provided, otherwise analyze all
        clusters_to_analyze = selected_cluster_ids if selected_cluster_ids else cluster_sentiment_map.keys()

        for cluster_id in clusters_to_analyze:
            emotions = cluster_sentiment_map.get(cluster_id, [])
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

            # Prepare prompt for LLM using the new format
            comment_list = "\n".join([f"- {comment}" for comment in sample_comments])

            prompt = f"""
We are analyzing a discussion topic based on viewer comments.

Below are comments from a single topic cluster.
Viewers express multiple emotions in this cluster.

Your task:
- Identify what this topic is mainly about
- Explain why the emotions are consistent or mixed
- Keep the explanation factual and grounded in comments

---

Topic cluster comments:
{comment_list}

---

Output format:

Topic summary:
<1 sentence>

Emotional pattern:
<2–3 sentences explaining why these emotions appear together>
"""

            try:
                response = self._call_llm(prompt)  # Use new calling pattern
                response_str = str(response)

                # Extract topic label from the response (first line after "Topic summary:")
                topic_label = "Topic"
                lines = response_str.split('\n')
                for i, line in enumerate(lines):
                    if "Topic summary:" in line:
                        if i + 1 < len(lines):
                            topic_label = lines[i + 1].strip()
                            # Clean up the topic label to get just the first sentence/phrase
                            if topic_label.startswith(':'):
                                topic_label = topic_label[1:].strip()
                            # Take only the first 50 characters to keep it concise
                            topic_label = topic_label[:50].split('.')[0]  # Stop at first sentence
                        break

                results[cluster_id] = {
                    'topic_label': topic_label,
                    'summary': response_str,
                    'emotion_distribution': emotion_dist,
                    'entropy': entropy
                }
            except Exception as e:
                logger.warning(f"Error analyzing cluster {cluster_id}: {e}")
                results[cluster_id] = {
                    'topic_label': f"Cluster {cluster_id}",
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

    def _calculate_sentiment_entropy(self, sentiment_dist: Dict[str, float]) -> float:
        """Calculate entropy of a sentiment distribution"""
        probabilities = list(sentiment_dist.values())
        return self._calculate_entropy(probabilities)

    def _select_eligible_emotions(self, raw_sentiment_dist: Dict[str, float],
                                 cluster_sentiment_map: Dict) -> List[str]:
        """Select emotions that are eligible for detailed analysis"""
        # Get emotions sorted by percentage
        sorted_emotions = sorted(raw_sentiment_dist.items(), key=lambda x: x[1], reverse=True)

        eligible_emotions = set()

        # Rule 1: Emotions >= 3% of all comments
        for emotion, percentage in sorted_emotions:
            if percentage >= 0.03:  # 3%
                eligible_emotions.add(emotion)

        # Rule 2: Top 6 by volume
        top_6_emotions = [emotion for emotion, _ in sorted_emotions[:6]]
        eligible_emotions.update(top_6_emotions)

        # Rule 3: Emotions with high entropy (diverse reasons)
        for emotion in raw_sentiment_dist.keys():
            # Count how many clusters express this emotion
            clusters_with_emotion = 0
            for cluster_id, emotions in cluster_sentiment_map.items():
                if emotion in emotions:
                    clusters_with_emotion += 1

            # If emotion appears in many clusters, it has diverse reasons
            if clusters_with_emotion > len(cluster_sentiment_map) * 0.3:  # More than 30% of clusters
                eligible_emotions.add(emotion)

        # Rule 4: Elevate neutral emotion if >= 40% of comments
        if raw_sentiment_dist.get('neutral', 0) >= 0.40:  # 40% threshold
            eligible_emotions.add('neutral')

        # Cap at 8 emotions max
        return list(eligible_emotions)[:8]

    def _select_clusters_for_analysis(self, cluster_sentiment_map: Dict) -> List[int]:
        """Select clusters for limited analysis based on size and entropy"""
        cluster_metrics = []

        for cluster_id, emotions in cluster_sentiment_map.items():
            total_emotions = len(emotions)
            emotion_counts = Counter(emotions)
            emotion_dist = {emotion: count/total_emotions for emotion, count in emotion_counts.items()}
            entropy = self._calculate_entropy(emotion_dist.values())

            cluster_metrics.append({
                'id': cluster_id,
                'size': total_emotions,
                'entropy': entropy
            })

        # Sort by size first, then by entropy
        cluster_metrics.sort(key=lambda x: (x['size'], x['entropy']), reverse=True)

        # Select top 5 by size
        top_by_size = [cm['id'] for cm in cluster_metrics[:5]]

        # Sort by entropy and select top 5
        cluster_metrics.sort(key=lambda x: x['entropy'], reverse=True)
        top_by_entropy = [cm['id'] for cm in cluster_metrics[:5]]

        # Combine and remove duplicates, cap at 8
        selected = list(set(top_by_size + top_by_entropy))[:8]

        return selected


    def _calculate_theme_sentiment_breakdown(self, cluster_sentiment_map: Dict) -> List[Dict]:
        """Calculate per-theme sentiment breakdown"""
        theme_breakdowns = []

        for cluster_id, emotions in cluster_sentiment_map.items():
            if not emotions:
                continue

            # Calculate sentiment distribution for this theme
            total_emotions = len(emotions)
            emotion_counts = Counter(emotions)
            sentiment_breakdown = {emotion: round((count/total_emotions) * 100)
                                 for emotion, count in emotion_counts.items()}

            # Calculate dominant sentiment
            dominant_sentiment = max(sentiment_breakdown, key=sentiment_breakdown.get)

            # Calculate sentiment entropy
            entropy = self._calculate_sentiment_entropy(
                {k: v/100 for k, v in sentiment_breakdown.items()}
            )

            # Determine confidence level
            theme_size = len(emotions)
            if theme_size >= 250 and entropy < 1.0:
                confidence = "high"
            elif theme_size >= 120 and entropy < 1.5:
                confidence = "medium"
            else:
                confidence = "low"

            theme_breakdowns.append({
                "cluster_id": int(cluster_id),  # Keep internal for mapping
                "label": f"Theme {cluster_id}",  # Changed label to be theme-focused
                "comment_count": theme_size,
                "sentiment_breakdown": sentiment_breakdown,
                "dominant_sentiment": dominant_sentiment,
                "sentiment_entropy": round(entropy, 2),
                "confidence": confidence
            })

        return theme_breakdowns


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
            logger.warning(f"Error generating engagement diagnostics: {e}")
            return f"Error generating diagnostics: {e}"

    def _generate_actionable_insights(self, raw_sentiment_dist: Dict[str, float],
                                    emotion_analysis: Dict,
                                    cluster_analysis: Dict,
                                    engagement_diagnostics: str) -> str:
        """Generate actionable insights from all analysis"""
        if not self.llm:
            return "LLM not available for actionable insights"

        # Create structured summaries from emotion analysis to reduce token usage
        # Focus on emotions and their reasons, not clusters
        emotion_summary = ""
        for emotion, analysis in list(emotion_analysis.items())[:3]:
            if isinstance(analysis, dict) and "analysis_text" in analysis:
                # Handle new format with analysis_text - use first paragraph instead of truncation
                analysis_text = analysis.get("analysis_text", "")
                # Get first paragraph or first 2 sentences to keep reasoning intact
                first_paragraph = analysis_text.split("\n")[0][:300] if "\n" in analysis_text else analysis_text.split(". ")[0][:300] if ". " in analysis_text else analysis_text[:300]
                emotion_summary += f"- {emotion}: {first_paragraph}...\n"
            elif isinstance(analysis, dict) and "reasons" in analysis:
                # Extract reasons from structured JSON (fallback)
                reasons = analysis.get("reasons", [])
                if reasons:
                    # Create a summary from the top reasons
                    top_reasons = reasons[:3]  # Take top 3 reasons
                    reason_labels = [reason.get("label", "Unknown reason") for reason in top_reasons]
                    emotion_summary += f"- {emotion}: {', '.join(reason_labels)}\n"
                else:
                    emotion_summary += f"- {emotion}: No specific reasons identified\n"
            else:
                emotion_summary += f"- {emotion}: {str(analysis)[:100]}...\n"  # Fallback for non-structured data

        # Format raw sentiment distribution for the prompt
        raw_sentiment_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in raw_sentiment_dist.items()])

        prompt = f"""
We analyzed viewer comments and found these emotion percentages:
{raw_sentiment_str}

Emotional Drivers:
{emotion_summary}

Engagement Diagnostics:
{engagement_diagnostics}

You are given insights about viewer emotions and their underlying reasons.

Your task:
- Identify meaningful opportunities, risks, or signals based on emotional patterns
- Focus on what the emotions and their drivers suggest for content strategy
- Do NOT repeat the emotion summaries themselves

Label each item clearly.

---

Output format:
- 3 opportunities or positive signals
- 2 risks or concerns
- 1 strategic recommendation
"""

        try:
            response = self._call_llm(prompt)  # Use new calling pattern
            return str(response)
        except Exception as e:
            logger.warning(f"Error generating actionable insights: {e}")
            return f"Error generating insights: {e}"

    def _create_final_report(self, raw_sentiment_dist: Dict[str, float],
                           emotion_analysis: Dict,
                           cluster_analysis: Dict,
                           engagement_diagnostics: str,
                           actionable_insights: str) -> str:
        """Create the final analysis report in markdown format"""
        # Format raw sentiment distribution
        raw_sentiment_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in raw_sentiment_dist.items()])

        report = f"""# Comment Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## About This Analysis
This report presents viewer sentiment analysis with an emotion-first approach:
- **Raw Sentiment**: Based on all comments, reflecting emotional intensity/volume
- **Emotion-Cluster Analysis**: Explains why viewers feel certain emotions

## 1. Overall Viewer Sentiment
**Emotion Distribution:** {raw_sentiment_str}

## 2. Key Emotional Drivers
"""
        for emotion, analysis in emotion_analysis.items():
            if isinstance(analysis, dict) and "analysis_text" in analysis:
                # Handle new format with analysis_text
                report += f"""
### Reasons for {emotion.capitalize()}
{analysis['analysis_text']}
"""
            else:
                report += f"""
### Reasons for {emotion.capitalize()}
{analysis}
"""

        report += f"""
## 3. Key Discussion Themes
"""
        # Include only 3-4 topic summaries, no cluster IDs
        for i, (cluster_id, analysis) in enumerate(list(cluster_analysis.items())[:4]):
            report += f"""
### {analysis.get('topic_label', f'Discussion Theme {i+1}')}
{analysis['summary']}
"""

        report += f"""
## 4. Engagement Diagnostics
{engagement_diagnostics}

## 5. Actionable Insights
{actionable_insights}

## 6. What to Do Next
Based on the analysis above, here are the recommended next steps:
- Focus on the main reasons behind dominant emotions
- Address concerns raised in negative emotion clusters
- Leverage positive emotional drivers for content strategy
"""
        return report

    def _build_dashboard_json(self, raw_sentiment_dist, cluster_weighted_sentiment_dist,
                             cluster_sentiment_analysis, emotion_cluster_analysis,
                             engagement_diagnostics, actionable_insights,
                             cluster_sentiment_map=None):
        """Build the dashboard JSON from analysis results"""
        # Extract video ID from the file name
        video_id = "unknown"
        if hasattr(self, '_current_sentiment_file'):
            import re
            match = re.search(r'([A-Za-z0-9_-]+)_comments_sentiments\.json', self._current_sentiment_file.name)
            if match:
                video_id = match.group(1)

        # Calculate total comments
        total_comments = self.raw_sentiment_data["total"] if hasattr(self, 'raw_sentiment_data') and self.raw_sentiment_data else 0
        raw_sentiment_dist = self.raw_sentiment_data["distribution"] if hasattr(self, 'raw_sentiment_data') and self.raw_sentiment_data else raw_sentiment_dist
        embedded_comments = len(self.id_to_index) if hasattr(self, 'id_to_index') else 0

        # Build cluster distribution
        cluster_distribution = []
        for cluster_id, analysis in cluster_sentiment_analysis.items():
            cluster_size = len(cluster_sentiment_map.get(cluster_id, []))
            if cluster_size > 0:
                cluster_info = analysis.get('emotion_distribution', {})
                dominant_sentiment = max(cluster_info, key=cluster_info.get) if cluster_info else "unknown"
                confidence = self._confidence_label(cluster_size, analysis.get('entropy', 0))

                cluster_distribution.append({
                    "cluster_id": int(cluster_id),
                    "label": f"Cluster {cluster_id}",
                    "comment_count": int(cluster_size),
                    "percentage": round((cluster_size / total_comments) * 100, 2) if total_comments > 0 else 0.0,
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
            cluster_size = len(cluster_sentiment_map.get(cluster_id, []))
            entropy = analysis.get('entropy', 0)
            confidence = self._confidence_label(cluster_size, entropy)

            # Get sample comments for this cluster
            sample_comments = self._get_sample_comments_for_cluster(cluster_id)

            clusters.append({
                "cluster_id": int(cluster_id),
                "label": analysis.get('topic_label', f"Cluster {cluster_id}"),
                "comment_count": int(cluster_size),
                "percentage": round((cluster_size / total_comments) * 100, 2) if total_comments > 0 else 0.0,
                "sentiment_distribution": {k: int(v * 100) for k, v in analysis.get('emotion_distribution', {}).items()},
                "entropy": round(entropy, 2),
                "confidence": confidence,
                "summary": analysis.get('summary', ''),
                "sample_comments": sample_comments
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

        # Calculate emotion confidence for each emotion
        emotion_confidence_map = {}
        for emotion, percentage in raw_sentiment_dist.items():
            # Count how many clusters express this emotion and calculate average entropy
            clusters_with_emotion = 0
            cluster_entropies = []
            for cluster_id, emotions in cluster_sentiment_map.items():
                # Get emotions in this cluster that match the current emotion
                cluster_emotions = [e for e in emotions if e == emotion]
                if cluster_emotions:
                    clusters_with_emotion += 1
                    # Calculate entropy for this cluster's emotion distribution
                    emotion_counts = Counter(emotions)
                    emotion_dist = {e: count/len(emotions) for e, count in emotion_counts.items()}
                    entropy = self._calculate_entropy(emotion_dist.values())
                    cluster_entropies.append(entropy)

            # Calculate average entropy for this emotion across clusters
            avg_entropy = sum(cluster_entropies) / len(cluster_entropies) if cluster_entropies else float('inf')

            # Calculate confidence based on percentage, cluster distribution, and entropy
            emotion_pct = percentage * 100  # Convert to percentage
            if emotion_pct >= 10 and clusters_with_emotion <= 3 and (avg_entropy == float('inf') or avg_entropy < 1.2):
                confidence = "high"
            elif emotion_pct >= 5:
                confidence = "medium"
            else:
                confidence = "low"

            emotion_confidence_map[emotion] = confidence

        # Calculate summary stats - NEW FEATURE FROM FIX5.md
        # Overall sentiment breakdown
        overall_sentiment = {
            "raw": {k: round(v * 100, 1) for k, v in raw_sentiment_dist.items()},
            "topic_level": {k: round(v * 100, 1) for k, v in cluster_weighted_sentiment_dist.items()},
            "dominant_raw": max(raw_sentiment_dist, key=raw_sentiment_dist.get) if raw_sentiment_dist else "unknown",
            "dominant_topic_level": max(cluster_weighted_sentiment_dist, key=cluster_weighted_sentiment_dist.get) if cluster_weighted_sentiment_dist else "unknown"
        }

        # Calculate theme sentiment breakdowns
        theme_sentiment_stats = self._calculate_theme_sentiment_breakdown(cluster_sentiment_map or {})

        # Build emotions array (primary output object)
        emotions_array = []
        for emotion, percentage in sorted(
            raw_sentiment_dist.items(), key=lambda x: x[1], reverse=True
        ):
            emotion_data = {
                "emotion": emotion,
                "percentage": round(percentage * 100, 1),
                "confidence": emotion_confidence_map.get(emotion, "low"),
                "summary": "",  # Will be filled below
                "reasons": []   # This would come from emotion analysis
            }

            # Add reasons from emotion analysis if available
            emotion_analysis = emotion_cluster_analysis.get(emotion, {})
            if isinstance(emotion_analysis, dict):
                # Handle new format with analysis_text and confidence
                if "analysis_text" in emotion_analysis:
                    emotion_data["summary"] = emotion_analysis["analysis_text"]
                    emotion_data["confidence"] = emotion_analysis.get("confidence", emotion_data["confidence"])
                    emotion_data["confidence_reason"] = emotion_analysis.get("confidence_reason", "")
                elif "reasons" in emotion_analysis:  # Fallback for old format
                    emotion_data["reasons"] = emotion_analysis["reasons"]

                    # Generate emotion summary from reasons
                    reasons = emotion_analysis.get("reasons", [])
                    if reasons:
                        # Create a summary from the top reasons
                        top_reasons = reasons[:3]  # Take top 3 reasons
                        reason_labels = [reason.get("label", "Unknown reason") for reason in top_reasons]
                        emotion_data["summary"] = (
                            f"Viewers mainly feel {emotion} due to "
                            f"{', '.join(reason_labels[:2])}."
                        )

            emotions_array.append(emotion_data)

        # Build summary_stats section
        summary_stats = {
            "overall_sentiment": overall_sentiment,
            "theme_sentiment_stats": theme_sentiment_stats
        }

        dashboard_json = {
            "meta": {
                "video_id": video_id,
                "total_comments": int(total_comments),
                "embedded_comments": int(embedded_comments),
                "analysis_timestamp": datetime.now().isoformat() + "Z",
                "model": "llama-3.1-8b-instant"
            },
            "summary_stats": summary_stats,  # NEW SECTION FROM FIX5.md
            "emotions": emotions_array,  # PRIMARY OUTPUT OBJECT - EMOTION-FIRST
            "focus_topics": clusters,  # Limited clusters only, renamed for clarity
            "sentiment_overview": sentiment_overview,
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
        summary_stats = dashboard_json.get('summary_stats', {})
        emotions = dashboard_json.get('emotions', [])
        focus_topics = dashboard_json.get('focus_topics', [])
        sentiment_overview = dashboard_json.get('sentiment_overview', {})
        opportunities = dashboard_json.get('opportunities', [])
        risks = dashboard_json.get('risks', [])

        md = f"""# Audience Insights Dashboard

## Overview
- Total comments: {meta.get('total_comments', 0):,}
- Embedded comments: {meta.get('embedded_comments', 0):,}
- Dominant sentiment: {sentiment_overview.get('dominant_sentiment', 'Unknown')}
- Analysis confidence: {self._get_overall_confidence(focus_topics)}

---

## Emotional Snapshot

"""

        # Add overall sentiment breakdown
        overall_sentiment = summary_stats.get('overall_sentiment', {})
        raw_sentiment = overall_sentiment.get('raw', {})

        if raw_sentiment:
            md += "**Overall viewer emotions (raw):**\n"
            for emotion, pct in sorted(raw_sentiment.items(), key=lambda x: x[1], reverse=True):
                md += f"- {emotion.title()}: {pct}%\n"
            md += "\n"

        md += "\n---\n\n## Emotion Analysis\n\n"

        # Emotion-first structure
        for emotion_data in emotions:
            emotion = emotion_data.get('emotion', 'unknown')
            percentage = emotion_data.get('percentage', 0)
            confidence = emotion_data.get('confidence', 'low')
            confidence_reason = emotion_data.get('confidence_reason', '')

            md += f"## Emotion: {emotion.capitalize()} ({percentage}% - {confidence} confidence)\n"
            if confidence_reason:
                md += f"**Confidence reason:** {confidence_reason}\n\n"

            # Show reasons for this emotion
            reasons = emotion_data.get('reasons', [])
            analysis_text = emotion_data.get('summary', '')

            # Check if there's analysis text in the emotion data
            if analysis_text:
                md += f"**Analysis:** {analysis_text}\n\n"
            elif reasons:
                md += "### Why viewers feel this emotion\n"
                for reason in reasons:
                    label = reason.get('label', 'Unknown reason')
                    explanation = reason.get('explanation', 'No explanation provided')
                    md += f"- **{label}**: {explanation}\n"
                md += "\n"

            md += "\n"

        md += "\n---\n\n## Focus Topics\n\n"

        for topic in focus_topics:
            # Use the topic label instead of exposing cluster ID
            topic_label = topic.get('label', f"Topic")
            md += f"### {topic_label} ({topic['confidence'].title()} confidence)\n"
            md += f"**Summary:** {topic['summary'][:200]}...\n\n"

            # Add per-topic sentiment breakdown
            topic_stats = next((stat for stat in summary_stats.get('theme_sentiment_stats', [])
                                if stat.get('cluster_id') == topic.get('cluster_id')), None)
            if topic_stats:
                sentiment_breakdown = topic_stats.get('sentiment_breakdown', {})
                if sentiment_breakdown:
                    md += "**Emotional makeup:**\n"
                    for emotion, pct in sorted(sentiment_breakdown.items(), key=lambda x: x[1], reverse=True):
                        md += f"- {emotion.title()} ({pct}%)\n"
                    md += "\n"

                # Add interpretation
                dominant_sentiment = topic_stats.get('dominant_sentiment', 'unknown')
                md += f"**Interpretation:**\n"
                md += f"This theme is primarily associated with {dominant_sentiment} sentiment.\n\n"

            sample_comments = topic.get('sample_comments', {})
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

        md += "---\n\n## Opportunities\n"
        for opportunity in opportunities:
            md += f"- 🚀 {opportunity['action']} ({opportunity['confidence'].title()} confidence)\n"

        md += "\n---\n\n## Risks\n"
        for risk in risks:
            md += f"- ⚠️ {risk['signal'][:100]}... ({risk['confidence'].title()} confidence)\n"

        return md

    def _calculate_emotion_confidence(self, emotion: str, percentage: float, cluster_count: int) -> str:
        """Calculate deterministic confidence for emotion analysis"""
        # Calculate confidence based on percentage and cluster distribution
        if percentage >= 0.10 and cluster_count <= 3:  # High percentage, few focused clusters
            return "high"
        elif percentage >= 0.05:  # At least 5% of comments
            return "medium"
        else:
            return "low"

    def _get_confidence_reason(self, emotion: str, percentage: float, cluster_count: int) -> str:
        """Generate reason for confidence level"""
        if percentage >= 0.10 and cluster_count <= 3:
            return "Large percentage with focused reasons"
        elif percentage >= 0.05:
            return "Moderate percentage with multiple reasons"
        else:
            return "Low percentage of comments"

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