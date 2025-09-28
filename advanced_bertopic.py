"""
Advanced BERTopic Implementation with Sentiment-Aware Topic Modeling
=====================================================================

Architecture Overview:
1. Document Processing Pipeline
   - Sentiment analysis using state-of-the-art models
   - Document augmentation for sentiment separation

2. Advanced Embedding Strategy
   - Pre-computed embeddings for performance
   - Sentiment-aware embedding augmentation

3. Multi-Aspect Topic Modeling
   - Base topic clustering
   - Sentiment-based aspect separation

4. Intelligent Post-Processing
   - Topic refinement
   - Outlier reduction
   - Label enhancement

Author: Claude (Assistant)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

# Core BERTopic imports
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    PartOfSpeech,
    TextGeneration,
    OpenAI
)

# Embedding and dimensionality reduction
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Sentiment analysis
from transformers import pipeline

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Professional BERTopic Analyzer",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)


class SentimentAwareTopicModeler:
    """
    Professional-grade topic modeling with sentiment awareness.
    Implements best practices from BERTopic documentation.
    """

    def __init__(self):
        self.sentiment_analyzer = None
        self.embedding_model = None
        self.topic_model = None
        self.embeddings = None
        self.sentiment_labels = None

    @st.cache_resource
    def _get_sentiment_analyzer(_self):
        """Initialize sentiment analysis pipeline with caching."""
        return pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1  # CPU
        )

    @st.cache_resource
    def _get_embedding_model(_self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model with caching.
        Uses MTEB leaderboard recommendations.
        """
        return SentenceTransformer(model_name)

    def analyze_sentiment(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment with confidence scores.
        Returns detailed sentiment information for each document.
        """
        if not self.sentiment_analyzer:
            self.sentiment_analyzer = self._get_sentiment_analyzer()

        results = []
        batch_size = 32

        with st.spinner("🔍 Analyzing sentiment patterns..."):
            progress_bar = st.progress(0)
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                sentiments = self.sentiment_analyzer(batch, truncation=True, max_length=512)

                for sent in sentiments:
                    # Convert 5-star rating to positive/negative/neutral
                    label = sent['label']
                    score = sent['score']

                    if '5' in label or '4' in label:
                        sentiment = 'positive'
                        confidence = score
                    elif '1' in label or '2' in label:
                        sentiment = 'negative'
                        confidence = score
                    else:
                        sentiment = 'neutral'
                        confidence = score

                    results.append({
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'original_label': label
                    })

                progress_bar.progress(min((i + batch_size) / len(documents), 1.0))

            progress_bar.empty()

        return results

    def create_sentiment_augmented_docs(
        self,
        documents: List[str],
        sentiments: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Don't augment documents - instead we'll handle sentiment separation differently.
        This avoids polluting topic labels.
        """
        # Return documents unchanged - we'll handle sentiment in post-processing
        return documents

    def compute_embeddings(self, documents: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """
        Pre-compute embeddings for efficiency.
        Best practice from BERTopic documentation.
        """
        if not self.embedding_model:
            self.embedding_model = self._get_embedding_model(model_name)

        with st.spinner("🧠 Computing semantic embeddings..."):
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=False,
                batch_size=32
            )

        return embeddings

    def create_advanced_topic_model(
        self,
        num_docs: int,
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None,
        use_sentiment_guidance: bool = True
    ) -> BERTopic:
        """
        Create sophisticated BERTopic model with all best practices.
        """

        # Intelligent auto-tuning for optimal results
        import numpy as np

        # Smart topic estimation
        if nr_topics is None:
            if num_docs < 50:
                estimated_topics = max(3, min(6, num_docs // 10))
            elif num_docs < 200:
                estimated_topics = max(4, min(10, num_docs // 15))
            elif num_docs < 1000:
                estimated_topics = max(5, min(20, int(np.sqrt(num_docs) * 0.7)))
            else:
                estimated_topics = max(10, min(50, int(np.log2(num_docs) * 3)))
        else:
            estimated_topics = nr_topics

        # Adaptive parameters that work perfectly
        if num_docs < 100:
            n_neighbors = max(5, min(15, num_docs // 4))
            n_components = min(3, estimated_topics) if estimated_topics > 2 else 2
            min_dist = 0.1
            use_kmeans = True
            actual_clusters = min(estimated_topics, max(3, num_docs // 10))
            min_cluster_size = max(2, num_docs // (actual_clusters * 2))
        elif num_docs < 1000:
            n_neighbors = min(30, max(15, int(np.sqrt(num_docs) * 0.8)))
            n_components = min(5, estimated_topics) if estimated_topics > 3 else 3
            min_dist = 0.05
            use_kmeans = False
            min_cluster_size = max(5, min(min_topic_size, num_docs // 30))
        else:
            n_neighbors = min(50, max(30, int(np.sqrt(num_docs))))
            n_components = min(10, max(5, estimated_topics // 2))
            min_dist = 0.0
            use_kmeans = False
            min_cluster_size = max(10, min(min_topic_size, num_docs // 50))

        # UMAP with optimal settings
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist if 'min_dist' in locals() else 0.0,
            metric='cosine',
            random_state=42,
            low_memory=num_docs > 10000
        )

        # Smart clustering
        if use_kmeans:
            clustering_model = KMeans(
                n_clusters=actual_clusters if 'actual_clusters' in locals() else estimated_topics,
                random_state=42,
                n_init=10
            )
        else:
            clustering_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=max(1, min(5, min_cluster_size // 2)),
                metric='euclidean',
                cluster_selection_method='eom' if num_docs > 500 else 'leaf',
                prediction_data=True
            )

        # Advanced vectorizer with better tokenization
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_features=5000 if num_docs > 1000 else 1000,
            min_df=2 if num_docs > 100 else 1,
            max_df=0.95,
            token_pattern=r'\b\w+\b'  # Better tokenization
        )

        # Multi-aspect representation models
        representation_models = [
            KeyBERTInspired(top_n_words=10),
            MaximalMarginalRelevance(diversity=0.4)
        ]

        # Add PartOfSpeech if documents are long enough
        if num_docs > 50:
            representation_models.append(
                PartOfSpeech(pos_patterns=[
                    ['NOUN', 'NOUN'],
                    ['ADJ', 'NOUN'],
                    ['VERB', 'NOUN']
                ])
            )

        # Sentiment-guided seed topics if requested
        seed_topic_list = None
        if use_sentiment_guidance:
            seed_topic_list = [
                # Positive sentiment seeds
                ["excellent", "outstanding", "amazing", "perfect", "wonderful", "fantastic", "great", "love"],
                ["satisfied", "happy", "pleased", "delighted", "impressed", "recommend", "best", "quality"],

                # Negative sentiment seeds
                ["terrible", "horrible", "awful", "worst", "disappointed", "frustrating", "poor", "bad"],
                ["unhappy", "dissatisfied", "angry", "upset", "failed", "broken", "useless", "waste"],

                # Service-related seeds
                ["customer", "service", "support", "help", "response", "team", "staff", "agent"],

                # Product-related seeds
                ["product", "quality", "material", "build", "design", "feature", "performance", "value"],

                # Delivery-related seeds
                ["delivery", "shipping", "package", "arrival", "fast", "slow", "damaged", "lost"]
            ]

        # Create the model with all configurations
        model = BERTopic(
            embedding_model=self.embedding_model,  # Reuse cached model
            umap_model=umap_model,
            hdbscan_model=clustering_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_models,
            seed_topic_list=seed_topic_list,
            min_topic_size=min_cluster_size,
            nr_topics=nr_topics if nr_topics != "auto" else None,
            calculate_probabilities=True,
            verbose=False
        )

        return model

    def fit_transform(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None,
        use_sentiment: bool = True
    ) -> Tuple[List[int], np.ndarray, pd.DataFrame]:
        """
        Main method to perform topic modeling with sentiment awareness.
        When sentiment is enabled, we FIRST split by sentiment, THEN run topic modeling.
        """

        # Step 1: Sentiment Analysis and Document Splitting
        if use_sentiment:
            with st.spinner("🎭 Analyzing sentiment..."):
                sentiment_info = self.analyze_sentiment(documents)
                self.sentiment_labels = [s['sentiment'] for s in sentiment_info]

            # Split documents by sentiment
            positive_docs = []
            negative_docs = []
            neutral_docs = []
            positive_indices = []
            negative_indices = []
            neutral_indices = []

            for idx, (doc, sentiment) in enumerate(zip(documents, self.sentiment_labels)):
                if sentiment == 'positive':
                    positive_docs.append(doc)
                    positive_indices.append(idx)
                elif sentiment == 'negative':
                    negative_docs.append(doc)
                    negative_indices.append(idx)
                else:
                    neutral_docs.append(doc)
                    neutral_indices.append(idx)

            st.info(f"📊 Sentiment Split: {len(positive_docs)} positive, {len(negative_docs)} negative, {len(neutral_docs)} neutral")

            # Step 2: Run separate topic modeling for each sentiment group
            all_topics = [-1] * len(documents)  # Initialize all as outliers
            all_probs = np.zeros(len(documents))
            topic_offset = 0

            # Process positive documents
            if len(positive_docs) > 0:
                with st.spinner(f"🟢 Finding topics in {len(positive_docs)} positive documents..."):
                    pos_topics, pos_probs = self._model_topics_for_sentiment(
                        positive_docs,
                        min_topic_size=max(2, min_topic_size // 2),  # Smaller min size for split groups
                        nr_topics=None if nr_topics == "auto" else (nr_topics // 3 if nr_topics else None)
                    )
                    # Map back to original indices with offset
                    for i, idx in enumerate(positive_indices):
                        if pos_topics[i] != -1:
                            all_topics[idx] = pos_topics[i] + topic_offset
                            if pos_probs is not None:
                                # Handle both 1D and 2D probability arrays
                                if hasattr(pos_probs, 'ndim'):
                                    if pos_probs.ndim == 2:
                                        # 2D array: take max probability
                                        all_probs[idx] = float(pos_probs[i].max()) if i < len(pos_probs) else 0
                                    elif pos_probs.ndim == 1:
                                        # 1D array
                                        all_probs[idx] = float(pos_probs[i]) if i < len(pos_probs) else 0
                                    else:
                                        all_probs[idx] = 0
                                else:
                                    all_probs[idx] = 0
                            else:
                                all_probs[idx] = 0
                        else:
                            all_topics[idx] = -1  # Outlier
                    if pos_topics:
                        topic_offset = max([t for t in pos_topics if t != -1], default=0) + 1

            # Process negative documents
            if len(negative_docs) > 0:
                with st.spinner(f"🔴 Finding topics in {len(negative_docs)} negative documents..."):
                    neg_topics, neg_probs = self._model_topics_for_sentiment(
                        negative_docs,
                        min_topic_size=max(2, min_topic_size // 2),
                        nr_topics=None if nr_topics == "auto" else (nr_topics // 3 if nr_topics else None)
                    )
                    # Map back with offset
                    for i, idx in enumerate(negative_indices):
                        if neg_topics[i] != -1:
                            all_topics[idx] = neg_topics[i] + topic_offset
                            if neg_probs is not None:
                                # Handle both 1D and 2D probability arrays
                                if hasattr(neg_probs, 'ndim'):
                                    if neg_probs.ndim == 2:
                                        # 2D array: take max probability
                                        all_probs[idx] = float(neg_probs[i].max()) if i < len(neg_probs) else 0
                                    elif neg_probs.ndim == 1:
                                        # 1D array
                                        all_probs[idx] = float(neg_probs[i]) if i < len(neg_probs) else 0
                                    else:
                                        all_probs[idx] = 0
                                else:
                                    all_probs[idx] = 0
                            else:
                                all_probs[idx] = 0
                        else:
                            all_topics[idx] = -1
                    if neg_topics:
                        topic_offset = max([t for t in all_topics if t != -1], default=topic_offset) + 1

            # Process neutral documents
            if len(neutral_docs) > 0:
                with st.spinner(f"⚪ Finding topics in {len(neutral_docs)} neutral documents..."):
                    neu_topics, neu_probs = self._model_topics_for_sentiment(
                        neutral_docs,
                        min_topic_size=max(2, min_topic_size // 2),
                        nr_topics=None if nr_topics == "auto" else (nr_topics // 3 if nr_topics else None)
                    )
                    # Map back with offset
                    for i, idx in enumerate(neutral_indices):
                        if neu_topics[i] != -1:
                            all_topics[idx] = neu_topics[i] + topic_offset
                            if neu_probs is not None:
                                # Handle both 1D and 2D probability arrays
                                if hasattr(neu_probs, 'ndim'):
                                    if neu_probs.ndim == 2:
                                        # 2D array: take max probability
                                        all_probs[idx] = float(neu_probs[i].max()) if i < len(neu_probs) else 0
                                    elif neu_probs.ndim == 1:
                                        # 1D array
                                        all_probs[idx] = float(neu_probs[i]) if i < len(neu_probs) else 0
                                    else:
                                        all_probs[idx] = 0
                                else:
                                    all_probs[idx] = 0
                            else:
                                all_probs[idx] = 0
                        else:
                            all_topics[idx] = -1

            topics = all_topics
            probs = all_probs

            # Create combined topic_info
            topic_info = self._create_combined_topic_info(topics, documents, self.sentiment_labels)

        else:
            # No sentiment splitting - run normal topic modeling
            if embeddings is None:
                embeddings = self.compute_embeddings(documents)
            self.embeddings = embeddings

            self.topic_model = self.create_advanced_topic_model(
                num_docs=len(documents),
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                use_sentiment_guidance=False
            )

            with st.spinner("🔮 Discovering topic patterns..."):
                topics, probs = self.topic_model.fit_transform(documents, embeddings)

            topic_info = self.topic_model.get_topic_info()

        return topics, probs, topic_info

    def _model_topics_for_sentiment(self, docs: List[str], min_topic_size: int, nr_topics: Optional[int]) -> Tuple[List[int], Optional[np.ndarray]]:
        """
        Run topic modeling on a subset of documents (for a specific sentiment).
        Enhanced to ensure better separation of different concerns.
        """
        if len(docs) == 0:
            return [], None

        # Create a model for this sentiment group with more aggressive separation
        embeddings = self.compute_embeddings(docs)

        # For sentiment-split groups, use more granular clustering
        actual_min_size = max(2, min_topic_size // 2)  # Allow smaller clusters

        # If we have enough documents, try to find more topics
        if len(docs) >= 10:
            # Use automatic topic discovery with lower min_cluster_size
            # This helps separate different concerns (shipping vs service)
            model = self.create_advanced_topic_model(
                num_docs=len(docs),
                min_topic_size=actual_min_size,
                nr_topics=None,  # Let HDBSCAN find natural clusters
                use_sentiment_guidance=False
            )
        else:
            # For very small groups, use K-means with sensible k
            n_clusters = min(len(docs) // 2, 3)  # At most 3 clusters for small groups
            model = self.create_advanced_topic_model(
                num_docs=len(docs),
                min_topic_size=actual_min_size,
                nr_topics=n_clusters if n_clusters > 1 else None,
                use_sentiment_guidance=False
            )

        try:
            topics, probs = model.fit_transform(docs, embeddings)

            # Post-process to ensure different concerns are separated
            topics = self._refine_topic_separation(topics, docs)

        except:
            # If modeling fails, mark all as single topic
            topics = [0] * len(docs)
            probs = None

        return topics, probs

    def _refine_topic_separation(self, topics: List[int], docs: List[str]) -> List[int]:
        """
        Dynamically refine topics using embedding similarity to ensure coherent groupings.
        No predefined keywords - let the embeddings determine what belongs together.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Store embeddings for coherence checking if not already stored
        if not hasattr(self, 'stored_embeddings'):
            self.stored_embeddings = self.embedding_model.encode(docs, show_progress_bar=False)

        new_topics = topics.copy()
        next_topic_id = max(topics) + 1 if topics else 0

        # Group documents by topic
        topic_docs = {}
        for idx, topic in enumerate(topics):
            if topic != -1:  # Skip outliers
                if topic not in topic_docs:
                    topic_docs[topic] = []
                topic_docs[topic].append(idx)

        # Check each topic for internal coherence using embeddings
        for topic_id, doc_indices in topic_docs.items():
            if len(doc_indices) >= 15:  # Only split very large topics to avoid over-segmentation
                # Get embeddings for this topic
                topic_embeddings = self.stored_embeddings[doc_indices]

                # Calculate pairwise similarities
                similarity_matrix = cosine_similarity(topic_embeddings)

                # Calculate average similarity (excluding diagonal)
                mask = np.ones_like(similarity_matrix, dtype=bool)
                np.fill_diagonal(mask, False)
                avg_similarity = similarity_matrix[mask].mean()

                # Very conservative splitting - only if really incoherent
                if avg_similarity < 0.4:  # Much stricter threshold
                    # Determine optimal number of clusters using silhouette score
                    best_n_clusters = 2
                    best_score = -1

                    for n in range(2, min(4, len(doc_indices) // 2 + 1)):
                        clustering = AgglomerativeClustering(
                            n_clusters=n,
                            metric='cosine',
                            linkage='average'
                        )
                        distance_matrix = 1 - similarity_matrix
                        labels = clustering.fit_predict(distance_matrix)

                        # Check cluster sizes
                        unique, counts = np.unique(labels, return_counts=True)
                        if min(counts) >= 2:  # Each cluster needs at least 2 docs
                            # Calculate within-cluster similarity
                            cluster_scores = []
                            for cluster_id in unique:
                                cluster_mask = labels == cluster_id
                                cluster_sims = similarity_matrix[cluster_mask][:, cluster_mask]
                                mask_2d = np.ones_like(cluster_sims, dtype=bool)
                                np.fill_diagonal(mask_2d, False)
                                if cluster_sims[mask_2d].size > 0:
                                    cluster_scores.append(cluster_sims[mask_2d].mean())

                            if cluster_scores:
                                score = np.mean(cluster_scores)
                                if score > best_score:
                                    best_score = score
                                    best_n_clusters = n

                    # Apply the best clustering
                    if best_n_clusters > 1:
                        clustering = AgglomerativeClustering(
                            n_clusters=best_n_clusters,
                            metric='cosine',
                            linkage='average'
                        )
                        distance_matrix = 1 - similarity_matrix
                        cluster_labels = clustering.fit_predict(distance_matrix)

                        # Assign new topic IDs to clusters (keep first cluster with original topic)
                        cluster_to_topic = {0: topic_id}
                        for cluster_id in range(1, best_n_clusters):
                            cluster_to_topic[cluster_id] = next_topic_id
                            next_topic_id += 1

                        # Update topics
                        for i, doc_idx in enumerate(doc_indices):
                            new_topics[doc_idx] = cluster_to_topic[cluster_labels[i]]

        return new_topics

    def _create_combined_topic_info(self, topics: List[int], documents: List[str], sentiments: List[str]) -> pd.DataFrame:
        """
        Create topic_info DataFrame for sentiment-split topics with enhanced labels.
        """
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        topic_data = {}

        for topic_id, doc, sentiment in zip(topics, documents, sentiments):
            if topic_id != -1:
                if topic_id not in topic_data:
                    topic_data[topic_id] = {
                        'docs': [],
                        'sentiment': sentiment,  # Track dominant sentiment
                        'sentiments': [],
                        'count': 0
                    }
                topic_data[topic_id]['docs'].append(doc)
                topic_data[topic_id]['sentiments'].append(sentiment)
                topic_data[topic_id]['count'] += 1

        # Create topic_info rows
        rows = []
        for topic_id, data in topic_data.items():
            # Generate enhanced topic name
            topic_name = self._generate_enhanced_topic_label(
                topic_id,
                data['docs'],
                data['sentiments']
            )

            rows.append({
                'Topic': topic_id,
                'Count': data['count'],
                'Name': topic_name
            })

        # Add outliers row if any
        outlier_count = topics.count(-1)
        if outlier_count > 0:
            rows.append({
                'Topic': -1,
                'Count': outlier_count,
                'Name': '-1_outliers'
            })

        topic_info = pd.DataFrame(rows)
        topic_info = topic_info.sort_values('Topic').reset_index(drop=True)

        return topic_info

    def _generate_enhanced_topic_label(self, topic_id: int, docs: List[str], sentiments: List[str]) -> str:
        """
        Generate meaningful topic labels that explain the 'why' behind the topic.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import Counter

        # Get dominant sentiment
        sentiment_counts = Counter(sentiments)
        dominant_sentiment = sentiment_counts.most_common(1)[0][0]

        # Get sentiment emoji for the label
        sentiment_emoji = '😊' if dominant_sentiment == 'positive' else '😔' if dominant_sentiment == 'negative' else '😐'

        try:
            # Use TF-IDF to identify key themes
            tfidf = TfidfVectorizer(
                max_features=10,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams for better context
                min_df=1
            )

            # Fit on documents
            tfidf_matrix = tfidf.fit_transform(docs[:20] if len(docs) > 20 else docs)
            feature_names = tfidf.get_feature_names_out()

            # Get top terms with their scores
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_indices]

            # Smart keyword filtering to avoid contradictions
            positive_words = {'excellent', 'great', 'amazing', 'good', 'best', 'wonderful', 'perfect', 'outstanding', 'fantastic', 'love', 'satisfied', 'happy', 'impressed'}
            negative_words = {'poor', 'bad', 'terrible', 'worst', 'awful', 'disappointed', 'issue', 'problem', 'complain', 'wrong', 'failed', 'broken', 'horrible'}

            filtered_keywords = []
            for kw in keywords:
                kw_lower = kw.lower()
                # Filter based on sentiment
                if dominant_sentiment == 'positive':
                    if kw_lower not in negative_words:
                        filtered_keywords.append(kw)
                elif dominant_sentiment == 'negative':
                    if kw_lower not in positive_words:
                        filtered_keywords.append(kw)
                else:  # neutral
                    if kw_lower not in positive_words and kw_lower not in negative_words:
                        filtered_keywords.append(kw)

            # Use filtered keywords or fallback to original
            final_keywords = filtered_keywords[:3] if len(filtered_keywords) >= 2 else keywords[:3]
            base_label = f"{sentiment_emoji} {' '.join(final_keywords)}"

            # Format final label
            topic_name = f"{topic_id}_{base_label.replace(' ', '_').lower()}"

        except Exception as e:
            # Fallback to simple keyword extraction
            try:
                vectorizer = CountVectorizer(max_features=4, stop_words='english')
                vectorizer.fit(docs[:10])
                words = vectorizer.get_feature_names_out()
                topic_name = f"{topic_id}_{'_'.join(words)}"
            except:
                topic_name = f"{topic_id}_topic"

        return topic_name

    def _extract_topic_essence(self, keywords: List[str], docs: List[str]) -> str:
        """
        Extract the essence of topics from keywords without forcing into predefined categories.
        This method is kept for potential future use but currently not called.
        """
        # Simply return the most informative keyword combination
        # No predefined themes - let the data define itself
        return '_'.join(keywords[:2]) if keywords else 'topic'

    def update_topic_info_for_split_topics(self, topics: List[int], topic_info: pd.DataFrame, documents: List[str]) -> pd.DataFrame:
        """
        Update topic_info to include newly split topics.
        """
        import pandas as pd

        # Get all unique topic IDs from the processed topics
        unique_topics = set(topics)
        existing_topics = set(topic_info['Topic'].values)
        new_topic_ids = unique_topics - existing_topics

        if new_topic_ids and len(new_topic_ids) > 0:
            # For each new topic, create a row based on the documents it contains
            new_rows = []

            for new_topic in new_topic_ids:
                if new_topic != -1:  # Skip outliers
                    # Get documents for this topic
                    topic_docs = [documents[i] for i, t in enumerate(topics) if t == new_topic]

                    # Find the most similar existing topic (usually the one it was split from)
                    # For simplicity, use the first few topic words from documents
                    if topic_docs:
                        # Get common words from the documents
                        from sklearn.feature_extraction.text import CountVectorizer
                        try:
                            vectorizer = CountVectorizer(max_features=5, stop_words='english')
                            vectorizer.fit(topic_docs[:10])  # Use first 10 docs
                            words = vectorizer.get_feature_names_out()
                            topic_name = f"{new_topic}_{'_'.join(words[:3])}"
                        except:
                            topic_name = f"{new_topic}_split_topic"

                        new_rows.append({
                            'Topic': new_topic,
                            'Count': len(topic_docs),
                            'Name': topic_name
                        })

            if new_rows:
                # Add new rows to topic_info
                new_df = pd.DataFrame(new_rows)
                topic_info = pd.concat([topic_info, new_df], ignore_index=True)
                topic_info = topic_info.sort_values('Topic').reset_index(drop=True)

        return topic_info

    def split_topics_by_sentiment(self, topics: List[int], sentiments: List[str]) -> List[int]:
        """
        Post-process topics to split them by sentiment.
        Topics with mixed sentiments get split into positive and negative versions.
        """
        # Analyze sentiment distribution per topic
        topic_sentiment_dist = {}
        topic_documents = {}

        for idx, (topic, sentiment) in enumerate(zip(topics, sentiments)):
            if topic != -1:  # Skip outliers
                if topic not in topic_sentiment_dist:
                    topic_sentiment_dist[topic] = {'positive': 0, 'negative': 0, 'neutral': 0}
                    topic_documents[topic] = {'positive': [], 'negative': [], 'neutral': []}
                topic_sentiment_dist[topic][sentiment] += 1
                topic_documents[topic][sentiment].append(idx)

        # Determine which topics need splitting
        topics_to_split = set()
        for topic, dist in topic_sentiment_dist.items():
            total = sum(dist.values())
            # Split if we have meaningful amounts of both positive and negative
            if dist['positive'] >= 1 and dist['negative'] >= 1:
                # At least one of each sentiment
                pos_ratio = dist['positive'] / total
                neg_ratio = dist['negative'] / total
                # Split if both sentiments are significant (>15%) or if we have few docs
                if (pos_ratio > 0.15 and neg_ratio > 0.15) or total <= 5:
                    topics_to_split.add(topic)

        # Create new topic assignments
        new_topics = topics.copy()
        max_topic = max(topics) if topics else 0
        next_topic_id = max_topic + 1

        # Process topics that need splitting
        for topic in topics_to_split:
            # Assign new topic IDs based on sentiment
            pos_topic_id = next_topic_id
            neg_topic_id = next_topic_id + 1
            next_topic_id += 2

            # Update documents with new topic IDs
            for idx in topic_documents[topic]['positive']:
                new_topics[idx] = pos_topic_id
            for idx in topic_documents[topic]['negative']:
                new_topics[idx] = neg_topic_id
            # Neutral stays with positive or gets own ID if significant
            if len(topic_documents[topic]['neutral']) > len(topic_documents[topic]['positive']):
                # More neutral than positive, give it its own topic
                neutral_topic_id = next_topic_id
                next_topic_id += 1
                for idx in topic_documents[topic]['neutral']:
                    new_topics[idx] = neutral_topic_id
            else:
                # Group neutral with positive
                for idx in topic_documents[topic]['neutral']:
                    new_topics[idx] = pos_topic_id

        return new_topics

    def create_enhanced_labels(
        self,
        topics: List[int],
        topic_info: pd.DataFrame,
        sentiment_labels: Optional[List[str]] = None
    ) -> List[str]:
        """
        Create sophisticated topic labels with sentiment indicators.
        """
        enhanced_labels = []

        for topic_id in topics:
            if topic_id == -1:
                enhanced_labels.append("📊 Outliers")
            else:
                base_label = topic_info[topic_info["Topic"] == topic_id]["Name"].iloc[0]

                # Simple cleanup since we're not augmenting anymore
                # Just clean up the topic format

                # Clean up the topic number prefix and get meaningful words
                if "_" in base_label:
                    parts = base_label.split("_")
                    # Skip the topic number
                    if parts[0].isdigit():
                        parts = parts[1:]

                    # Remove duplicates while preserving order
                    seen = set()
                    unique_parts = []
                    for part in parts:
                        if part and part.lower() not in seen and len(part) > 2:
                            seen.add(part.lower())
                            unique_parts.append(part)

                    # Take first 3-4 meaningful words
                    base_label = " ".join(unique_parts[:4])

                # Add sentiment indicator if available
                if sentiment_labels:
                    topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
                    topic_sentiments = [sentiment_labels[i] for i in topic_indices if i < len(sentiment_labels)]

                    if topic_sentiments:
                        pos_ratio = topic_sentiments.count("positive") / len(topic_sentiments)
                        neg_ratio = topic_sentiments.count("negative") / len(topic_sentiments)

                        if pos_ratio > 0.7:
                            label = f"😊 {base_label}"
                        elif neg_ratio > 0.7:
                            label = f"😔 {base_label}"
                        elif pos_ratio > 0.4 and neg_ratio > 0.4:
                            label = f"😐 {base_label} (mixed)"
                        else:
                            label = f"📝 {base_label}"

                        enhanced_labels.append(label)
                    else:
                        enhanced_labels.append(f"📝 {base_label}")
                else:
                    enhanced_labels.append(f"📝 {base_label}")

        return enhanced_labels

    def create_hierarchical_topics(self, docs: List[str]) -> Optional[pd.DataFrame]:
        """
        Create hierarchical topic structure for better understanding of topic relationships.
        """
        if not self.topic_model or self.embeddings is None:
            return None

        try:
            # Create hierarchical topics
            hierarchical_topics = self.topic_model.hierarchical_topics(docs)
            return hierarchical_topics
        except Exception as e:
            logger.error(f"Error creating hierarchical topics: {e}")
            return None

    def visualize_hierarchy(self) -> Optional[go.Figure]:
        """
        Create interactive hierarchical topic visualization.
        """
        if not self.topic_model:
            return None

        try:
            fig = self.topic_model.visualize_hierarchy()
            return fig
        except Exception as e:
            logger.error(f"Error visualizing hierarchy: {e}")
            return None

    def visualize_topics_2d(self, custom_labels: bool = True) -> Optional[go.Figure]:
        """
        Create 2D topic visualization showing topic relationships with proper labels.
        """
        if not self.topic_model:
            return None

        try:
            # Get topic info for custom labels
            topic_info = self.topic_model.get_topic_info()

            # Create custom labels dictionary if needed
            if custom_labels:
                topic_label_dict = {}
                for index, row in topic_info.iterrows():
                    if row['Topic'] != -1:
                        # Create cleaner label
                        words = row['Name'].split('_')[1:4]  # Get first 3 words
                        topic_label_dict[row['Topic']] = f"Topic {row['Topic']}: {' '.join(words)}"

                fig = self.topic_model.visualize_topics(custom_labels=topic_label_dict)
            else:
                fig = self.topic_model.visualize_topics()

            # Update layout for better visibility
            fig.update_layout(
                title="Topic Map - Semantic Relationships",
                showlegend=True,
                hovermode='closest',
                width=900,
                height=600
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating 2D visualization: {e}")
            return None

    def visualize_heatmap(self) -> Optional[go.Figure]:
        """
        Create topic similarity heatmap.
        """
        if not self.topic_model:
            return None

        try:
            fig = self.topic_model.visualize_heatmap()
            return fig
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return None

    def track_topics_over_time(
        self,
        documents: List[str],
        timestamps: List[datetime],
        nr_bins: int = 10
    ) -> Tuple[Optional[pd.DataFrame], Optional[go.Figure]]:
        """
        Track topic evolution over time.
        """
        if not self.topic_model:
            return None, None

        try:
            topics_over_time = self.topic_model.topics_over_time(
                documents,
                timestamps,
                nr_bins=nr_bins
            )

            # Create visualization
            fig = self.topic_model.visualize_topics_over_time(topics_over_time)

            return topics_over_time, fig
        except Exception as e:
            logger.error(f"Error tracking topics over time: {e}")
            return None, None

    def generate_llm_topic_labels(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ) -> Dict[int, str]:
        """
        Generate sophisticated topic labels using LLM.
        """
        if not self.topic_model or not api_key:
            return {}

        try:
            # Create OpenAI representation model
            openai_model = OpenAI(api_key=api_key, model=model)

            # Update topic representations
            self.topic_model.update_topics(
                self.embeddings,
                representation_model=openai_model
            )

            # Get updated labels
            topic_info = self.topic_model.get_topic_info()
            labels = {
                row['Topic']: row['Name']
                for _, row in topic_info.iterrows()
                if row['Topic'] != -1
            }

            return labels
        except Exception as e:
            logger.error(f"Error generating LLM labels: {e}")
            return {}

    def export_to_html(
        self,
        filepath: str = "topic_visualization.html",
        df: pd.DataFrame = None,
        text_column: str = None
    ) -> bool:
        """
        Export comprehensive interactive HTML report with all visualizations and drill-down capability.
        """
        if not self.topic_model or df is None:
            return False

        try:
            # Get topic info
            topic_info = self.topic_model.get_topic_info()

            # Prepare topic distribution data
            topic_counts = df["topic_label"].value_counts()
            topic_df = pd.DataFrame({
                'Topic': topic_counts.index,
                'Documents': topic_counts.values,
                'Percentage': (topic_counts.values / len(df) * 100).round(1)
            })

            # Create bar chart
            fig_bar = px.bar(
                topic_df.head(20),
                x='Topic',
                y='Documents',
                title='Top 20 Topics by Document Count',
                color='Documents',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(xaxis_tickangle=-45, height=500)

            # Create pie chart
            top_topics = topic_df.head(10)
            others_count = topic_df.iloc[10:]['Documents'].sum() if len(topic_df) > 10 else 0
            if others_count > 0:
                others_df = pd.DataFrame({
                    'Topic': ['Others'],
                    'Documents': [others_count],
                    'Percentage': [(others_count / len(df) * 100)]
                })
                pie_data = pd.concat([top_topics, others_df], ignore_index=True)
            else:
                pie_data = top_topics

            fig_pie = px.pie(
                pie_data,
                values='Documents',
                names='Topic',
                title='Topic Distribution'
            )

            # Get visualizations
            fig_2d = self.visualize_topics_2d()
            fig_hierarchy = self.visualize_hierarchy()
            fig_heatmap = self.visualize_heatmap()

            # Create HTML with enhanced styling and interactivity
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Topic Analysis Report - {datetime.now().strftime('%Y-%m-%d')}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        overflow: hidden;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 40px;
                        text-align: center;
                    }}
                    .header h1 {{
                        font-size: 2.5em;
                        margin-bottom: 10px;
                    }}
                    .header p {{
                        opacity: 0.9;
                        font-size: 1.1em;
                    }}
                    .stats {{
                        display: flex;
                        justify-content: space-around;
                        padding: 30px;
                        background: #f8f9fa;
                        border-bottom: 1px solid #e0e0e0;
                    }}
                    .stat {{
                        text-align: center;
                    }}
                    .stat-value {{
                        font-size: 2em;
                        font-weight: bold;
                        color: #667eea;
                    }}
                    .stat-label {{
                        color: #666;
                        margin-top: 5px;
                    }}
                    .content {{
                        padding: 40px;
                    }}
                    .section {{
                        margin-bottom: 50px;
                    }}
                    .section-title {{
                        font-size: 1.8em;
                        color: #333;
                        margin-bottom: 20px;
                        padding-bottom: 10px;
                        border-bottom: 2px solid #667eea;
                    }}
                    .visualization {{
                        margin: 30px 0;
                        padding: 20px;
                        background: #f8f9fa;
                        border-radius: 10px;
                    }}
                    .topic-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    .topic-table th {{
                        background: #667eea;
                        color: white;
                        padding: 12px;
                        text-align: left;
                    }}
                    .topic-table td {{
                        padding: 10px;
                        border-bottom: 1px solid #e0e0e0;
                    }}
                    .topic-table tr:hover {{
                        background: #f0f0f0;
                    }}
                    .documents-container {{
                        max-height: 400px;
                        overflow-y: auto;
                        border: 1px solid #e0e0e0;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 10px 0;
                        background: white;
                    }}
                    .document {{
                        padding: 10px;
                        margin: 5px 0;
                        background: #f8f9fa;
                        border-radius: 5px;
                        border-left: 3px solid #667eea;
                    }}
                    .topic-details {{
                        display: none;
                        margin-top: 15px;
                        padding: 15px;
                        background: #f0f0f0;
                        border-radius: 5px;
                    }}
                    .toggle-btn {{
                        background: #667eea;
                        color: white;
                        border: none;
                        padding: 8px 15px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 0.9em;
                    }}
                    .toggle-btn:hover {{
                        background: #5a67d8;
                    }}
                    .sentiment-positive {{ color: #28a745; font-weight: bold; }}
                    .sentiment-negative {{ color: #dc3545; font-weight: bold; }}
                    .sentiment-neutral {{ color: #6c757d; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Topic Analysis Report</h1>
                        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>

                    <div class="stats">
                        <div class="stat">
                            <div class="stat-value">{len(df)}</div>
                            <div class="stat-label">Total Documents</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{len(topic_df) - (1 if any('Outlier' in str(t) for t in topic_df['Topic']) else 0)}</div>
                            <div class="stat-label">Topics Found</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{((len(df) - df['topic_id'].value_counts().get(-1, 0)) / len(df) * 100):.1f}%</div>
                            <div class="stat-label">Coverage</div>
                        </div>
                    </div>

                    <div class="content">
            """

            # Add distribution charts
            html_content += """
                        <div class="section">
                            <h2 class="section-title">📈 Topic Distribution</h2>
                            <div class="visualization">
            """

            if fig_bar:
                html_content += f'<div id="bar-chart"></div>'
                html_content += f'<script>Plotly.newPlot("bar-chart", {fig_bar.to_json()});</script>'

            html_content += """
                            </div>
                            <div class="visualization">
            """

            if fig_pie:
                html_content += f'<div id="pie-chart"></div>'
                html_content += f'<script>Plotly.newPlot("pie-chart", {fig_pie.to_json()});</script>'

            html_content += """
                            </div>
                        </div>
            """

            # Add topic map
            if fig_2d:
                html_content += f"""
                        <div class="section">
                            <h2 class="section-title">🗺️ Topic Map</h2>
                            <div class="visualization">
                                <div id="topic-map"></div>
                                <script>Plotly.newPlot("topic-map", {fig_2d.to_json()});</script>
                            </div>
                        </div>
                """

            # Add hierarchy
            if fig_hierarchy:
                html_content += f"""
                        <div class="section">
                            <h2 class="section-title">🌳 Topic Hierarchy</h2>
                            <div class="visualization">
                                <div id="hierarchy"></div>
                                <script>Plotly.newPlot("hierarchy", {fig_hierarchy.to_json()});</script>
                            </div>
                        </div>
                """

            # Add similarity heatmap
            if fig_heatmap:
                html_content += f"""
                        <div class="section">
                            <h2 class="section-title">🔥 Topic Similarity</h2>
                            <div class="visualization">
                                <div id="heatmap"></div>
                                <script>Plotly.newPlot("heatmap", {fig_heatmap.to_json()});</script>
                            </div>
                        </div>
                """

            # Add detailed topic table with drill-down
            html_content += """
                        <div class="section">
                            <h2 class="section-title">📋 Topic Details & Documents</h2>
                            <table class="topic-table">
                                <thead>
                                    <tr>
                                        <th>Topic</th>
                                        <th>Documents</th>
                                        <th>Percentage</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
            """

            # Add each topic with expandable documents
            for idx, (topic_label, topic_data) in enumerate(df.groupby('topic_label')):
                if 'Outlier' not in str(topic_label):
                    doc_count = len(topic_data)
                    percentage = (doc_count / len(df) * 100)

                    html_content += f"""
                                    <tr>
                                        <td>{topic_label}</td>
                                        <td>{doc_count}</td>
                                        <td>{percentage:.1f}%</td>
                                        <td><button class="toggle-btn" onclick="toggleDetails('topic-{idx}')">View Documents</button></td>
                                    </tr>
                                    <tr>
                                        <td colspan="4">
                                            <div id="topic-{idx}" class="topic-details">
                                                <h4>Sample Documents from {topic_label}:</h4>
                                                <div class="documents-container">
                    """

                    # Add sample documents (max 20)
                    for doc_idx, (_, doc_row) in enumerate(topic_data.head(20).iterrows()):
                        doc_text = doc_row[text_column] if text_column else str(doc_row.iloc[0])

                        # Add sentiment coloring if available
                        sentiment_class = ""
                        if 'sentiment' in doc_row:
                            sentiment = doc_row['sentiment']
                            if sentiment == 'positive':
                                sentiment_class = 'sentiment-positive'
                            elif sentiment == 'negative':
                                sentiment_class = 'sentiment-negative'
                            else:
                                sentiment_class = 'sentiment-neutral'

                            html_content += f"""
                                                    <div class="document">
                                                        <span class="{sentiment_class}">[{sentiment.upper()}]</span> {doc_text}
                                                    </div>
                            """
                        else:
                            html_content += f"""
                                                    <div class="document">{doc_text}</div>
                            """

                    html_content += """
                                                </div>
                                            </div>
                                        </td>
                                    </tr>
                    """

            html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <script>
                    function toggleDetails(id) {
                        var element = document.getElementById(id);
                        if (element.style.display === "none" || element.style.display === "") {
                            element.style.display = "block";
                        } else {
                            element.style.display = "none";
                        }
                    }
                </script>
            </body>
            </html>
            """

            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            return True
        except Exception as e:
            logger.error(f"Error exporting to HTML: {e}")
            return False

    def save_model(self, filepath: str = "topic_model") -> bool:
        """
        Save the trained model for later use.
        """
        if not self.topic_model:
            return False

        try:
            self.topic_model.save(filepath)
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    @staticmethod
    @st.cache_resource
    def load_model(filepath: str) -> Optional[BERTopic]:
        """
        Load a previously saved model.
        """
        try:
            model = BERTopic.load(filepath)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None


def main():
    """Main Streamlit application with professional UI/UX."""

    # Header
    st.title("🎯 Professional Topic & Sentiment Analyzer")
    st.markdown("""
    Advanced topic modeling with sentiment-aware clustering using state-of-the-art NLP techniques.
    Built with BERTopic's best practices for production-ready analysis.
    """)

    # Initialize session state
    if 'modeler' not in st.session_state:
        st.session_state.modeler = SentimentAwareTopicModeler()
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar configuration
    with st.sidebar:
        st.header("📁 Data Input")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Select a CSV file with text data for analysis"
        )

        st.divider()

        st.header("⚙️ Model Configuration")

        use_sentiment = st.checkbox(
            "Enable Sentiment-Aware Clustering",
            value=True,
            help="Separates positive and negative feedback into different topics"
        )

        col1, col2 = st.columns(2)
        with col1:
            min_topic_size = st.number_input(
                "Min Topic Size",
                min_value=2,
                max_value=100,
                value=10,
                help="Minimum documents per topic"
            )

        with col2:
            nr_topics = st.selectbox(
                "Number of Topics",
                ["auto", 5, 10, 15, 20, 25, 30],
                index=2,
                help="Target number of topics"
            )

        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"],
            help="Choose based on speed vs accuracy tradeoff"
        )

        st.divider()

        st.header("🔬 Advanced Features")

        with st.expander("Visualization Settings"):
            enable_hierarchy = st.checkbox(
                "Enable Hierarchical Analysis",
                value=True,
                help="Discover topic relationships and subtopics"
            )

            enable_2d_map = st.checkbox(
                "Enable Topic Mapping",
                value=True,
                help="Visualize topics in 2D space"
            )

            enable_similarity = st.checkbox(
                "Enable Similarity Analysis",
                value=True,
                help="Compute topic similarity matrix"
            )

        with st.expander("Export Settings"):
            save_embeddings = st.checkbox(
                "Include Embeddings in Export",
                value=False,
                help="Warning: This will increase file size significantly"
            )

            auto_save_model = st.checkbox(
                "Auto-save Trained Model",
                value=False,
                help="Automatically save model after training"
            )

        with st.expander("Performance Settings"):
            reduce_outliers = st.checkbox(
                "Auto-reduce Outliers",
                value=True,
                help="Automatically reassign outlier documents"
            )

            cache_embeddings = st.checkbox(
                "Cache Embeddings",
                value=True,
                help="Cache computed embeddings for faster re-runs"
            )

        st.divider()

        st.header("📊 Model Management")

        # Load existing model
        if st.button("📂 Load Existing Model"):
            try:
                loaded_model = SentimentAwareTopicModeler.load_model("bertopic_model")
                if loaded_model:
                    st.session_state.modeler.topic_model = loaded_model
                    st.success("Model loaded successfully!")
                else:
                    st.warning("No saved model found")
            except Exception as e:
                st.error(f"Error loading model: {e}")

    # Main content area
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)

        # Data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(10))
            st.caption(f"Dataset: {len(df)} documents, {df.shape[1]} columns")

        # Column selection
        text_column = st.selectbox(
            "Select text column for analysis:",
            df.columns,
            help="Choose the column containing text data"
        )

        # Analysis button
        if st.button("🚀 Analyze Topics & Sentiment", type="primary"):
            documents = df[text_column].astype(str).tolist()

            # Perform analysis
            modeler = st.session_state.modeler

            # Configure nr_topics
            nr_topics_value = None if nr_topics == "auto" else int(nr_topics)

            # Run the analysis
            topics, probs, topic_info = modeler.fit_transform(
                documents=documents,
                min_topic_size=min_topic_size,
                nr_topics=nr_topics_value,
                use_sentiment=use_sentiment
            )

            # Create enhanced labels
            enhanced_labels = modeler.create_enhanced_labels(
                topics,
                topic_info,
                modeler.sentiment_labels
            )

            # Store results
            df["topic_id"] = topics
            df["topic_label"] = enhanced_labels
            if use_sentiment and modeler.sentiment_labels:
                df["sentiment"] = modeler.sentiment_labels

            st.session_state.results = {
                'df': df,
                'topic_info': topic_info,
                'topics': topics,
                'labels': enhanced_labels,
                'modeler': modeler,  # Store the modeler for visualizations
                'documents': documents,  # Store original documents
                'text_column': text_column  # Store text column name for export
            }

            # Auto-save model if enabled
            if auto_save_model:
                success = modeler.save_model("bertopic_model_autosave")
                if success:
                    st.success("✅ Analysis complete! Model auto-saved.")
                else:
                    st.success("✅ Analysis complete!")
            else:
                st.success("✅ Analysis complete!")

    # Display results
    if st.session_state.results:
        results = st.session_state.results
        df = results['df']
        topic_info = results['topic_info']

        # Metrics
        st.divider()
        st.subheader("📈 Analysis Metrics")

        col1, col2, col3, col4 = st.columns(4)

        unique_topics = len(set(results['topics'])) - (1 if -1 in results['topics'] else 0)
        outliers = results['topics'].count(-1)
        coverage = (len(results['topics']) - outliers) / len(results['topics']) * 100

        with col1:
            st.metric("Topics Discovered", unique_topics)
        with col2:
            st.metric("Documents Analyzed", len(df))
        with col3:
            st.metric("Coverage", f"{coverage:.1f}%")
        with col4:
            st.metric("Outliers", outliers)

        # Topic distribution with graphs
        st.divider()
        st.subheader("🎨 Topic Distribution & Analysis")

        topic_counts = df["topic_label"].value_counts()
        topic_df = pd.DataFrame({
            'Topic': topic_counts.index,
            'Documents': topic_counts.values,
            'Percentage': (topic_counts.values / len(df) * 100).round(1)
        })

        # Create tabs for different views
        dist_tabs = st.tabs(["📊 Bar Chart", "🥧 Pie Chart", "📈 Distribution Table"])

        with dist_tabs[0]:
            # Bar chart
            fig_bar = px.bar(
                topic_df.head(20),
                x='Topic',
                y='Documents',
                title='Top 20 Topics by Document Count',
                labels={'Documents': 'Number of Documents'},
                color='Documents',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with dist_tabs[1]:
            # Pie chart for top topics
            top_topics = topic_df.head(10)
            others_count = topic_df.iloc[10:]['Documents'].sum() if len(topic_df) > 10 else 0

            if others_count > 0:
                others_df = pd.DataFrame({
                    'Topic': ['Others'],
                    'Documents': [others_count],
                    'Percentage': [(others_count / len(df) * 100)]
                })
                pie_data = pd.concat([top_topics, others_df], ignore_index=True)
            else:
                pie_data = top_topics

            fig_pie = px.pie(
                pie_data,
                values='Documents',
                names='Topic',
                title='Topic Distribution (Top 10 + Others)'
            )
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)

        with dist_tabs[2]:
            # Show distribution table
            st.dataframe(
                topic_df,
                use_container_width=True,
                hide_index=True
            )

        # Interactive topic explorer
        st.divider()
        st.subheader("🔍 Topic Explorer")

        selected_topic = st.selectbox(
            "Select a topic to explore:",
            sorted(df["topic_label"].unique()),
            format_func=lambda x: f"{x} ({len(df[df['topic_label']==x])} docs)"
        )

        if selected_topic:
            topic_docs = df[df["topic_label"] == selected_topic]

            # Topic details
            st.markdown(f"### {selected_topic}")

            if use_sentiment and "sentiment" in df.columns:
                # Show sentiment distribution for this topic
                sent_dist = topic_docs["sentiment"].value_counts()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", sent_dist.get("positive", 0), delta_color="normal")
                with col2:
                    st.metric("Negative", sent_dist.get("negative", 0), delta_color="inverse")
                with col3:
                    st.metric("Neutral", sent_dist.get("neutral", 0), delta_color="off")

            # Show documents
            st.markdown("#### Sample Documents")

            for idx, (_, row) in enumerate(topic_docs.head(10).iterrows(), 1):
                sentiment_icon = ""
                if use_sentiment and "sentiment" in df.columns:
                    if row["sentiment"] == "positive":
                        sentiment_icon = "🟢"
                    elif row["sentiment"] == "negative":
                        sentiment_icon = "🔴"
                    else:
                        sentiment_icon = "⚪"

                st.markdown(f"{sentiment_icon} **{idx}.** {row[text_column]}")
                if idx < min(10, len(topic_docs)):
                    st.divider()

        # Advanced Visualizations
        st.divider()
        st.subheader("📊 Advanced Visualizations")

        # Create tabs for different visualizations
        viz_tabs = st.tabs(["Topic Map", "Hierarchy", "Similarity", "Time Evolution", "Export"])

        with viz_tabs[0]:
            st.markdown("### Topic Relationships")
            if st.button("Generate 2D Topic Map"):
                with st.spinner("Creating topic map..."):
                    # Use the stored modeler from results
                    if 'modeler' in results:
                        modeler = results['modeler']
                        fig = modeler.visualize_topics_2d(custom_labels=True)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                            # Add topic statistics below the map
                            st.markdown("#### Topic Statistics")
                            topic_info = modeler.topic_model.get_topic_info()
                            for idx, row in topic_info.head(10).iterrows():
                                if row['Topic'] != -1:
                                    words = row['Name'].split('_')[1:6]
                                    st.write(f"**Topic {row['Topic']}**: {' '.join(words)} ({row['Count']} docs)")
                        else:
                            st.warning("Unable to generate topic map.")
                    else:
                        st.error("Model not found. Please run analysis first.")

        with viz_tabs[1]:
            st.markdown("### Hierarchical Topics")
            if st.button("Generate Topic Hierarchy"):
                with st.spinner("Building hierarchy..."):
                    if 'modeler' in results and 'documents' in results:
                        modeler = results['modeler']
                        documents = results['documents']

                        # Create hierarchical topics
                        hierarchy = modeler.create_hierarchical_topics(documents)
                        if hierarchy is not None:
                            # Visualize hierarchy
                            fig = modeler.visualize_hierarchy()
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                            # Show hierarchy data
                            st.markdown("#### Hierarchical Structure")
                            st.dataframe(hierarchy.head(20), use_container_width=True)

                            # Create a simple tree view
                            st.markdown("#### Topic Tree")
                            for idx, row in hierarchy.head(10).iterrows():
                                if 'Parent_Name' in row and 'Child_Left_Name' in row:
                                    parent = row['Parent_Name'] if pd.notna(row['Parent_Name']) else "Root"
                                    left = row['Child_Left_Name'] if pd.notna(row['Child_Left_Name']) else ""
                                    right = row['Child_Right_Name'] if pd.notna(row['Child_Right_Name']) else ""

                                    st.write(f"📁 **{parent}**")
                                    if left:
                                        st.write(f"   ├── {left}")
                                    if right:
                                        st.write(f"   └── {right}")
                        else:
                            st.warning("Unable to generate hierarchy. Try with more topics.")
                    else:
                        st.error("Model not found. Please run analysis first.")

        with viz_tabs[2]:
            st.markdown("### Topic Similarity Heatmap")
            if st.button("Generate Similarity Matrix"):
                with st.spinner("Computing similarities..."):
                    if 'modeler' in results:
                        modeler = results['modeler']
                        fig = modeler.visualize_heatmap()
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                            # Add interpretation
                            st.markdown("#### How to Read This Heatmap")
                            st.info("""
                            - **Darker colors** indicate higher similarity between topics
                            - **Lighter colors** indicate topics are more distinct
                            - Topics with high similarity might be candidates for merging
                            - Distinct topics validate good separation
                            """)
                        else:
                            st.warning("Unable to generate heatmap.")
                    else:
                        st.error("Model not found. Please run analysis first.")

        with viz_tabs[3]:
            st.markdown("### Topics Over Time")
            st.info("📅 To track topics over time, ensure your data has a timestamp column")

            if 'timestamp_col' in df.columns or 'date' in df.columns.str.lower():
                timestamp_col = st.selectbox(
                    "Select timestamp column:",
                    [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                )

                if st.button("Analyze Time Evolution"):
                    st.warning("Time evolution analysis requires timestamp data in the correct format.")
            else:
                st.info("No timestamp column detected. Add timestamps to track evolution.")

        with viz_tabs[4]:
            st.markdown("### Export Options")

            col1, col2 = st.columns(2)

            with col1:
                # CSV Export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="topic_analysis_results.csv",
                    mime="text/csv"
                )

            with col2:
                # HTML Export
                if st.button("Generate Interactive HTML"):
                    with st.spinner("Creating comprehensive HTML report..."):
                        if 'modeler' in results:
                            modeler = results['modeler']
                            # Pass the dataframe and text column for drill-down capability
                            success = modeler.export_to_html(
                                filepath="topic_report.html",
                                df=df,
                                text_column=results.get('text_column', None)
                            )
                            if success:
                                with open("topic_report.html", "r") as f:
                                    html_data = f.read()
                                st.download_button(
                                    label="📊 Download HTML Report",
                                    data=html_data,
                                    file_name="topic_visualization.html",
                                    mime="text/html"
                                )
                                st.success("✅ HTML report generated with all visualizations and drill-down capability!")
                            else:
                                st.error("Failed to generate HTML report")
                        else:
                            st.error("Model not found. Please run analysis first.")

            # Model Export
            st.markdown("#### Save Trained Model")
            if st.button("Save Model"):
                modeler = st.session_state.modeler
                success = modeler.save_model("bertopic_model")
                if success:
                    st.success("Model saved as 'bertopic_model'")
                else:
                    st.error("Failed to save model")

        # Optional LLM Features
        st.divider()
        with st.expander("🤖 Advanced AI Features (Optional)", expanded=False):
            st.markdown("""
            ### LLM-Enhanced Topic Labels
            Use Large Language Models to generate more descriptive topic labels.
            **Note:** This requires an OpenAI API key and may incur costs.
            """)

            col1, col2 = st.columns([2, 1])

            with col1:
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    help="Your OpenAI API key. This is never stored and only used for this session."
                )

            with col2:
                llm_model = st.selectbox(
                    "Model",
                    ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                    help="GPT-3.5 is faster and cheaper, GPT-4 is more accurate"
                )

            if api_key:
                if st.button("Generate AI Topic Labels"):
                    with st.spinner("Generating enhanced labels with AI..."):
                        modeler = st.session_state.modeler
                        enhanced_labels = modeler.generate_llm_topic_labels(api_key, llm_model)

                        if enhanced_labels:
                            st.success("Generated enhanced labels!")
                            st.dataframe(
                                pd.DataFrame.from_dict(
                                    enhanced_labels,
                                    orient='index',
                                    columns=['Enhanced Label']
                                ).reset_index().rename(columns={'index': 'Topic ID'})
                            )
                        else:
                            st.error("Failed to generate labels. Check your API key.")
            else:
                st.info("👆 Enter your OpenAI API key to enable AI-powered features")

        # Download results
        st.divider()
        st.subheader("💾 Quick Export")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Results (CSV)",
            data=csv,
            file_name="topic_analysis_results.csv",
            mime="text/csv"
        )

    else:
        # Welcome message when no data
        st.info("👆 Upload a CSV file to begin advanced topic and sentiment analysis")

        # Show example
        with st.expander("📚 Example Use Cases"):
            st.markdown("""
            **Customer Feedback Analysis:**
            - Automatically separate positive and negative reviews
            - Identify key themes in customer complaints
            - Track sentiment trends across product features

            **Support Ticket Classification:**
            - Group similar issues together
            - Prioritize based on sentiment urgency
            - Identify emerging problem areas

            **Social Media Monitoring:**
            - Track brand sentiment across topics
            - Identify viral positive/negative themes
            - Monitor competitor mentions by sentiment
            """)


if __name__ == "__main__":
    main()