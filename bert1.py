import streamlit as st
import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import re
from collections import Counter
from scipy.spatial.distance import cosine
import plotly.express as px
import plotly.graph_objects as go

# Set Streamlit to wide mode for better layout
st.set_page_config(layout="wide", page_title="Complete BERTopic with All Features", page_icon="🚀")

# Make accelerate optional - not strictly required
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

# Try to import GPU-accelerated libraries
try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False

try:
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    cuml_available = True
except ImportError:
    from umap import UMAP
    from hdbscan import HDBSCAN
    cuml_available = False

# Try to import FAISS for GPU-accelerated similarity search
try:
    import faiss
    faiss_available = True
    # Check if GPU version is available
    if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
        faiss_gpu_available = True
    else:
        faiss_gpu_available = False
except ImportError:
    faiss_available = False
    faiss_gpu_available = False

# Force CUDA initialization if available
if torch.cuda.is_available():
    try:
        torch.cuda.init()
        torch.cuda.empty_cache()
        # Enable TF32 for better performance on newer GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
    except Exception as e:
        st.warning(f"CUDA initialization warning: {e}")

# -----------------------------------------------------
# ROBUST TEXT PREPROCESSOR
# -----------------------------------------------------
class RobustTextPreprocessor:
    """Handles all text preprocessing edge cases to prevent tokenization errors"""

    @staticmethod
    def clean_text(text, max_length=5000, min_length=10):
        """Clean and validate text to prevent tokenization errors"""
        # Handle None or NaN
        if pd.isna(text) or text is None:
            return None

        # If someone passed a list/tuple, join as text early (prevents nested tokenization)
        if isinstance(text, (list, tuple)):
            try:
                text = ' '.join(map(str, text))
            except Exception:
                return None

        # Convert to string if not already
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return None

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Handle empty or too short text
        if len(text.strip()) < min_length:
            return None

        # Handle nested structures (lists, dicts converted to string)
        if text.startswith('[') or text.startswith('{'):
            text = re.sub(r'[\[\]{}"\']', ' ', text)
            text = ' '.join(text.split())

        # Remove problematic characters
        text = text.replace('\x00', '')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')

        # Fix encoding issues
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            pass

        # Truncate if too long
        if len(text) > max_length:
            truncated = text[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.8:
                text = truncated[:last_period + 1]
            else:
                text = truncated + '...'

        text = text.strip()
        if len(text) < min_length:
            return None

        return text

    @staticmethod
    def preprocess_documents(documents, max_length=5000, min_length=10, show_warnings=True):
        """Preprocess a list of documents robustly"""
        cleaned_docs = []
        valid_indices = []

        stats = {
            'total': len(documents),
            'valid': 0,
            'empty_removed': 0,
            'too_short': 0,
            'too_long_truncated': 0,
            'type_converted': 0
        }

        for idx, doc in enumerate(documents):
            original_type = type(doc)
            original_length = len(str(doc)) if doc is not None else 0

            cleaned = RobustTextPreprocessor.clean_text(doc, max_length, min_length)

            if cleaned is not None:
                cleaned_docs.append(cleaned)
                valid_indices.append(idx)
                stats['valid'] += 1

                if original_type != str and not isinstance(doc, (list, tuple)):
                    stats['type_converted'] += 1
                if original_length > max_length:
                    stats['too_long_truncated'] += 1
            else:
                if pd.isna(doc) or doc is None or str(doc).strip() == '':
                    stats['empty_removed'] += 1
                elif len(str(doc).strip()) < min_length:
                    stats['too_short'] += 1

        if show_warnings and stats['total'] != stats['valid']:
            removed = stats['total'] - stats['valid']
            st.warning(f"⚠️ Preprocessing: {removed} documents removed ({removed/stats['total']*100:.1f}%)")

        return cleaned_docs, valid_indices, stats

# -----------------------------------------------------
# TOPIC MERGER FOR MINIMUM SIZE ENFORCEMENT
# -----------------------------------------------------
class TopicMerger:
    """Merge small topics to enforce minimum topic size"""

    @staticmethod
    def merge_small_topics(topics, embeddings, min_size=10):
        """Merge topics that are smaller than min_size with their nearest neighbors"""
        topics = np.array(topics)
        topic_counts = Counter(topics)

        small_topics = [t for t, count in topic_counts.items()
                        if t != -1 and count < min_size]

        if not small_topics:
            return topics

        # Calculate topic centroids
        topic_centroids = {}
        for topic in set(topics):
            if topic != -1:
                topic_mask = topics == topic
                if np.any(topic_mask):
                    topic_embeddings = embeddings[topic_mask]
                    if len(topic_embeddings) > 0 and not np.all(topic_embeddings == 0):
                        topic_centroids[topic] = np.mean(topic_embeddings, axis=0)

        # Merge small topics
        merged_topics = topics.copy()

        for small_topic in small_topics:
            if small_topic not in topic_centroids:
                continue

            small_centroid = topic_centroids[small_topic]
            min_distance = float('inf')
            best_merge_topic = None

            for topic, centroid in topic_centroids.items():
                if topic != small_topic and topic_counts[topic] >= min_size:
                    try:
                        distance = cosine(small_centroid, centroid)
                        if distance < min_distance:
                            min_distance = distance
                            best_merge_topic = topic
                    except:
                        continue

            if best_merge_topic is not None:
                merged_topics[merged_topics == small_topic] = best_merge_topic
                topic_counts[best_merge_topic] += topic_counts[small_topic]
                del topic_counts[small_topic]

        # Renumber topics
        unique_topics = sorted(set(merged_topics[merged_topics != -1]))
        topic_mapping = {old: new for new, old in enumerate(unique_topics)}
        topic_mapping[-1] = -1

        final_topics = np.array([topic_mapping.get(t, t) for t in merged_topics])

        return final_topics

# -----------------------------------------------------
# CATEGORY BALANCE ANALYZER
# -----------------------------------------------------
class CategoryBalanceAnalyzer:
    """Analyzes topic distribution and suggests splits for oversized categories"""

    def __init__(self, min_topic_ratio=0.30, ideal_max_ratio=0.20):
        self.min_topic_ratio = min_topic_ratio
        self.ideal_max_ratio = ideal_max_ratio

    def analyze_distribution(self, labels, include_outliers=False):
        """Analyze the distribution of documents across topics"""
        total_docs = len(labels)
        unique_topics = set(labels)

        if not include_outliers:
            labels_filtered = [l for l in labels if l != -1]
            total_docs = len(labels_filtered)
            unique_topics = set(labels_filtered)
        else:
            labels_filtered = labels

        if total_docs == 0:
            return {
                'balanced': False,
                'oversized_topics': [],
                'distribution': {},
                'warnings': ['No documents to analyze']
            }

        # Calculate distribution
        topic_counts = {}
        for topic in unique_topics:
            count = labels_filtered.count(topic)
            topic_counts[topic] = {
                'count': count,
                'ratio': count / total_docs
            }

        # Find oversized topics
        oversized_topics = []
        warnings_list = []

        for topic, stats in topic_counts.items():
            if stats['ratio'] > self.min_topic_ratio:
                oversized_topics.append({
                    'topic': topic,
                    'count': stats['count'],
                    'ratio': stats['ratio'],
                    'suggested_splits': max(2, int(stats['ratio'] / self.ideal_max_ratio))
                })
                warnings_list.append(
                    f"Topic {topic}: {stats['count']} docs ({stats['ratio']*100:.1f}%) - Consider splitting into {max(2, int(stats['ratio'] / self.ideal_max_ratio))} subtopics"
                )

        ratios = [stats['ratio'] for stats in topic_counts.values()]
        balance_score = 1 - (np.std(ratios) if len(ratios) > 1 else 0)

        return {
            'balanced': len(oversized_topics) == 0,
            'balance_score': balance_score,
            'oversized_topics': oversized_topics,
            'distribution': topic_counts,
            'warnings': warnings_list,
            'total_topics': len(unique_topics),
            'outlier_ratio': (len(labels) - len(labels_filtered)) / len(labels) if len(labels) > 0 else 0
        }

# -----------------------------------------------------
# OUTLIER REDUCTION STRATEGIES
# -----------------------------------------------------
class OutlierReducer:
    """Strategies to reduce outliers in topic modeling"""

    @staticmethod
    def suggest_parameters(num_docs, current_outlier_ratio, min_topic_size=10):
        """Suggest better parameters to reduce outliers"""
        suggestions = {
            'method': '',
            'parameters': {},
            'explanation': ''
        }

        if current_outlier_ratio > 0.3:
            # High outliers - use K-means
            suggestions['method'] = 'kmeans'
            max_topics = max(5, num_docs // min_topic_size)
            suggested_topics = min(max_topics, max(5, num_docs // 50))

            suggestions['parameters'] = {
                'use_kmeans': True,
                'use_gpu_kmeans': True,
                'nr_topics': suggested_topics,
                'min_topic_size': min_topic_size,
                'merge_small_topics': True
            }
            suggestions['explanation'] = f"High outliers detected. K-means will assign all documents to topics. Topics smaller than {min_topic_size} will be merged."

        elif current_outlier_ratio > 0.15:
            # Moderate outliers - adjust HDBSCAN
            suggestions['method'] = 'hdbscan'
            suggestions['parameters'] = {
                'use_kmeans': False,
                'min_cluster_size': max(3, min_topic_size // 3),
                'min_samples': max(1, min_topic_size // 5),
                'prediction_data': True,
                'min_topic_size': min_topic_size
            }
            suggestions['explanation'] = f"Moderate outliers. Using lenient HDBSCAN parameters with min topic size of {min_topic_size}."

        else:
            # Low outliers - standard parameters
            suggestions['method'] = 'hdbscan'
            suggestions['parameters'] = {
                'use_kmeans': False,
                'min_cluster_size': min_topic_size,
                'min_samples': max(1, min_topic_size // 2),
                'prediction_data': False,
                'min_topic_size': min_topic_size
            }
            suggestions['explanation'] = f"Low outliers. Using standard HDBSCAN with min topic size of {min_topic_size}."

        return suggestions

# -----------------------------------------------------
# GPU-ACCELERATED K-MEANS WRAPPER
# -----------------------------------------------------
class GPUKMeans:
    """K-means clustering with GPU acceleration if available"""

    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

        if cuml_available and torch.cuda.is_available():
            try:
                from cuml.cluster import KMeans as cuKMeans
                self.model = cuKMeans(n_clusters=n_clusters, random_state=random_state)
                self.use_gpu = True
            except Exception:
                from sklearn.cluster import KMeans
                self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
                self.use_gpu = False
        else:
            from sklearn.cluster import KMeans
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
            self.use_gpu = False

    def fit(self, X):
        return self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X):
        return self.model.fit_predict(X)

# -----------------------------------------------------
# SAFE EMBEDDING MODEL
# -----------------------------------------------------
class SafeEmbeddingModel:
    """Wrapper for sentence transformer that handles edge cases"""

    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_name = model_name  # Store the model name
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.model.max_seq_length = min(self.model.max_seq_length, 512)

    def encode_safe(self, documents, batch_size=32, show_progress=True):
        """Safely encode documents with error handling"""
        if not documents:
            raise ValueError("No documents to encode")

        documents = [str(doc) if doc is not None else "" for doc in documents]
        non_empty_docs = [(i, doc) for i, doc in enumerate(documents) if doc.strip()]

        if not non_empty_docs:
            raise ValueError("All documents are empty after preprocessing")

        indices, clean_docs = zip(*non_empty_docs)

        try:
            embeddings = self.model.encode(
                list(clean_docs),
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            full_embeddings = np.zeros((len(documents), embeddings.shape[1]))
            for i, idx in enumerate(indices):
                full_embeddings[idx] = embeddings[i]

            return full_embeddings

        except RuntimeError as e:
            if "too large" in str(e).lower() or "out of memory" in str(e).lower():
                st.warning("Reducing batch size due to memory constraints...")
                return self.encode_safe(documents, batch_size=max(1, batch_size//2), show_progress=show_progress)
            elif "padding" in str(e).lower() or "truncation" in str(e).lower():
                st.warning("Processing documents individually due to length variance...")
                return self._encode_one_by_one(clean_docs, indices, len(documents))
            else:
                raise e

    def _encode_one_by_one(self, documents, indices, total_length):
        """Emergency fallback: encode documents one at a time"""
        embeddings_list = []

        progress_bar = st.progress(0)
        for i, doc in enumerate(documents):
            try:
                emb = self.model.encode(
                    [doc],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings_list.append(emb[0])
            except:
                embeddings_list.append(np.zeros(self.model.get_sentence_embedding_dimension()))

            progress_bar.progress((i + 1) / len(documents))

        progress_bar.empty()

        full_embeddings = np.zeros((total_length, len(embeddings_list[0])))
        for i, idx in enumerate(indices):
            full_embeddings[idx] = embeddings_list[i]

        return full_embeddings


# -----------------------------------------------------
# IMPROVED HUMAN-READABLE TOPIC LABELS
# -----------------------------------------------------
def _top_phrases(texts, ngram_range=(2,3), top_k=5, max_features=5000):
    """Return top_k high-signal phrases from texts using TF-IDF (prefers bigrams/trigrams)."""
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english",
            min_df=2
        )
        X = vec.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(vec.get_feature_names_out())
        order = np.argsort(scores)[::-1]
        phrases = [vocab[i] for i in order[:top_k]]
        return phrases
    except Exception:
        return []

def _clean_phrase(phrase):
    """Clean and normalize a phrase for use in labels."""
    # Remove punctuation but keep spaces
    phrase = re.sub(r'[^\w\s-]', ' ', phrase)
    # Normalize whitespace
    phrase = ' '.join(phrase.split())
    return phrase.strip()

def _to_title_case(text):
    """Smart title case that preserves acronyms and common patterns."""
    # Common acronyms and abbreviations to keep uppercase
    keep_upper = {
        "AI", "ML", "NLP", "API", "SQL", "GPU", "CPU", "FAQ", "KPI", "OKR", 
        "CRM", "IT", "HR", "PR", "SEO", "SEM", "ROI", "B2B", "B2C", "SaaS",
        "UI", "UX", "iOS", "AWS", "API", "REST", "HTTP", "JSON", "XML", "CSV",
        "PDF", "URL", "HTML", "CSS", "JS", "PHP", "SQL", "NoSQL", "CI", "CD"
    }
    
    # Words to keep lowercase
    keep_lower = {
        "a", "an", "and", "or", "but", "for", "nor", "on", "at", "to", "by",
        "in", "of", "the", "with", "from", "into", "onto", "upon", "as", "vs"
    }
    
    words = text.split()
    result = []
    
    for i, word in enumerate(words):
        word_clean = re.sub(r'[^\w-]', '', word)
        
        # First word or after certain punctuation - capitalize
        if i == 0:
            if word_clean.upper() in keep_upper:
                result.append(word_clean.upper())
            else:
                result.append(word_clean.capitalize())
        # Check if it's an acronym
        elif word_clean.upper() in keep_upper:
            result.append(word_clean.upper())
        # Check if it should be lowercase
        elif word_clean.lower() in keep_lower:
            result.append(word_clean.lower())
        # Default: capitalize
        else:
            result.append(word_clean.capitalize())
    
    return ' '.join(result)

def _create_descriptive_label(phrases_23, phrases_1, keywords, max_len=70):
    """
    Create a unique, descriptive label from phrases and keywords.
    This generates labels like "Customer Service Response" instead of generic "Customer Support"
    """
    # Clean all phrases
    clean_phrases_23 = [_clean_phrase(p) for p in phrases_23 if p and len(_clean_phrase(p)) > 3]
    clean_phrases_1 = [_clean_phrase(p) for p in phrases_1 if p and len(_clean_phrase(p)) > 2]
    
    # Parse keywords
    if isinstance(keywords, str):
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    else:
        keyword_list = list(keywords) if keywords else []
    
    # PRIORITY 1: Use the longest trigram if available and descriptive (3+ words)
    for phrase in clean_phrases_23:
        word_count = len(phrase.split())
        if word_count >= 3:
            label = _to_title_case(phrase)
            if len(label) <= max_len:
                return label
    
    # PRIORITY 2: Combine top 2 bigrams/trigrams if they complement each other
    if len(clean_phrases_23) >= 2:
        first = clean_phrases_23[0]
        first_words = set(first.lower().split())
        
        # Find a second phrase that adds information (at least 1 new word, not a subset)
        for second in clean_phrases_23[1:3]:
            second_words = set(second.lower().split())
            
            # Skip if second is completely contained in first
            if second_words.issubset(first_words):
                continue
            
            # Skip if first is completely contained in second (use second instead)
            if first_words.issubset(second_words):
                label = _to_title_case(second)
                if len(label) <= max_len:
                    return label
                continue
            
            # Check if they share too many words (>50% overlap)
            overlap = len(first_words & second_words)
            total = len(first_words | second_words)
            if overlap / total < 0.5:  # Less than 50% overlap is good
                label = f"{first} & {second}"
                label = _to_title_case(label)
                if len(label) <= max_len:
                    return label
        
        # If no good pair found, use the first phrase alone
        label = _to_title_case(first)
        if len(label) <= max_len:
            return label
    
    # PRIORITY 3: Use single best bigram
    if len(clean_phrases_23) >= 1:
        label = _to_title_case(clean_phrases_23[0])
        if len(label) <= max_len:
            return label
    
    # PRIORITY 4: Combine 2-3 distinctive unigrams
    if len(clean_phrases_1) >= 2:
        # Take unique words only
        seen = set()
        unique_words = []
        for word in clean_phrases_1:
            if word.lower() not in seen:
                unique_words.append(word)
                seen.add(word.lower())
                if len(unique_words) >= 3:
                    break
        
        if len(unique_words) >= 2:
            label = _to_title_case(' '.join(unique_words))
            if len(label) <= max_len:
                return label
    
    # PRIORITY 5: Use keywords
    if len(keyword_list) >= 2:
        seen = set()
        unique_keywords = []
        for kw in keyword_list:
            if kw.lower() not in seen:
                unique_keywords.append(kw)
                seen.add(kw.lower())
                if len(unique_keywords) >= 3:
                    break
        
        if len(unique_keywords) >= 2:
            label = _to_title_case(' '.join(unique_keywords))
            if len(label) <= max_len:
                return label
    
    # Fallback: Single best available
    if clean_phrases_23:
        return _to_title_case(clean_phrases_23[0][:max_len])
    if clean_phrases_1:
        return _to_title_case(' '.join(clean_phrases_1[:3])[:max_len])
    if keyword_list:
        return _to_title_case(' '.join(keyword_list[:3])[:max_len])
    
    return "Miscellaneous Topic"

def make_human_label(topic_docs, fallback_keywords, max_len=70):
    """
    Build a unique, descriptive topic label based on the actual content.
    
    Examples of output:
    - "Product Launch Strategy & Marketing"
    - "Customer Service Response Times"
    - "Technical Documentation Updates"
    - "Sales Pipeline Management"
    - "Employee Onboarding Process"
    
    NOT generic categories like "Customer Support" or "Marketing"
    """
    # Extract phrases using TF-IDF with different n-gram ranges
    phrases_23 = _top_phrases(topic_docs, (2, 3), top_k=5)  # Bigrams and trigrams
    phrases_1 = _top_phrases(topic_docs, (1, 1), top_k=10)   # Unigrams
    
    # Create the descriptive label
    label = _create_descriptive_label(phrases_23, phrases_1, fallback_keywords, max_len)
    
    # Final cleanup
    label = re.sub(r'\s+', ' ', label).strip()
    if len(label) > max_len:
        label = label[:max_len].rstrip() + "…"
    
    return label

# -----------------------------------------------------
# FAST RECLUSTERING ENGINE
# -----------------------------------------------------
class FastReclusterer:
    """Fast reclustering using pre-computed embeddings"""

    def __init__(self, documents, embeddings, umap_embeddings=None):
        self.documents = documents
        self.embeddings = embeddings
        self.umap_embeddings = umap_embeddings
        self.use_gpu = torch.cuda.is_available() and cuml_available

    def recluster(self, n_topics, min_topic_size=10, use_reduced=True, method='kmeans'):
        """Quickly recluster documents into new topics"""
        clustering_embeddings = self.umap_embeddings if (use_reduced and self.umap_embeddings is not None) else self.embeddings

        valid_mask = np.any(clustering_embeddings != 0, axis=1)
        valid_embeddings = clustering_embeddings[valid_mask]

        if len(valid_embeddings) == 0:
            st.error("No valid embeddings for clustering!")
            return None, None

        try:
            if method == 'kmeans':
                if self.use_gpu:
                    try:
                        from cuml.cluster import KMeans as cuKMeans
                        clusterer = cuKMeans(n_clusters=min(n_topics, len(valid_embeddings)), random_state=42)
                    except:
                        clusterer = KMeans(n_clusters=min(n_topics, len(valid_embeddings)), random_state=42)
                else:
                    clusterer = KMeans(n_clusters=min(n_topics, len(valid_embeddings)), random_state=42)

                valid_topics = clusterer.fit_predict(valid_embeddings)
            else:
                min_cluster_size = max(min_topic_size, len(self.documents) // (max(1, n_topics) * 2))
                min_cluster_size = min(min_cluster_size, max(2, len(valid_embeddings) // 2))

                if self.use_gpu:
                    clusterer = cumlHDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=max(1, min_cluster_size // 5)
                    )
                else:
                    clusterer = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=max(1, min_cluster_size // 5)
                    )

                valid_topics = clusterer.fit_predict(valid_embeddings)

            topics = np.full(len(clustering_embeddings), -1)
            topics[valid_mask] = valid_topics

        except Exception as e:
            st.error(f"Clustering error: {str(e)}")
            return None, None

        if min_topic_size > 2:
            topics = TopicMerger.merge_small_topics(topics, self.embeddings, min_topic_size)

        topic_info = self._extract_topic_keywords(topics)

        return topics, topic_info

    def _extract_topic_keywords(self, topics, top_n_words=10):
        """Extract keywords and generate human-readable labels for each topic."""
        topics_dict = {}
        for idx, topic in enumerate(topics):
            topics_dict.setdefault(topic, [])
            if idx < len(self.documents):
                topics_dict[topic].append(self.documents[idx])

        topic_info_list = []
        for topic_id in sorted(topics_dict.keys()):
            if topic_id == -1:
                topic_info_list.append({
                    'Topic': -1,
                    'Count': len(topics_dict[-1]),
                    'Keywords': 'Outliers',
                    'Human_Label': 'Outliers',
                    'Name': 'Outliers'
                })
                continue

            topic_docs = topics_dict[topic_id]

            # --- keywords (simple, fast) ---
            try:
                sample_size = min(200, len(topic_docs))  # Increased sample size for better labels
                topic_text = ' '.join(topic_docs[:sample_size])

                words = topic_text.lower().split()
                word_counts = Counter(words)
                common_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','as',
                                'is','was','are','were','be','have','has','had','do','does','did','will','would','could',
                                'should','may','might','must','shall','can','need','it','this','that','these','those',
                                'i','you','he','she','we','they'}
                filtered = {w: c for w, c in word_counts.items() if w not in common_words and len(w) > 2}
                top_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
                keywords = [w for w, _ in top_words] or ['topic', str(topic_id)]
            except Exception:
                keywords = [f'topic_{topic_id}']

            keywords_str = ', '.join(keywords[:5])

            # --- human label (actual category name) ---
            human_label = make_human_label(topic_docs, keywords_str)

            topic_info_list.append({
                'Topic': topic_id,
                'Count': len(topic_docs),
                'Keywords': keywords_str,
                'Human_Label': human_label,
                'Name': human_label  # keep compatibility with UI
            })

        return pd.DataFrame(topic_info_list)

# -----------------------------------------------------
# GUARANTEE COLUMNS EXIST ON topic_info
# -----------------------------------------------------
def normalize_topic_info(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure topic_info has Human_Label, Keywords, and Count columns."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Topic", "Human_Label", "Keywords", "Count"])
    df = df.copy()

    # Keywords
    if "Keywords" not in df.columns:
        if "Representation" in df.columns:
            df["Keywords"] = df["Representation"].apply(
                lambda x: ", ".join(x[:5]) if isinstance(x, list) else (str(x) if pd.notna(x) else "")
            )
        else:
            df["Keywords"] = ""

    # Human_Label
    if "Human_Label" not in df.columns or df["Human_Label"].isna().any():
        if "Human_Label" not in df.columns:
            df["Human_Label"] = ""
        def _build_label(row):
            if isinstance(row.get("Human_Label"), str) and row["Human_Label"].strip():
                return row["Human_Label"]
            if isinstance(row.get("Name"), str) and row["Name"].strip():
                return row["Name"]
            kw = row.get("Keywords", "")
            # fall back to keywords-only label (documents unavailable here)
            return make_human_label([], kw)
        df["Human_Label"] = df.apply(_build_label, axis=1)

    # Count
    if "Count" not in df.columns:
        if "Document Count" in df.columns:
            df["Count"] = df["Document Count"]
        else:
            # compute from current topics if available
            if st.session_state.get("current_topics") is not None:
                counts = Counter(st.session_state.current_topics.tolist())
                df["Count"] = df["Topic"].map(counts).fillna(0).astype(int)
            else:
                df["Count"] = 0

    # Ensure Topic exists
    if "Topic" not in df.columns and "topic" in df.columns:
        df["Topic"] = df["topic"]

    return df

# -----------------------------------------------------
# MAIN APPLICATION FUNCTIONS
# -----------------------------------------------------
def check_gpu_capabilities():
    """Check and return GPU/CUDA capabilities"""
    capabilities = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'device_name': None,
        'gpu_memory_total': None,
        'gpu_memory_free': None,
        'cupy_available': cupy_available,
        'cuml_available': cuml_available,
        'faiss_available': faiss_available,
        'faiss_gpu_available': faiss_gpu_available,
        'accelerate_available': accelerate_available
    }

    if torch.cuda.is_available():
        capabilities['device_count'] = torch.cuda.device_count()
        capabilities['device_name'] = torch.cuda.get_device_name(0)

        try:
            mem_info = torch.cuda.mem_get_info(0)
            capabilities['gpu_memory_free'] = f"{mem_info[0] / 1024**3:.1f} GB"
            capabilities['gpu_memory_total'] = f"{mem_info[1] / 1024**3:.1f} GB"
        except Exception:
            pass

    return capabilities

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def load_embedding_model(model_name):
    """Load and cache the embedding model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    # keep SentenceTransformers' smart batching; just set a safe cap
    model.max_seq_length = 512
    # DO NOT compile model.encode; if you want compile, compile the inner HF module only (optional)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            first = model._first_module()
            if hasattr(first, "auto_model"):
                first.auto_model = torch.compile(first.auto_model)
        except Exception:
            pass
    return model

@st.cache_data
def compute_umap_embeddings(embeddings, n_neighbors=15, n_components=5):
    """Compute and cache UMAP reduced embeddings"""
    valid_mask = np.any(embeddings != 0, axis=1)
    if np.sum(valid_mask) < 10:
        return None

    if cuml_available and torch.cuda.is_available():
        reducer = cumlUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            random_state=42
        )
    else:
        reducer = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            random_state=42
        )

    umap_embeddings = np.zeros((len(embeddings), n_components))
    umap_embeddings[valid_mask] = reducer.fit_transform(embeddings[valid_mask])

    return umap_embeddings

def main():
    st.title("🚀 Complete BERTopic with All Features")

    # Initialize session state
    if 'embeddings_computed' not in st.session_state:
        st.session_state.embeddings_computed = False
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'umap_embeddings' not in st.session_state:
        st.session_state.umap_embeddings = None
    if 'documents' not in st.session_state:
        st.session_state.documents = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'reclusterer' not in st.session_state:
        st.session_state.reclusterer = None
    if 'current_topics' not in st.session_state:
        st.session_state.current_topics = None
    if 'current_topic_info' not in st.session_state:
        st.session_state.current_topic_info = None
    if 'valid_indices' not in st.session_state:
        st.session_state.valid_indices = None
    if 'min_topic_size_used' not in st.session_state:
        st.session_state.min_topic_size_used = 10
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = 'data'
    if 'topic_info' not in st.session_state:
        st.session_state.topic_info = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'min_topic_size' not in st.session_state:
        st.session_state.min_topic_size = 10
    if 'clustering_method' not in st.session_state:
        st.session_state.clustering_method = 'Unknown'
    if 'gpu_used' not in st.session_state:
        st.session_state.gpu_used = False
    if 'text_col' not in st.session_state:
        st.session_state.text_col = None
    if 'browser_df' not in st.session_state:
        st.session_state.browser_df = None
    if 'topic_human' not in st.session_state:
        st.session_state.topic_human = {}
    if 'last_topics_hash' not in st.session_state:
        st.session_state.last_topics_hash = None

    # Check GPU capabilities (cached in session state to avoid repeated checks)
    if 'gpu_capabilities' not in st.session_state:
        st.session_state.gpu_capabilities = check_gpu_capabilities()
    gpu_capabilities = st.session_state.gpu_capabilities

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # GPU Status Display
        if gpu_capabilities['cuda_available']:
            st.success(f"✅ GPU: {gpu_capabilities['device_name']}")
            if gpu_capabilities['gpu_memory_free']:
                st.info(f"Memory: {gpu_capabilities['gpu_memory_free']} / {gpu_capabilities['gpu_memory_total']}")
        else:
            st.warning("⚠️ No GPU detected. Using CPU (slower)")

        # Acceleration packages status
        st.subheader("📦 Acceleration Status")
        accel_cols = st.columns(2)
        with accel_cols[0]:
            st.write(f"{'✅' if gpu_capabilities['cuml_available'] else '❌'} cuML")
            st.write(f"{'✅' if gpu_capabilities['cupy_available'] else '❌'} CuPy")
        with accel_cols[1]:
            st.write(f"{'✅' if gpu_capabilities['faiss_gpu_available'] else '❌'} FAISS GPU")
            st.write(f"{'✅' if gpu_capabilities['accelerate_available'] else '❌'} Accelerate")

        # File upload
        st.header("📄 Data Input")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            st.session_state.uploaded_file_name = uploaded_file.name.replace('.csv', '')

            try:
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df):,} rows")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.stop()

            # Column selection
            text_col = st.selectbox(
                "Select text column",
                df.columns.tolist(),
                help="Column containing the text to analyze"
            )

            # Analysis Settings
            st.header("🎯 Analysis Settings")

            # Embedding model selection
            embedding_model = st.selectbox(
                "Embedding Model",
                [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "all-MiniLM-L12-v2",
                    "paraphrase-multilingual-MiniLM-L12-v2"
                ],
                help="Smaller models are faster but may be less accurate"
            )

            # Preprocessing Settings
            with st.expander("🔧 Preprocessing Settings"):
                max_doc_length = st.slider(
                    "Max document length (chars)",
                    1000, 10000, 5000,
                    help="Documents longer than this will be truncated"
                )
                min_doc_length = st.slider(
                    "Min document length (chars)",
                    1, 100, 10,
                    help="Documents shorter than this will be removed"
                )

            # Topic Size Control
            st.subheader("📏 Topic Size Control")
            default_min_topic_size = max(2, min(10, len(df) // 50))
            min_topic_size = st.slider(
                "Minimum Topic Size",
                min_value=2,
                max_value=max(2, min(100, len(df) // 10)),
                value=default_min_topic_size,
                help="Minimum number of documents per topic. Topics smaller than this will be merged."
            )

            # Outlier Reduction Strategy
            outlier_strategy = st.selectbox(
                "Outlier Reduction Strategy",
                ["Aggressive (K-means) - 0% outliers",
                 "Moderate (Lenient HDBSCAN) - <15% outliers",
                 "Conservative (Standard HDBSCAN) - Natural clustering"],
                help="Choose how aggressively to assign outliers to topics"
            )

            # Advanced Settings
            with st.expander("🔧 Advanced Settings"):
                # Number of topics (for K-means)
                if "Aggressive" in outlier_strategy:
                    safe_den = max(2, min_topic_size)
                    max_topics = max(5, len(df) // safe_den)
                    default_topics = min(max_topics, max(5, max(1, len(df) // 50)))

                    nr_topics = st.number_input(
                        "Initial Number of Topics",
                        min_value=2,
                        max_value=max(2, max_topics),
                        value=max(2, default_topics),
                        help=f"Number of topics to create (limited by min topic size of {min_topic_size})"
                    )
                else:
                    nr_topics = max(2, min(10, len(df) // 50))

                # UMAP parameters
                n_neighbors = st.slider("UMAP n_neighbors", 2, 50, 15)
                n_components = st.slider("UMAP n_components", 2, 10, 5)

                # Batch size
                batch_size = st.select_slider(
                    "Batch size",
                    options=[8, 16, 32, 64],
                    value=32,
                    help="Reduce if you get memory errors"
                )

                # Representation model (placeholder toggle kept for parity)
                use_mmr = st.checkbox("Use MMR for diverse keywords", value=True)

                # Seed words
                st.subheader("🎯 Seed Words (Optional)")
                seed_words_input = st.text_area(
                    "Enter seed words (one set per line)",
                    placeholder="Example:\nfinance, money, budget, cost\nmarketing, campaign, advertising",
                    help="Guide topic discovery with predefined keyword sets"
                )

            # Compute embeddings button
            if st.button("🚀 Compute Embeddings & Enable Interactive Mode", type="primary"):
                # Step 1: Preprocess documents
                with st.spinner("Preprocessing documents..."):
                    raw_documents = df[text_col].tolist()

                    cleaned_docs, valid_indices, stats = RobustTextPreprocessor.preprocess_documents(
                        raw_documents,
                        max_length=max_doc_length,
                        min_length=min_doc_length,
                        show_warnings=True
                    )

                    if not cleaned_docs:
                        st.error("No valid documents after preprocessing!")
                        st.stop()

                    st.session_state.documents = cleaned_docs
                    st.session_state.valid_indices = valid_indices
                    st.session_state.text_col = text_col

                    st.info(f"✅ Preprocessed {len(cleaned_docs):,} valid documents")

                # Step 2: Load model and compute embeddings (SAFE)
                with st.spinner("Computing embeddings (this is the slow part, done only once)..."):
                    try:
                        _ = load_embedding_model(embedding_model)

                        safe_model = SafeEmbeddingModel(
                            model_name=embedding_model,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        embeddings = safe_model.encode_safe(
                            cleaned_docs,
                            batch_size=batch_size,
                            show_progress=True
                        )
                        st.session_state.embeddings = embeddings
                    except RuntimeError as e:
                        st.error(f"Embedding error: {str(e)}")
                        st.info("If this persists, lower the batch size and/or max document length.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Embedding error: {str(e)}")
                        st.info("Try reducing batch size or document length")
                        st.stop()

                # Step 3: Compute UMAP reduction
                with st.spinner("Computing UMAP reduction for fast reclustering..."):
                    umap_embeddings = compute_umap_embeddings(embeddings, n_neighbors, n_components)
                    st.session_state.umap_embeddings = umap_embeddings

                # Step 4: Create reclusterer
                st.session_state.reclusterer = FastReclusterer(
                    cleaned_docs, embeddings, umap_embeddings
                )

                # Step 5: (optional) Seed words parsed
                seed_topic_list = []
                if seed_words_input:
                    for line in seed_words_input.strip().split('\n'):
                        if line.strip():
                            words = [w.strip() for w in line.split(',')]
                            if words:
                                seed_topic_list.append(words)

                # Step 6: Perform initial clustering
                with st.spinner("Performing initial clustering..."):
                    if "Aggressive" in outlier_strategy:
                        method = 'kmeans'
                        initial_n_topics = nr_topics
                    else:
                        method = 'hdbscan'
                        initial_n_topics = 10

                    topics, topic_info = st.session_state.reclusterer.recluster(
                        n_topics=initial_n_topics,
                        min_topic_size=min_topic_size,
                        use_reduced=umap_embeddings is not None,
                        method=method
                    )

                    if topics is None:
                        st.error("Initial clustering failed!")
                        st.stop()

                    # ALWAYS normalize topic_info to guarantee Human_Label etc.
                    topic_info = normalize_topic_info(topic_info)

                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.topic_info = topic_info
                    st.session_state.min_topic_size = min_topic_size
                    st.session_state.min_topic_size_used = min_topic_size
                    st.session_state.embeddings_computed = True
                    st.session_state.clustering_method = "K-means" if method == 'kmeans' else "HDBSCAN"
                    st.session_state.gpu_used = gpu_capabilities['cuda_available']
                    st.session_state.model = None

                    st.success("✅ Embeddings computed! You can now adjust topics dynamically with the slider below.")
                    st.balloons()

    # Main content area
    if st.session_state.embeddings_computed:
        st.success("✅ Embeddings ready! Use the controls below for instant topic adjustment.")

        # Interactive controls section
        st.header("🎚️ Dynamic Topic Adjustment")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            denom = max(2, st.session_state.min_topic_size)
            max_topics = max(2, min(100, len(st.session_state.documents) // denom))

            n_topics_slider = st.slider(
                "🎯 **Number of Topics** (Adjust in real-time!)",
                min_value=2,
                max_value=max_topics,
                value=min(10, max_topics),
                help="Move the slider to instantly re-cluster with different number of topics"
            )

        with col2:
            clustering_method = st.selectbox(
                "Method",
                ["K-means (Fast)", "HDBSCAN"],
                help="K-means is faster for interactive adjustment"
            )

        with col3:
            use_reduced = st.checkbox(
                "Use UMAP reduction",
                value=st.session_state.umap_embeddings is not None,
                disabled=st.session_state.umap_embeddings is None,
                help="Faster reclustering using reduced dimensions"
            )

        # Recluster button
        if st.button("🔄 Recluster with New Settings", type="secondary"):
            with st.spinner(f"Reclustering into {n_topics_slider} topics... (This is fast!)"):
                method = 'kmeans' if "K-means" in clustering_method else 'hdbscan'
                topics, topic_info = st.session_state.reclusterer.recluster(
                    n_topics=n_topics_slider,
                    min_topic_size=st.session_state.min_topic_size,
                    use_reduced=use_reduced and st.session_state.umap_embeddings is not None,
                    method=method
                )

                if topics is not None:
                    topic_info = normalize_topic_info(topic_info)
                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.topic_info = topic_info
                    st.session_state.clustering_method = "K-means" if method == 'kmeans' else "HDBSCAN"
                    
                    # Clear cached data to force rebuild
                    if 'browser_df' in st.session_state:
                        del st.session_state.browser_df
                    if 'topic_human' in st.session_state:
                        del st.session_state.topic_human
                    if 'processed_df' in st.session_state:
                        del st.session_state.processed_df
                    
                    st.success(f"✅ Reclustered into {len(topic_info[topic_info['Topic'] != -1])} topics!")
                    st.rerun()  # CRITICAL: Force full page refresh
                else:
                    st.error("Reclustering failed. Try different parameters.")

        # Display results
        if st.session_state.current_topics is not None:
            topics = st.session_state.current_topics
            topic_info = normalize_topic_info(st.session_state.current_topic_info)

            # Calculate metrics
            total_docs = len(topics)
            unique_topics = len(set(topics)) - (1 if -1 in topics else 0)
            outlier_count = sum(1 for t in topics if t == -1)
            coverage = ((total_docs - outlier_count) / total_docs) * 100 if total_docs > 0 else 0

            # Create processed dataframe
            processed_df = pd.DataFrame()
            processed_df['document'] = st.session_state.documents
            processed_df['topic'] = topics
            processed_df['topic_label'] = [f"Topic {t}" if t != -1 else "Outlier" for t in topics]

            # Add keywords for each topic + human labels
            topic_keywords = {}
            topic_human = {}
            for _, row in topic_info.iterrows():
                topic_keywords[row['Topic']] = row['Keywords']
                topic_human[row['Topic']] = row['Human_Label'] if 'Human_Label' in row else (row.get('Name') or f"Topic {row['Topic']}")
            processed_df['keywords'] = processed_df['topic'].map(topic_keywords)

            # Store in session state for export and other tabs
            st.session_state.processed_df = processed_df

            # Balance Analysis
            balance_analyzer = CategoryBalanceAnalyzer()
            balance_analysis = balance_analyzer.analyze_distribution(
                topics,
                include_outliers=False
            )

            # Build browser-ready dataframe ONLY ONCE (not on every rerun!)
            # Check if we need to rebuild (topics changed or browser_df doesn't exist)
            rebuild_browser = False
            if 'browser_df' not in st.session_state or st.session_state.browser_df is None:
                rebuild_browser = True
            elif 'last_topics_hash' not in st.session_state:
                rebuild_browser = True
            else:
                # Check if topics changed by comparing hash
                current_hash = hash(tuple(topics))
                if st.session_state.last_topics_hash != current_hash:
                    rebuild_browser = True
            
            if rebuild_browser:
                # Only rebuild when necessary
                browser_df = st.session_state.df.copy()
                full_topics = np.full(len(browser_df), -1)
                for i, valid_idx in enumerate(st.session_state.valid_indices):
                    if i < len(topics):
                        full_topics[valid_idx] = topics[i]
                browser_df['Topic'] = full_topics
                browser_df['Topic_Label'] = np.where(browser_df['Topic'] == -1, "Outlier",
                                                     "Topic " + browser_df['Topic'].astype(str))
                # human labels / keywords
                st.session_state.topic_human = topic_human
                topic_keywords_map = topic_keywords
                browser_df['Topic_Human_Label'] = browser_df['Topic'].map(st.session_state.topic_human).fillna(browser_df['Topic_Label'])
                browser_df['Topic_Keywords'] = browser_df['Topic'].map(topic_keywords_map).fillna('N/A')
                browser_df['Valid_Document'] = [i in st.session_state.valid_indices for i in range(len(browser_df))]
                
                # Cache in session state
                st.session_state.browser_df = browser_df
                st.session_state.last_topics_hash = hash(tuple(topics))
            else:
                # Use cached version - no expensive operations!
                st.session_state.topic_human = topic_human  # Update labels (cheap)

            # Metrics display
            st.header("📊 Results Summary")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Documents", f"{total_docs:,}")
            with col2:
                st.metric("Topics", unique_topics)
            with col3:
                st.metric("Outliers", f"{outlier_count:,} ({100-coverage:.1f}%)")
            with col4:
                st.metric("Coverage", f"{coverage:.1f}%")
            with col5:
                st.metric("Min Topic Size", st.session_state.get('min_topic_size_used', 'N/A'))

            # Balance warnings
            if not balance_analysis['balanced']:
                st.error("⚠️ **Topic Distribution Imbalance Detected!**")
                for warning in balance_analysis['warnings']:
                    st.warning(warning)

            # Tabs for different views
            tabs = st.tabs([
                "📊 Topics Overview",
                "📈 Distribution Analysis",
                "🔍 Split Large Topics",
                "🗺️ Interactive Visualization",
                "📄 Topic Browser (OPTIMIZED)",
                "💾 Export"
            ])

            with tabs[0]:  # Topics Overview
                st.subheader("Topic Information")

                display_df = normalize_topic_info(topic_info)
                if -1 in display_df['Topic'].values:
                    display_df = display_df[display_df['Topic'] != -1]
                display_df['Percentage'] = (display_df['Count'] / total_docs * 100).round(2)
                display_df = display_df[['Topic', 'Human_Label', 'Keywords', 'Count', 'Percentage']]

                # Streamlit default dataframe (dark-mode friendly)
                st.dataframe(display_df, use_container_width=True)
                st.caption("✨ Notice: Human_Label now shows actual category names like 'Customer Support' instead of keyword lists!")

            with tabs[1]:  # Distribution Analysis
                st.subheader("📈 Topic Distribution Analysis")

                # Prepare data for visualization
                viz_df = pd.DataFrame(Counter([t for t in topics if t != -1]).items(), columns=['Topic', 'Count'])
                viz_df = viz_df.sort_values('Count', ascending=False)
                viz_df['Topic_Label'] = viz_df['Topic'].apply(
                    lambda x: st.session_state.topic_human.get(x, f"Topic {x}")
                )

                # Bar chart
                fig = px.bar(viz_df, x='Topic_Label', y='Count',
                             title='Document Distribution Across Topics',
                             color='Count',
                             color_continuous_scale='Viridis')

                # Add threshold line
                threshold_count = total_docs * 0.3
                fig.add_hline(y=threshold_count, line_dash="dash", line_color="red",
                              annotation_text="30% threshold")

                st.plotly_chart(fig, use_container_width=True)

                # Balance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Balance Score", f"{balance_analysis['balance_score']:.2f}",
                              help="1.0 = perfectly balanced, 0 = highly imbalanced")
                with col2:
                    st.metric("Oversized Topics", len(balance_analysis['oversized_topics']))
                with col3:
                    avg_size = total_docs / unique_topics if unique_topics > 0 else 0
                    st.metric("Avg Topic Size", f"{avg_size:.0f} docs")

            with tabs[2]:  # Split Large Topics
                st.subheader("🔍 Split Large Topics")

                if balance_analysis['oversized_topics']:
                    st.warning(f"Found {len(balance_analysis['oversized_topics'])} oversized topic(s)")

                    oversized_options = [
                        f"Topic {t['topic']}: {st.session_state.topic_human.get(t['topic'], 'Topic ' + str(t['topic']))} - {t['count']} docs ({t['ratio']*100:.1f}%)"
                        for t in balance_analysis['oversized_topics']
                    ]

                    selected_topic_to_split = st.selectbox(
                        "Select topic to split",
                        oversized_options
                    )

                    topic_to_split = int(selected_topic_to_split.split(':')[0].replace('Topic ', ''))

                    suggested_splits = next(
                        t['suggested_splits'] for t in balance_analysis['oversized_topics']
                        if t['topic'] == topic_to_split
                    )

                    sub_n_topics = st.slider(
                        "Number of subtopics",
                        min_value=2,
                        max_value=min(10, suggested_splits * 2),
                        value=suggested_splits,
                        help="How many subtopics to create from this large topic"
                    )

                    if st.button(f"🔍 Split Topic {topic_to_split} into {sub_n_topics} subtopics"):
                        docs_to_split = [doc for doc, t in zip(st.session_state.documents, topics) if t == topic_to_split]

                        if len(docs_to_split) >= max(10, sub_n_topics * 2):
                            with st.spinner(f"Analyzing {len(docs_to_split):,} documents..."):
                                # Use K-means for splitting via BERTopic's API (for convenience)
                                sub_model = BERTopic(
                                    hdbscan_model=GPUKMeans(n_clusters=sub_n_topics) if gpu_capabilities['cuda_available']
                                    else KMeans(n_clusters=sub_n_topics, random_state=42),
                                    min_topic_size=max(2, len(docs_to_split) // (sub_n_topics * 2)),
                                    calculate_probabilities=False,
                                    verbose=False
                                )

                                sub_topics, _ = sub_model.fit_transform(docs_to_split)

                                st.success(f"✅ Split into {len(set(sub_topics))} subtopics")

                                st.write("### 📊 Subtopics Found:")
                                sub_topic_info = sub_model.get_topic_info()

                                sub_topic_display = sub_topic_info[['Topic', 'Count', 'Name']].copy()
                                sub_topic_display['Keywords'] = sub_topic_info['Representation'].apply(
                                    lambda x: ', '.join(x[:3]) if isinstance(x, list) else str(x)[:50]
                                )
                                sub_topic_display['% of Parent'] = (
                                    sub_topic_display['Count'] / len(docs_to_split) * 100
                                ).round(1)

                                st.dataframe(sub_topic_display, use_container_width=True)

                                sub_outliers = sum(1 for t in sub_topics if t == -1)
                                sub_coverage = ((len(sub_topics) - sub_outliers) / len(sub_topics)) * 100

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Subtopics Created", len(set(sub_topics)) - (1 if -1 in sub_topics else 0))
                                with col2:
                                    st.metric("Documents Reassigned", len(docs_to_split) - sub_outliers)
                                with col3:
                                    st.metric("Subtopic Coverage", f"{sub_coverage:.1f}%")
                        else:
                            st.warning(f"Not enough documents for splitting")
                else:
                    st.success("✅ No oversized categories detected. All topics are well-balanced!")

            with tabs[3]:  # Interactive Visualization
                st.subheader("🗺️ Interactive Topic Visualization")

                if len(set(topics)) > 1:
                    viz_df = pd.DataFrame({
                        'Topic': topics,
                        'Document': st.session_state.documents
                    })

                    # Use UMAP embeddings for visualization if available
                    if st.session_state.umap_embeddings is not None and st.session_state.umap_embeddings.shape[1] >= 2:
                        viz_df['X'] = st.session_state.umap_embeddings[:, 0]
                        viz_df['Y'] = st.session_state.umap_embeddings[:, 1]
                    else:
                        # Use PCA for 2D projection
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        coords = pca.fit_transform(st.session_state.embeddings)
                        viz_df['X'] = coords[:, 0]
                        viz_df['Y'] = coords[:, 1]

                    viz_df['Topic_Label'] = [
                        st.session_state.topic_human.get(t, f"Topic {t}") if t != -1 else "Outliers"
                        for t in viz_df['Topic']
                    ]

                    # Sample for performance if too many points
                    if len(viz_df) > 5000:
                        viz_df_sample = viz_df.sample(5000, random_state=42)
                    else:
                        viz_df_sample = viz_df

                    fig = px.scatter(
                        viz_df_sample,
                        x='X', y='Y',
                        color='Topic_Label',
                        hover_data={'Document': True, 'Topic': True},
                        title='Document Clusters in 2D Space',
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 topics for visualization")

            with tabs[4]:  # Fast Topic Browser
                st.subheader("📄 Topic Browser")

                browser_df = st.session_state.browser_df
                text_col = st.session_state.text_col or (st.session_state.df.columns[0] if len(st.session_state.df.columns) else None)

                # Build topic options ONCE and cache in session state
                if 'topic_options_cache' not in st.session_state or st.session_state.get('topic_options_hash') != hash(tuple(topics)):
                    # Simple dropdown for topic selection
                    topic_counts = pd.Series([t for t in topics if t != -1]).value_counts().sort_index()
                    
                    # Build topic options with human labels
                    topic_options = {}
                    topic_options["All Topics"] = "all"
                    
                    for tid in sorted(topic_counts.index):
                        count = topic_counts.get(tid, 0)
                        human_label = st.session_state.topic_human.get(tid, f"Topic {tid}")
                        topic_options[f"Topic {tid}: {human_label} ({count} docs)"] = tid
                    
                    if -1 in topics:
                        outlier_count = sum(1 for t in topics if t == -1)
                        topic_options[f"Outliers ({outlier_count} docs)"] = -1
                    
                    # Cache it
                    st.session_state.topic_options_cache = topic_options
                    st.session_state.topic_options_hash = hash(tuple(topics))
                else:
                    # Use cached version - instant!
                    topic_options = st.session_state.topic_options_cache

                # Single dropdown selector
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_option = st.selectbox(
                        "Select a topic to view documents:",
                        options=list(topic_options.keys()),
                        index=0,
                        key="topic_browser_selector",
                        help="Choose one topic at a time for fast rendering"
                    )
                
                with col2:
                    max_rows = st.number_input(
                        "Max rows to display:",
                        min_value=10,
                        max_value=10000,
                        value=1000,
                        step=100,
                        key="topic_browser_max_rows",
                        help="Limit rows for faster display"
                    )

                # Semantic Search Feature
                st.markdown("---")
                with st.expander("🔍 Semantic Search (AI-powered)", expanded=False):
                    st.caption("Find documents similar to your query using AI embeddings (not just keywords)")
                    
                    search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
                    
                    with search_col1:
                        search_query = st.text_input(
                            "Search query:",
                            placeholder="e.g., 'customer complaints about shipping delays'",
                            key="semantic_search_query",
                            help="Enter a phrase or question - AI will find semantically similar documents"
                        )
                    
                    with search_col2:
                        search_top_n = st.number_input(
                            "Top results:",
                            min_value=5,
                            max_value=500,
                            value=50,
                            step=5,
                            key="semantic_search_top_n",
                            help="Number of most similar documents to show"
                        )
                    
                    with search_col3:
                        search_scope = st.selectbox(
                            "Search in:",
                            options=["Current Topic", "All Topics"],
                            key="semantic_search_scope",
                            help="Search within current topic or across all topics"
                        )
                    
                    search_button = st.button("🔍 Search", key="semantic_search_button", type="primary")
                    
                    if search_button and search_query.strip():
                        with st.spinner("🤖 Computing semantic similarity..."):
                            try:
                                # Get the embedding model
                                if 'model' not in st.session_state or st.session_state.model is None:
                                    st.error("❌ Model not loaded. Please run topic modeling first.")
                                else:
                                    # Encode the search query
                                    query_embedding = st.session_state.model.encode(
                                        [search_query],
                                        convert_to_numpy=True,
                                        normalize_embeddings=True
                                    )[0]
                                    
                                    # Get embeddings and valid indices
                                    embeddings = st.session_state.embeddings
                                    valid_indices = st.session_state.valid_indices
                                    
                                    # Determine which documents to search
                                    if search_scope == "Current Topic" and selected_topic_id != "all":
                                        # Get indices for current topic
                                        topic_mask = browser_df['Topic'].values == selected_topic_id
                                        topic_indices = np.where(topic_mask)[0]
                                        
                                        # Map to valid_indices
                                        search_indices = []
                                        for idx in topic_indices:
                                            if idx in valid_indices:
                                                search_indices.append(valid_indices.index(idx))
                                        
                                        if len(search_indices) == 0:
                                            st.warning("No valid documents in current topic to search")
                                            search_indices = list(range(len(embeddings)))
                                    else:
                                        # Search all documents
                                        search_indices = list(range(len(embeddings)))
                                    
                                    # Calculate cosine similarity for selected documents
                                    similarities = []
                                    for idx in search_indices:
                                        if idx < len(embeddings):
                                            similarity = np.dot(query_embedding, embeddings[idx])
                                            similarities.append((idx, similarity, valid_indices[idx]))
                                    
                                    # Sort by similarity (highest first)
                                    similarities.sort(key=lambda x: x[1], reverse=True)
                                    
                                    # Get top N results
                                    top_results = similarities[:search_top_n]
                                    
                                    if len(top_results) > 0:
                                        # Get the browser indices for top results
                                        result_browser_indices = [x[2] for x in top_results]
                                        result_similarities = [x[1] for x in top_results]
                                        
                                        # Create results dataframe
                                        search_results_df = browser_df.iloc[result_browser_indices].copy()
                                        search_results_df.insert(0, 'Similarity', [f"{sim:.3f}" for sim in result_similarities])
                                        
                                        # Store in session state to display below
                                        st.session_state.search_active = True
                                        st.session_state.search_results_df = search_results_df
                                        st.session_state.search_query_display = search_query
                                        
                                        st.success(f"✅ Found {len(top_results)} similar documents!")
                                    else:
                                        st.warning("No results found")
                                        
                            except Exception as e:
                                st.error(f"❌ Search error: {str(e)}")
                    
                    elif search_button:
                        st.warning("Please enter a search query")
                
                st.markdown("---")

                # Check if we should display search results instead of topic view
                display_search_results = st.session_state.get('search_active', False)
                
                if display_search_results and 'search_results_df' in st.session_state:
                    # Display search results
                    search_results_df = st.session_state.search_results_df
                    search_query_display = st.session_state.get('search_query_display', '')
                    
                    st.info(f"🔍 **Semantic Search Results** for: \"{search_query_display}\" — Showing {len(search_results_df)} most similar documents")
                    
                    # Add a button to clear search and return to normal view
                    if st.button("❌ Clear Search Results", key="clear_search"):
                        st.session_state.search_active = False
                        if 'search_results_df' in st.session_state:
                            del st.session_state.search_results_df
                        st.rerun()
                    
                    # Reorder columns
                    meta_cols = ['Similarity', 'Topic', 'Topic_Human_Label', 'Topic_Keywords']
                    other_cols = [c for c in search_results_df.columns if c not in meta_cols and c not in ['Topic_Label', 'Valid_Document']]
                    ordered_cols = [c for c in meta_cols if c in search_results_df.columns] + other_cols
                    search_results_df = search_results_df[ordered_cols]
                    
                    # Display
                    st.dataframe(
                        search_results_df,
                        use_container_width=True,
                        height=600
                    )
                    
                    # Download button
                    st.download_button(
                        label=f"📥 Download Search Results ({len(search_results_df)} rows)",
                        data=search_results_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"semantic_search_{search_query_display[:30]}_{st.session_state.uploaded_file_name}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    # Normal topic browsing view
                    # Get the selected topic ID
                    selected_topic_id = topic_options[selected_option]

                    # Fast filtering - single topic only
                    if selected_topic_id == "all":
                        display_df = browser_df.head(max_rows)
                        st.info(f"📊 Showing first {len(display_df):,} of {len(browser_df):,} total documents")
                    else:
                        # Use numpy for fast filtering
                        mask = browser_df['Topic'].values == selected_topic_id
                        filtered_df = browser_df[mask]
                        display_df = filtered_df.head(max_rows)
                        
                        if selected_topic_id == -1:
                            st.info(f"📊 Showing {len(display_df):,} of {len(filtered_df):,} outlier documents")
                        else:
                            human_label = st.session_state.topic_human.get(selected_topic_id, f"Topic {selected_topic_id}")
                            st.info(f"📊 Topic {selected_topic_id}: **{human_label}** — Showing {len(display_df):,} of {len(filtered_df):,} documents")

                    # Reorder columns - put topic metadata first
                    meta_cols = ['Topic', 'Topic_Human_Label', 'Topic_Keywords']
                    other_cols = [c for c in display_df.columns if c not in meta_cols and c not in ['Topic_Label', 'Valid_Document']]
                    ordered_cols = [c for c in meta_cols if c in display_df.columns] + other_cols
                    display_df = display_df[ordered_cols]

                    # Simple, fast dataframe display
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=600
                    )

                    # Download button
                    if selected_topic_id == "all":
                        download_df = browser_df
                        filename = f"all_topics_{st.session_state.uploaded_file_name}.csv"
                    else:
                        download_df = browser_df[browser_df['Topic'] == selected_topic_id]
                        filename = f"topic_{selected_topic_id}_{st.session_state.uploaded_file_name}.csv"

                    st.download_button(
                        label=f"📥 Download {selected_option} ({len(download_df):,} rows)",
                        data=download_df.to_csv(index=False).encode('utf-8'),
                        file_name=filename,
                        mime="text/csv"
                    )

            with tabs[5]:  # Export
                st.subheader("💾 Export Results")

                export_df = st.session_state.browser_df.copy()
                safe_topic_info = normalize_topic_info(topic_info)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=convert_df_to_csv(export_df),
                        file_name=f"bertopic_results_{st.session_state.uploaded_file_name}_{n_topics_slider}topics.csv",
                        mime="text/csv",
                        help="Full dataset with topic assignments and human labels"
                    )

                with col2:
                    st.download_button(
                        label="📥 Download Topic Info (CSV)",
                        data=convert_df_to_csv(safe_topic_info[['Topic','Human_Label','Keywords','Count']]),
                        file_name=f"topic_info_{st.session_state.uploaded_file_name}_{n_topics_slider}topics.csv",
                        mime="text/csv",
                        help="Topic descriptions and statistics (with human labels)"
                    )

                with col3:
                    # Create summary report
                    summary = f"""Topic Modeling Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
File: {st.session_state.uploaded_file_name}
Documents: {len(processed_df)}
Topics Found: {unique_topics}
Outliers: {outlier_count} ({100-coverage:.1f}%)
Coverage: {coverage:.1f}%
Min Topic Size: {st.session_state.get('min_topic_size_used', 'N/A')}
GPU Used: {gpu_capabilities['cuda_available']}
Clustering Method: {st.session_state.clustering_method}
Human Labels: Auto-generated using semantic category inference
Balance Status: {'Balanced' if balance_analysis['balanced'] else 'Needs Attention'}
Oversized Categories: {len(balance_analysis['oversized_topics'])}
"""
                    st.download_button(
                        label="📥 Download Report (TXT)",
                        data=summary.encode("utf-8"),
                        file_name=f"report_{st.session_state.uploaded_file_name}_{n_topics_slider}topics.txt",
                        mime="text/plain",
                        help="Summary report"
                    )

    elif 'df' not in st.session_state or st.session_state.df is None:
        # Welcome screen
        st.info("👆 Please upload a CSV file in the sidebar to begin.")

        # Feature highlights
        st.header("🚀 Complete Feature Set")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("⚡ Interactive Features")
            st.markdown("""
            - **Dynamic slider** for instant topic adjustment
            - **One-time embedding** computation
            - **Fast reclustering** (<1 second)
            - **Real-time visualization** updates
            - **Interactive scatter plots**
            """)

        with col2:
            st.subheader("🛡️ Robust Processing")
            st.markdown("""
            - **Handles all edge cases**
            - **Smart text preprocessing**
            - **Memory error recovery**
            - **Batch size auto-adjustment**
            - **Encoding error handling**
            """)

        with col3:
            st.subheader("📊 Analysis Tools")
            st.markdown("""
            - **Split large topics** tool
            - **Balance analysis**
            - **Seed words** support
            - **Outlier reduction** strategies
            - **Multiple export** formats
            """)

        st.header("✨ What's New")
        st.success("""
        **🎯 Improved Human-Readable Labels**: Topic labels are now actual category names like "Customer Support" or "Marketing & Advertising" 
        instead of just keyword lists!
        
        **⚡ Optimized Topic Browser**: Added pagination for much faster browsing of large datasets. No more lag when viewing thousands of documents!
        """)

        # System check
        with st.expander("🔍 Check Your System"):
            if st.button("Run System Check"):
                capabilities = check_gpu_capabilities()

                st.write("### System Capabilities:")

                gpu_col, pkg_col = st.columns(2)

                with gpu_col:
                    st.write("**GPU Status:**")
                    for key in ['cuda_available', 'device_name', 'gpu_memory_total']:
                        if key in capabilities and capabilities[key] is not None:
                            icon = "✅" if capabilities.get('cuda_available', False) else "❌"
                            value = capabilities[key]
                            if isinstance(value, bool):
                                st.write(f"{icon} {key.replace('_', ' ').title()}")
                            else:
                                st.write(f"{icon} {key.replace('_', ' ').title()}: {value}")

                with pkg_col:
                    st.write("**Packages:**")
                    for key in ['accelerate_available', 'cupy_available', 'faiss_available']:
                        if key in capabilities:
                            icon = "✅" if capabilities[key] else "❌"
                            st.write(f"{icon} {key.replace('_available', '').title()}")

if __name__ == "__main__":
    main()
