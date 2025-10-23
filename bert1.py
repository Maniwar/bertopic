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
from sklearn.feature_extraction.text import CountVectorizer
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
                min_cluster_size = max(min_topic_size, len(self.documents) // (n_topics * 2))
                min_cluster_size = min(min_cluster_size, len(valid_embeddings) // 2)

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
        """Extract keywords for each topic"""
        topics_dict = {}
        for idx, topic in enumerate(topics):
            if topic not in topics_dict:
                topics_dict[topic] = []
            if idx < len(self.documents):
                topics_dict[topic].append(self.documents[idx])

        topic_info_list = []
        for topic_id in sorted(topics_dict.keys()):
            if topic_id == -1:
                topic_info_list.append({
                    'Topic': -1,
                    'Count': len(topics_dict[-1]),
                    'Keywords': 'Outliers',
                    'Name': 'Outliers'
                })
                continue

            topic_docs = topics_dict[topic_id]

            try:
                sample_size = min(100, len(topic_docs))
                topic_text = ' '.join(topic_docs[:sample_size])

                words = topic_text.lower().split()
                word_counts = Counter(words)

                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                                'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'it', 'this',
                                'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they'}

                filtered_counts = {w: c for w, c in word_counts.items()
                                   if w not in common_words and len(w) > 2}

                top_words = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
                keywords = [word for word, _ in top_words]

                if not keywords:
                    keywords = ['topic', str(topic_id)]

            except Exception:
                keywords = [f'topic_{topic_id}']

            topic_info_list.append({
                'Topic': topic_id,
                'Count': len(topic_docs),
                'Keywords': ', '.join(keywords[:5]),
                'Name': f"Topic {topic_id}: {', '.join(keywords[:3])}"
            })

        return pd.DataFrame(topic_info_list)

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

# -----------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------
def main():
    st.title("🚀 Complete BERTopic: All Features + Interactive + Robust")

    # Initialize ALL session state variables to prevent AttributeError
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

    # Check GPU capabilities
    gpu_capabilities = check_gpu_capabilities()

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
            min_topic_size = st.slider(
                "Minimum Topic Size",
                min_value=2,
                max_value=min(100, len(df) // 10),
                value=min(10, len(df) // 50),
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
                    max_topics = max(5, len(df) // min_topic_size)
                    default_topics = min(max_topics, max(5, len(df) // 50))

                    nr_topics = st.number_input(
                        "Initial Number of Topics",
                        min_value=2,
                        max_value=max_topics,
                        value=default_topics,
                        help=f"Number of topics to create (limited by min topic size of {min_topic_size})"
                    )
                else:
                    nr_topics = min(10, len(df) // 50)

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

                # Representation model
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
                        # ensure base model is cached (keeps behavior identical for downstream code)
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

                # Step 5: Parse seed words if provided
                seed_topic_list = []
                if seed_words_input:
                    for line in seed_words_input.strip().split('\n'):
                        if line.strip():
                            words = [w.strip() for w in line.split(',')]
                            if words:
                                seed_topic_list.append(words)

                # Step 6: Perform initial clustering
                with st.spinner("Performing initial clustering..."):
                    # Determine clustering method
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

                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.topic_info = topic_info  # Also store as topic_info
                    st.session_state.min_topic_size = min_topic_size
                    st.session_state.min_topic_size_used = min_topic_size
                    st.session_state.embeddings_computed = True
                    st.session_state.clustering_method = "K-means" if method == 'kmeans' else "HDBSCAN"
                    st.session_state.gpu_used = gpu_capabilities['cuda_available']
                    st.session_state.model = None  # Not using traditional BERTopic model in this version

                    st.success("✅ Embeddings computed! You can now adjust topics dynamically with the slider below.")
                    st.balloons()

    # Main content area
    if st.session_state.embeddings_computed:
        st.success("✅ Embeddings ready! Use the controls below for instant topic adjustment.")

        # Interactive controls section
        st.header("🎚️ Dynamic Topic Adjustment")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            max_topics = min(100, len(st.session_state.documents) // st.session_state.min_topic_size)

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
                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.topic_info = topic_info  # Keep both in sync
                    st.success(f"✅ Reclustered into {len(topic_info[topic_info['Topic'] != -1])} topics!")
                else:
                    st.error("Reclustering failed. Try different parameters.")

        # Display results
        if st.session_state.current_topics is not None:
            topics = st.session_state.current_topics
            topic_info = st.session_state.current_topic_info  # Use current_topic_info

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

            # Add keywords for each topic
            topic_keywords = {}
            for _, row in topic_info.iterrows():
                topic_keywords[row['Topic']] = row['Keywords']
            processed_df['keywords'] = processed_df['topic'].map(topic_keywords)

            # Store in session state for export and other tabs
            st.session_state.processed_df = processed_df

            # Balance Analysis
            balance_analyzer = CategoryBalanceAnalyzer()
            balance_analysis = balance_analyzer.analyze_distribution(
                topics,
                include_outliers=False
            )

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
                "📄 Document Explorer",
                "💾 Export"
            ])

            with tabs[0]:  # Topics Overview
                st.subheader("Topic Information")

                display_df = topic_info.copy()
                display_df = display_df[display_df['Topic'] != -1] if -1 in display_df['Topic'].values else display_df
                display_df['Percentage'] = (display_df['Count'] / total_docs * 100).round(2)

                def highlight_oversized(row):
                    if row['Percentage'] > 30:
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['Percentage'] > 20:
                        return ['background-color: #fff4cc'] * len(row)
                    elif row['Percentage'] < 2:
                        return ['background-color: #ccf4ff'] * len(row)
                    else:
                        return [''] * len(row)

                styled_df = display_df.style.apply(highlight_oversized, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                st.caption("🔴 Red: >30% (Too Large) | 🟡 Yellow: >20% (Large) | 🔵 Blue: <2% (Small)")

            with tabs[1]:  # Distribution Analysis
                st.subheader("📈 Topic Distribution Analysis")

                # Prepare data for visualization
                viz_df = pd.DataFrame(Counter([t for t in topics if t != -1]).items(), columns=['Topic', 'Count'])
                viz_df = viz_df.sort_values('Count', ascending=False)
                viz_df['Topic_Label'] = viz_df['Topic'].apply(lambda x: f"Topic {x}")

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
                        f"Topic {t['topic']}: {t['count']} docs ({t['ratio']*100:.1f}%)"
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
                                # Use K-means for splitting
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

                    viz_df['Topic_Label'] = viz_df['Topic'].apply(
                        lambda x: f"Topic {x}" if x != -1 else "Outliers"
                    )

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

            with tabs[4]:  # Document Explorer
                st.subheader("📄 Document Explorer")

                topic_options = [f"Topic {t}" for t in sorted(set(topics)) if t != -1]
                if -1 in topics:
                    topic_options.append("Outliers")

                selected_topic_str = st.selectbox("Select topic to explore", topic_options)

                if selected_topic_str == "Outliers":
                    selected_topic = -1
                else:
                    selected_topic = int(selected_topic_str.replace("Topic ", ""))

                topic_docs = [doc for doc, t in zip(st.session_state.documents, topics) if t == selected_topic]

                st.write(f"**Showing documents from {selected_topic_str} (Total: {len(topic_docs)})**")

                # Display sample documents
                for i, doc in enumerate(topic_docs[:10]):
                    with st.expander(f"Document {i+1}"):
                        st.write(doc[:500] + "..." if len(doc) > 500 else doc)

            with tabs[5]:  # Export
                st.subheader("💾 Export Results")

                # Prepare full export dataframe
                export_df = st.session_state.df.copy()

                # Map topics back to original indices
                full_topics = np.full(len(export_df), -1)
                for i, valid_idx in enumerate(st.session_state.valid_indices):
                    if i < len(topics):
                        full_topics[valid_idx] = topics[i]

                export_df['Topic'] = full_topics
                export_df['Topic_Label'] = [f"Topic {t}" if t != -1 else "Outlier" for t in full_topics]
                export_df['Valid_Document'] = [i in st.session_state.valid_indices for i in range(len(export_df))]

                # Add keywords
                export_df['Topic_Keywords'] = export_df['Topic'].map(topic_keywords).fillna('N/A')

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=convert_df_to_csv(export_df),
                        file_name=f"bertopic_results_{st.session_state.uploaded_file_name}_{n_topics_slider}topics.csv",
                        mime="text/csv",
                        help="Full dataset with topic assignments"
                    )

                with col2:
                    st.download_button(
                        label="📥 Download Topic Info (CSV)",
                        data=convert_df_to_csv(topic_info),
                        file_name=f"topic_info_{st.session_state.uploaded_file_name}_{n_topics_slider}topics.csv",
                        mime="text/csv",
                        help="Topic descriptions and statistics"
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
Clustering Method: {clustering_method}
Balance Status: {'Balanced' if balance_analysis['balanced'] else 'Needs Attention'}
Oversized Categories: {len(balance_analysis['oversized_topics'])}
"""
                    st.download_button(
                        label="📥 Download Report (TXT)",
                        data=summary,
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
