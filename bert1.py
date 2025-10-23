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

# -----------------------------------------------------
# PAGE / GLOBAL CONFIG
# -----------------------------------------------------
st.set_page_config(layout="wide", page_title="Complete BERTopic with All Features", page_icon="🚀")

# Optional libs (accelerate)
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# GPU-accelerated libs (optional)
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

# FAISS (optional)
try:
    import faiss
    faiss_available = True
    faiss_gpu_available = torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
except ImportError:
    faiss_available = False
    faiss_gpu_available = False

# Force CUDA init if available (best effort)
if torch.cuda.is_available():
    try:
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.8)
    except Exception as e:
        st.warning(f"CUDA initialization warning: {e}")

# -----------------------------------------------------
# UTILITIES
# -----------------------------------------------------
def title_case_label(words):
    """Make a nice title-cased label from a list of words."""
    words = [w.strip() for w in words if w and isinstance(w, str)]
    words = [re.sub(r'[^a-zA-Z0-9\- ]+', '', w) for w in words]
    words = [w for w in words if len(w) >= 2]
    if not words:
        return None
    label = " • ".join([w.title() for w in words[:3]])
    return label if label.strip() else None

def generate_human_label(topic_id, keywords_str, fallback_name=None):
    """Derive a readable label from Keywords/Name with safe fallbacks."""
    # Prefer provided Name if sensible
    if fallback_name and isinstance(fallback_name, str) and fallback_name.strip() and "Topic" not in fallback_name:
        return fallback_name.strip()

    # Use keywords (CSV string)
    kw_list = []
    if isinstance(keywords_str, str) and keywords_str.strip():
        kw_list = [k.strip() for k in keywords_str.split(",") if k.strip()]

    # Further refine by removing common stopwords & deduping
    stop = set("""
        the a an and or but in on at to for of with by from as is was are were be have has had do does did
        will would could should may might must can it this that these those i you he she we they your our their
        rt via https http www com amp
    """.split())
    kw_list = [w for w in kw_list if w.lower() not in stop]
    # make sure unique preserving order
    seen = set()
    kw_list = [w for w in kw_list if not (w.lower() in seen or seen.add(w.lower()))]

    lbl = title_case_label(kw_list)
    if lbl:
        return lbl

    # Last resorts
    if isinstance(keywords_str, str) and keywords_str.strip():
        return f"Topic {int(topic_id)}: {keywords_str.strip()}"
    return f"Topic {int(topic_id)}"

def ensure_human_labels(topic_info: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a Human_Label column exists on topic_info."""
    if topic_info is None or len(topic_info) == 0:
        return topic_info
    df = topic_info.copy()
    # Ensure required cols exist
    if 'Topic' not in df.columns:
        df['Topic'] = np.arange(len(df))
    if 'Keywords' not in df.columns:
        df['Keywords'] = ""

    # Try to use 'Name' if present
    if 'Name' not in df.columns:
        df['Name'] = df.apply(lambda r: f"Topic {int(r['Topic'])}", axis=1)

    # Build Human_Label
    df['Human_Label'] = df.apply(
        lambda r: generate_human_label(r['Topic'], r.get('Keywords', ''), r.get('Name', None)),
        axis=1
    )
    # Make Outliers explicit
    df.loc[df['Topic'] == -1, 'Human_Label'] = 'Outliers'
    return df

# -----------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------
class RobustTextPreprocessor:
    """Handles all text preprocessing edge cases to prevent tokenization errors"""
    @staticmethod
    def clean_text(text, max_length=5000, min_length=10):
        if pd.isna(text) or text is None:
            return None
        # If list/tuple -> join
        if isinstance(text, (list, tuple)):
            try:
                text = ' '.join(map(str, text))
            except Exception:
                return None
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return None
        text = ' '.join(text.split())
        if len(text.strip()) < min_length:
            return None
        if text.startswith('[') or text.startswith('{'):
            text = re.sub(r'[\[\]{}"\']', ' ', text)
            text = ' '.join(text.split())
        text = text.replace('\x00', '').replace('\r', ' ').replace('\t', ' ')
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception:
            pass
        if len(text) > max_length:
            truncated = text[:max_length]
            last_period = truncated.rfind('.')
            text = truncated[:last_period + 1] if last_period > max_length * 0.8 else truncated + '...'
        text = text.strip()
        if len(text) < min_length:
            return None
        return text

    @staticmethod
    def preprocess_documents(documents, max_length=5000, min_length=10, show_warnings=True):
        cleaned_docs, valid_indices = [], []
        stats = {'total': len(documents), 'valid': 0, 'empty_removed': 0, 'too_short': 0,
                 'too_long_truncated': 0, 'type_converted': 0}
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
# TOPIC MERGE (min size)
# -----------------------------------------------------
class TopicMerger:
    @staticmethod
    def merge_small_topics(topics, embeddings, min_size=10):
        topics = np.array(topics)
        topic_counts = Counter(topics)
        small_topics = [t for t, c in topic_counts.items() if t != -1 and c < min_size]
        if not small_topics:
            return topics
        # centroids
        topic_centroids = {}
        for t in set(topics):
            if t != -1:
                mask = topics == t
                if np.any(mask):
                    te = embeddings[mask]
                    if len(te) > 0 and not np.all(te == 0):
                        topic_centroids[t] = np.mean(te, axis=0)
        merged = topics.copy()
        for stp in small_topics:
            if stp not in topic_centroids:
                continue
            sc = topic_centroids[stp]
            min_dist, best = float('inf'), None
            for t, c in topic_centroids.items():
                if t != stp and topic_counts[t] >= min_size:
                    try:
                        d = cosine(sc, c)
                        if d < min_dist:
                            min_dist, best = d, t
                    except Exception:
                        continue
            if best is not None:
                merged[merged == stp] = best
                topic_counts[best] += topic_counts[stp]
                del topic_counts[stp]
        # renumber
        uniq = sorted(set(merged[merged != -1]))
        mapping = {old: new for new, old in enumerate(uniq)}
        mapping[-1] = -1
        return np.array([mapping.get(t, t) for t in merged])

# -----------------------------------------------------
# BALANCE ANALYZER
# -----------------------------------------------------
class CategoryBalanceAnalyzer:
    def __init__(self, min_topic_ratio=0.30, ideal_max_ratio=0.20):
        self.min_topic_ratio = min_topic_ratio
        self.ideal_max_ratio = ideal_max_ratio

    def analyze_distribution(self, labels, include_outliers=False):
        total_docs = len(labels)
        labels_filtered = labels if include_outliers else [l for l in labels if l != -1]
        total_docs = len(labels_filtered)
        if total_docs == 0:
            return {'balanced': False, 'oversized_topics': [], 'distribution': {}, 'warnings': ['No documents to analyze']}
        uniq = sorted(set(labels_filtered))
        topic_counts = {}
        for t in uniq:
            c = labels_filtered.count(t)
            topic_counts[t] = {'count': c, 'ratio': c/total_docs}
        oversized, warns = [], []
        for t, s in topic_counts.items():
            if s['ratio'] > self.min_topic_ratio:
                splits = max(2, int(s['ratio'] / self.ideal_max_ratio))
                oversized.append({'topic': t, 'count': s['count'], 'ratio': s['ratio'], 'suggested_splits': splits})
                warns.append(f"Topic {t}: {s['count']} docs ({s['ratio']*100:.1f}%) - Consider splitting into {splits} subtopics")
        ratios = [s['ratio'] for s in topic_counts.values()]
        balance_score = 1 - (np.std(ratios) if len(ratios) > 1 else 0)
        return {'balanced': len(oversized) == 0, 'balance_score': balance_score,
                'oversized_topics': oversized, 'distribution': topic_counts,
                'warnings': warns, 'total_topics': len(uniq),
                'outlier_ratio': 0.0}

# -----------------------------------------------------
# OUTLIER STRATEGY (kept for completeness)
# -----------------------------------------------------
class OutlierReducer:
    @staticmethod
    def suggest_parameters(num_docs, current_outlier_ratio, min_topic_size=10):
        suggestions = {'method': '', 'parameters': {}, 'explanation': ''}
        if current_outlier_ratio > 0.3:
            suggestions['method'] = 'kmeans'
            max_topics = max(5, num_docs // min_topic_size)
            suggested = min(max_topics, max(5, num_docs // 50))
            suggestions['parameters'] = {'use_kmeans': True, 'use_gpu_kmeans': True,
                                         'nr_topics': suggested, 'min_topic_size': min_topic_size,
                                         'merge_small_topics': True}
            suggestions['explanation'] = f"High outliers. K-means assigns all docs; merge topics < {min_topic_size}."
        elif current_outlier_ratio > 0.15:
            suggestions['method'] = 'hdbscan'
            suggestions['parameters'] = {'use_kmeans': False, 'min_cluster_size': max(3, min_topic_size // 3),
                                         'min_samples': max(1, min_topic_size // 5),
                                         'prediction_data': True, 'min_topic_size': min_topic_size}
            suggestions['explanation'] = "Moderate outliers. Lenient HDBSCAN."
        else:
            suggestions['method'] = 'hdbscan'
            suggestions['parameters'] = {'use_kmeans': False, 'min_cluster_size': min_topic_size,
                                         'min_samples': max(1, min_topic_size // 2),
                                         'prediction_data': False, 'min_topic_size': min_topic_size}
            suggestions['explanation'] = "Low outliers. Standard HDBSCAN."
        return suggestions

# -----------------------------------------------------
# GPU KMEANS WRAPPER
# -----------------------------------------------------
class GPUKMeans:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        if cuml_available and torch.cuda.is_available():
            try:
                from cuml.cluster import KMeans as cuKMeans
                self.model = cuKMeans(n_clusters=n_clusters, random_state=random_state)
                self.use_gpu = True
            except Exception:
                self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
                self.use_gpu = False
        else:
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
            self.use_gpu = False
    def fit(self, X): return self.model.fit(X)
    def predict(self, X): return self.model.predict(X)
    def fit_predict(self, X): return self.model.fit_predict(X)

# -----------------------------------------------------
# SAFE EMBEDDINGS (fixes padding/truncation errors)
# -----------------------------------------------------
class SafeEmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = min(getattr(self.model, "max_seq_length", 512), 512)

    def encode_safe(self, documents, batch_size=32, show_progress=True):
        if not documents:
            raise ValueError("No documents to encode")
        documents = [str(d) if d is not None else "" for d in documents]
        non_empty = [(i, d) for i, d in enumerate(documents) if d.strip()]
        if not non_empty:
            raise ValueError("All documents are empty after preprocessing")
        idxs, docs = zip(*non_empty)
        try:
            embs = self.model.encode(
                list(docs),
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            full = np.zeros((len(documents), embs.shape[1]))
            for i, idx in enumerate(idxs):
                full[idx] = embs[i]
            return full
        except RuntimeError as e:
            msg = str(e).lower()
            if "too large" in msg or "out of memory" in msg:
                st.warning("Reducing batch size due to memory constraints…")
                return self.encode_safe(documents, batch_size=max(1, batch_size // 2), show_progress=show_progress)
            if "padding" in msg or "truncation" in msg:
                st.warning("Processing documents one-by-one due to length variance…")
                return self._encode_one_by_one(docs, idxs, len(documents))
            raise

    def _encode_one_by_one(self, documents, indices, total_len):
        progress = st.progress(0.0)
        out = []
        for i, d in enumerate(documents):
            try:
                e = self.model.encode([d], convert_to_numpy=True, normalize_embeddings=True)
                out.append(e[0])
            except Exception:
                out.append(np.zeros(self.model.get_sentence_embedding_dimension()))
            progress.progress((i + 1) / len(documents))
        progress.empty()
        full = np.zeros((total_len, len(out[0])))
        for i, idx in enumerate(indices):
            full[idx] = out[i]
        return full

# -----------------------------------------------------
# RECLUSTER ENGINE
# -----------------------------------------------------
class FastReclusterer:
    def __init__(self, documents, embeddings, umap_embeddings=None):
        self.documents = documents
        self.embeddings = embeddings
        self.umap_embeddings = umap_embeddings
        self.use_gpu = torch.cuda.is_available() and cuml_available

    def recluster(self, n_topics, min_topic_size=10, use_reduced=True, method='kmeans'):
        X = self.umap_embeddings if (use_reduced and self.umap_embeddings is not None) else self.embeddings
        valid_mask = np.any(X != 0, axis=1)
        valid_X = X[valid_mask]
        if len(valid_X) == 0:
            st.error("No valid embeddings for clustering!")
            return None, None
        try:
            if method == 'kmeans':
                if self.use_gpu:
                    try:
                        from cuml.cluster import KMeans as cuKMeans
                        clusterer = cuKMeans(n_clusters=min(n_topics, len(valid_X)), random_state=42)
                    except Exception:
                        clusterer = KMeans(n_clusters=min(n_topics, len(valid_X)), random_state=42)
                else:
                    clusterer = KMeans(n_clusters=min(n_topics, len(valid_X)), random_state=42)
                valid_topics = clusterer.fit_predict(valid_X)
            else:
                min_cluster_size = max(min_topic_size, len(self.documents) // max(2, (n_topics * 2)))
                min_cluster_size = min(min_cluster_size, max(2, len(valid_X) // 2))
                if self.use_gpu:
                    clusterer = cumlHDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(1, min_cluster_size // 5))
                else:
                    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(1, min_cluster_size // 5))
                valid_topics = clusterer.fit_predict(valid_X)

            topics = np.full(len(X), -1)
            topics[valid_mask] = valid_topics
        except Exception as e:
            st.error(f"Clustering error: {str(e)}")
            return None, None

        if min_topic_size > 2:
            topics = TopicMerger.merge_small_topics(topics, self.embeddings, min_topic_size)

        topic_info = self._extract_topic_keywords(topics)
        topic_info = ensure_human_labels(topic_info)
        return topics, topic_info

    def _extract_topic_keywords(self, topics, top_n_words=10):
        dmap = {}
        for i, t in enumerate(topics):
            dmap.setdefault(t, []).append(self.documents[i] if i < len(self.documents) else "")
        rows = []
        for t in sorted(dmap.keys()):
            if t == -1:
                rows.append({'Topic': -1, 'Count': len(dmap[-1]), 'Keywords': 'Outliers', 'Name': 'Outliers'})
                continue
            docs = dmap[t]
            sample_size = min(200, len(docs))
            text = ' '.join(docs[:sample_size]).lower()
            words = re.findall(r"[a-z0-9\-]+", text)
            stop = set("""
                the a an and or but in on at to for of with by from as is was are were be have has had do does did will would could should may might must
                can need it this that these those i you he she we they your our their rt via https http www com amp
            """.split())
            counts = Counter([w for w in words if len(w) > 2 and w not in stop])
            top = [w for w, _ in counts.most_common(top_n_words)]
            keywords = ', '.join(top[:10]) if top else f"topic_{t}"
            rows.append({'Topic': t, 'Count': len(docs), 'Keywords': keywords, 'Name': f"Topic {t}: {', '.join(top[:3])}"})
        return pd.DataFrame(rows)

# -----------------------------------------------------
# CACHING / GPU INFO
# -----------------------------------------------------
def check_gpu_capabilities():
    caps = {'cuda_available': torch.cuda.is_available(), 'device_count': 0, 'device_name': None,
            'gpu_memory_total': None, 'gpu_memory_free': None, 'cupy_available': cupy_available,
            'cuml_available': cuml_available, 'faiss_available': faiss_available,
            'faiss_gpu_available': faiss_gpu_available, 'accelerate_available': accelerate_available}
    if torch.cuda.is_available():
        caps['device_count'] = torch.cuda.device_count()
        caps['device_name'] = torch.cuda.get_device_name(0)
        try:
            mem_info = torch.cuda.mem_get_info(0)
            caps['gpu_memory_free'] = f"{mem_info[0] / 1024**3:.1f} GB"
            caps['gpu_memory_total'] = f"{mem_info[1] / 1024**3:.1f} GB"
        except Exception:
            pass
    return caps

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def load_embedding_model(model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 512
    # Optional light compile of inner HF model (best-effort)
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
    valid_mask = np.any(embeddings != 0, axis=1)
    if np.sum(valid_mask) < 10:
        return None
    if cuml_available and torch.cuda.is_available():
        reducer = cumlUMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=42)
    else:
        reducer = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=42)
    umap_embeddings = np.zeros((len(embeddings), n_components))
    umap_embeddings[valid_mask] = reducer.fit_transform(embeddings[valid_mask])
    return umap_embeddings

# -----------------------------------------------------
# MAIN APP
# -----------------------------------------------------
def main():
    st.title("🚀 Complete BERTopic: All Features + Interactive + Robust")

    # Initialize state
    defaults = {
        'embeddings_computed': False, 'embeddings': None, 'umap_embeddings': None,
        'documents': None, 'df': None, 'reclusterer': None, 'current_topics': None,
        'current_topic_info': None, 'valid_indices': None, 'min_topic_size_used': 10,
        'uploaded_file_name': 'data', 'topic_info': None, 'model': None,
        'processed_df': None, 'min_topic_size': 10, 'clustering_method': 'Unknown',
        'gpu_used': False, 'text_col': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    gpu_capabilities = check_gpu_capabilities()

    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Configuration")
        if gpu_capabilities['cuda_available']:
            st.success(f"✅ GPU: {gpu_capabilities['device_name']}")
            if gpu_capabilities['gpu_memory_free']:
                st.info(f"Memory: {gpu_capabilities['gpu_memory_free']} / {gpu_capabilities['gpu_memory_total']}")
        else:
            st.warning("⚠️ No GPU detected. Using CPU (slower)")

        st.subheader("📦 Acceleration Status")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"{'✅' if gpu_capabilities['cuml_available'] else '❌'} cuML")
            st.write(f"{'✅' if gpu_capabilities['cupy_available'] else '❌'} CuPy")
        with c2:
            st.write(f"{'✅' if gpu_capabilities['faiss_gpu_available'] else '❌'} FAISS GPU")
            st.write(f"{'✅' if gpu_capabilities['accelerate_available'] else '❌'} Accelerate")

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

            text_col = st.selectbox("Select text column", df.columns.tolist(), help="Column containing the text to analyze")

            st.header("🎯 Analysis Settings")
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-MiniLM-L12-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
                help="Smaller models are faster but may be less accurate"
            )

            with st.expander("🔧 Preprocessing Settings"):
                max_doc_length = st.slider("Max document length (chars)", 1000, 10000, 5000)
                min_doc_length = st.slider("Min document length (chars)", 1, 100, 10)

            st.subheader("📏 Topic Size Control")
            min_topic_size = st.slider(
                "Minimum Topic Size",
                min_value=2,
                max_value=min(100, len(df) // 10),
                value=min(10, len(df) // 50) if len(df) > 0 else 10,
                help="Topics smaller than this will be merged."
            )

            outlier_strategy = st.selectbox(
                "Outlier Reduction Strategy",
                ["Aggressive (K-means) - 0% outliers",
                 "Moderate (Lenient HDBSCAN) - <15% outliers",
                 "Conservative (Standard HDBSCAN) - Natural clustering"],
                help="Choose how aggressively to assign outliers to topics"
            )

            with st.expander("🔧 Advanced Settings"):
                if "Aggressive" in outlier_strategy:
                    max_topics = max(5, len(df) // max(2, min_topic_size))
                    default_topics = min(max_topics, max(5, len(df) // 50)) if len(df) > 0 else 5
                    nr_topics = st.number_input("Initial Number of Topics", min_value=2, max_value=max_topics, value=default_topics)
                else:
                    nr_topics = min(10, len(df) // 50) if len(df) > 0 else 5

                n_neighbors = st.slider("UMAP n_neighbors", 2, 50, 15)
                n_components = st.slider("UMAP n_components", 2, 10, 5)
                batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=32)

                st.subheader("🎯 Seed Words (Optional)")
                seed_words_input = st.text_area(
                    "Enter seed words (one set per line)",
                    placeholder="Example:\nfinance, money, budget, cost\nmarketing, campaign, advertising"
                )

            if st.button("🚀 Compute Embeddings & Enable Interactive Mode", type="primary"):
                with st.spinner("Preprocessing documents…"):
                    raw_documents = df[text_col].tolist()
                    cleaned_docs, valid_indices, _ = RobustTextPreprocessor.preprocess_documents(
                        raw_documents, max_length=max_doc_length, min_length=min_doc_length, show_warnings=True
                    )
                    if not cleaned_docs:
                        st.error("No valid documents after preprocessing!")
                        st.stop()
                    st.session_state.documents = cleaned_docs
                    st.session_state.valid_indices = valid_indices
                    st.session_state.text_col = text_col
                    st.info(f"✅ Preprocessed {len(cleaned_docs):,} valid documents")

                with st.spinner("Computing embeddings (one-time)…"):
                    try:
                        _ = load_embedding_model(embedding_model)  # warm cache
                        safe_model = SafeEmbeddingModel(model_name=embedding_model, device='cuda' if torch.cuda.is_available() else 'cpu')
                        embeddings = safe_model.encode_safe(cleaned_docs, batch_size=batch_size, show_progress=True)
                        st.session_state.embeddings = embeddings
                    except Exception as e:
                        st.error(f"Embedding error: {str(e)}")
                        st.info("If this persists, lower batch size and/or max document length.")
                        st.stop()

                with st.spinner("Computing UMAP reduction…"):
                    umap_embeddings = compute_umap_embeddings(embeddings, n_neighbors, n_components)
                    st.session_state.umap_embeddings = umap_embeddings

                st.session_state.reclusterer = FastReclusterer(cleaned_docs, embeddings, umap_embeddings)

                # Parse seed words (kept for completeness)
                seed_topic_list = []
                if seed_words_input:
                    for line in seed_words_input.strip().split('\n'):
                        if line.strip():
                            words = [w.strip() for w in line.split(',')]
                            if words:
                                seed_topic_list.append(words)

                with st.spinner("Performing initial clustering…"):
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

                    topic_info = ensure_human_labels(topic_info)

                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.topic_info = topic_info
                    st.session_state.min_topic_size = min_topic_size
                    st.session_state.min_topic_size_used = min_topic_size
                    st.session_state.embeddings_computed = True
                    st.session_state.clustering_method = "K-means" if method == 'kmeans' else "HDBSCAN"
                    st.session_state.gpu_used = gpu_capabilities['cuda_available']

                    st.success("✅ Embeddings computed! You can now adjust topics dynamically.")
                    st.balloons()

    # MAIN AREA
    if st.session_state.embeddings_computed:
        st.success("✅ Embeddings ready! Use the controls below for instant topic adjustment.")
        st.header("🎚️ Dynamic Topic Adjustment")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            max_topics = max(2, min(100, len(st.session_state.documents) // max(2, st.session_state.min_topic_size)))
            n_topics_slider = st.slider("🎯 Number of Topics", min_value=2, max_value=max_topics, value=min(10, max_topics))

        with col2:
            clustering_method = st.selectbox("Method", ["K-means (Fast)", "HDBSCAN"])

        with col3:
            use_reduced = st.checkbox("Use UMAP reduction",
                                      value=st.session_state.umap_embeddings is not None,
                                      disabled=st.session_state.umap_embeddings is None,
                                      help="Faster reclustering using reduced dimensions")

        if st.button("🔄 Recluster with New Settings", type="secondary"):
            with st.spinner(f"Reclustering into {n_topics_slider} topics…"):
                method = 'kmeans' if "K-means" in clustering_method else 'hdbscan'
                topics, topic_info = st.session_state.reclusterer.recluster(
                    n_topics=n_topics_slider,
                    min_topic_size=st.session_state.min_topic_size,
                    use_reduced=use_reduced and st.session_state.umap_embeddings is not None,
                    method=method
                )
                if topics is not None:
                    topic_info = ensure_human_labels(topic_info)
                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.topic_info = topic_info
                    st.success(f"✅ Reclustered into {len(topic_info[topic_info['Topic'] != -1])} topics!")
                else:
                    st.error("Reclustering failed. Try different parameters.")

        # Results summary
        if st.session_state.current_topics is not None:
            topics = st.session_state.current_topics
            topic_info = ensure_human_labels(st.session_state.current_topic_info)

            total_docs = len(topics)
            unique_topics = len(set([t for t in topics if t != -1]))
            outlier_count = sum(1 for t in topics if t == -1)
            coverage = ((total_docs - outlier_count) / total_docs) * 100 if total_docs > 0 else 0.0

            # Build processed_df (docs + topics + labels + keywords)
            processed_df = pd.DataFrame({
                'document': st.session_state.documents,
                'topic': topics
            })
            label_map = dict(zip(topic_info['Topic'], topic_info['Human_Label']))
            kw_map = dict(zip(topic_info['Topic'], topic_info['Keywords']))
            processed_df['topic_label'] = processed_df['topic'].map(label_map).fillna('Outliers')
            processed_df['keywords'] = processed_df['topic'].map(kw_map).fillna('')

            st.session_state.processed_df = processed_df

            # Balance analysis (on non-outliers)
            balance_analyzer = CategoryBalanceAnalyzer()
            balance_analysis = balance_analyzer.analyze_distribution(list(topics), include_outliers=False)

            st.header("📊 Results Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: st.metric("Documents", f"{total_docs:,}")
            with m2: st.metric("Topics", unique_topics)
            with m3: st.metric("Outliers", f"{outlier_count:,} ({100-coverage:.1f}%)")
            with m4: st.metric("Coverage", f"{coverage:.1f}%")
            with m5: st.metric("Min Topic Size", st.session_state.get('min_topic_size_used', 'N/A'))

            if not balance_analysis['balanced']:
                st.error("⚠️ Topic Distribution Imbalance Detected")
                for w in balance_analysis['warnings']:
                    st.warning(w)

            tabs = st.tabs([
                "📊 Topics Overview",
                "📈 Distribution Analysis",
                "🗺️ Interactive Visualization",
                "📄 Topic Browser (Skim All Columns)",
                "🔍 Split Large Topics",
                "💾 Export"
            ])

            # -------- Topics Overview --------
            with tabs[0]:
                st.subheader("Topic Information")
                display_df = topic_info.copy()
                if -1 in display_df['Topic'].values:
                    display_df = display_df[display_df['Topic'] != -1]
                display_df = display_df[['Topic', 'Human_Label', 'Count', 'Keywords']].sort_values('Count', ascending=False)
                st.dataframe(display_df, use_container_width=True)
                st.caption("Dark-mode friendly: using Streamlit's default dataframe rendering.")

            # -------- Distribution Analysis --------
            with tabs[1]:
                st.subheader("📈 Topic Distribution")
                counts = processed_df[processed_df['topic'] != -1]['topic'].value_counts().sort_values(ascending=False)
                viz_df = pd.DataFrame({'Topic': counts.index, 'Count': counts.values})
                viz_df['Label'] = viz_df['Topic'].map(label_map).fillna(viz_df['Topic'].apply(lambda x: f"Topic {x}"))
                fig = px.bar(viz_df, x='Label', y='Count', title='Document Distribution Across Topics')
                fig.update_layout(xaxis_title="Topic", yaxis_title="Documents")
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Balance Score", f"{balance_analysis['balance_score']:.2f}")
                with c2: st.metric("Oversized Topics", len(balance_analysis['oversized_topics']))
                with c3:
                    avg_size = total_docs / max(1, unique_topics)
                    st.metric("Avg Topic Size", f"{avg_size:.0f} docs")

            # -------- Interactive Visualization --------
            with tabs[2]:
                st.subheader("🗺️ 2D Visualization (UMAP/PCA)")
                if st.session_state.umap_embeddings is not None and st.session_state.umap_embeddings.shape[1] >= 2:
                    coords = st.session_state.umap_embeddings[:, :2]
                else:
                    # Fallback to PCA
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(st.session_state.embeddings)
                viz = pd.DataFrame({
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    'Topic': topics,
                    'Label': [label_map.get(t, 'Outliers') for t in topics],
                    'Text': st.session_state.documents
                })
                # Sample if huge
                sample = viz.sample(5000, random_state=42) if len(viz) > 5000 else viz
                fig = px.scatter(sample, x='x', y='y', color='Label',
                                 hover_data={'Text': True, 'Topic': True}, title='Document Clusters')
                st.plotly_chart(fig, use_container_width=True)

            # -------- Topic Browser (Skim All Columns) --------
            with tabs[3]:
                st.subheader("📄 Topic Browser — Skim Full Rows Quickly")
                # Build export_df aligning topics back to original df rows
                export_df = st.session_state.df.copy()
                full_topics = np.full(len(export_df), -1)
                for i, orig_idx in enumerate(st.session_state.valid_indices):
                    if i < len(topics):
                        full_topics[orig_idx] = topics[i]
                export_df['Topic'] = full_topics
                export_df['Topic_Label'] = export_df['Topic'].map(label_map).fillna('Outliers')
                export_df['Topic_Keywords'] = export_df['Topic'].map(kw_map).fillna('')

                # Let user choose topic with human labels
                ordered_topics = display_df['Topic'].tolist() if 'display_df' in locals() else sorted([t for t in set(topics) if t != -1])
                options = [(t, label_map.get(t, f"Topic {t}")) for t in ordered_topics]
                opt_labels = [f"{lab}  (#{t})" for t, lab in options]
                selection = st.selectbox("Select a topic to skim", opt_labels, index=0 if opt_labels else None)

                if selection:
                    # parse selected topic id from label
                    sel_t = int(re.search(r"\(#(\-?\d+)\)", selection).group(1))
                    filtered = export_df[export_df['Topic'] == sel_t].copy()
                    # Show full dataframe with the chosen text column front-and-center
                    if st.session_state.text_col in filtered.columns:
                        # move text column to front for quick skimming
                        cols = [st.session_state.text_col] + [c for c in filtered.columns if c != st.session_state.text_col]
                        filtered = filtered[cols]
                    st.write(f"Showing **{len(filtered):,}** rows for **{label_map.get(sel_t, f'Topic {sel_t}') }**")
                    st.dataframe(filtered, use_container_width=True, height=500)

            # -------- Split Large Topics --------
            with tabs[4]:
                st.subheader("🔍 Split Large Topics")
                oversized = balance_analysis['oversized_topics']
                if oversized:
                    st.warning(f"Found {len(oversized)} oversized topic(s)")
                    choices = [f"{label_map.get(t['topic'], f'Topic {t['topic']}')} (# {t['topic']}): {t['count']} docs ({t['ratio']*100:.1f}%)"
                               for t in oversized]
                    selected = st.selectbox("Select topic to split", choices)
                    if selected:
                        sel_id = int(re.search(r"# (\-?\d+)\)", selected).group(1))
                        suggested = next(x['suggested_splits'] for x in oversized if x['topic'] == sel_id)
                        sub_n = st.slider("Number of subtopics", 2, min(10, suggested * 2), suggested)
                        if st.button(f"🔍 Split “{label_map.get(sel_id, f'Topic {sel_id}')}” into {sub_n} subtopics"):
                            docs_to_split = [doc for doc, t in zip(st.session_state.documents, topics) if t == sel_id]
                            if len(docs_to_split) >= max(10, sub_n * 2):
                                with st.spinner(f"Analyzing {len(docs_to_split):,} documents…"):
                                    sub_model = BERTopic(
                                        hdbscan_model=GPUKMeans(n_clusters=sub_n) if torch.cuda.is_available() else KMeans(n_clusters=sub_n, random_state=42),
                                        min_topic_size=max(2, len(docs_to_split) // max(2, sub_n)),
                                        calculate_probabilities=False,
                                        verbose=False
                                    )
                                    sub_topics, _ = sub_model.fit_transform(docs_to_split)
                                    st.success(f"✅ Split into {len(set(sub_topics))} subtopics (incl. -1 outliers)")
                                    sub_info = sub_model.get_topic_info()
                                    # Normalize into Human_Label for subtopics too
                                    if 'Representation' in sub_info.columns:
                                        sub_info['Keywords'] = sub_info['Representation'].apply(
                                            lambda x: ', '.join(x[:5]) if isinstance(x, list) else str(x)
                                        )
                                    if 'Name' not in sub_info.columns:
                                        sub_info['Name'] = sub_info['Topic'].apply(lambda t: f"Subtopic {t}")
                                    sub_info = ensure_human_labels(sub_info)
                                    show_cols = ['Topic', 'Human_Label', 'Count', 'Keywords']
                                    show_cols = [c for c in show_cols if c in sub_info.columns]
                                    st.dataframe(sub_info[show_cols], use_container_width=True)
                            else:
                                st.warning("Not enough documents in this topic to split reliably.")
                else:
                    st.success("✅ No oversized categories detected. All topics are well-balanced!")

            # -------- Export --------
            with tabs[5]:
                st.subheader("💾 Export Results")
                export_df = st.session_state.df.copy()
                full_topics = np.full(len(export_df), -1)
                for i, orig_idx in enumerate(st.session_state.valid_indices):
                    if i < len(topics):
                        full_topics[orig_idx] = topics[i]
                export_df['Topic'] = full_topics
                export_df['Topic_Label'] = export_df['Topic'].map(label_map).fillna('Outliers')
                export_df['Topic_Keywords'] = export_df['Topic'].map(kw_map).fillna('N/A')
                export_df['Valid_Document'] = [i in st.session_state.valid_indices for i in range(len(export_df))]

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=convert_df_to_csv(export_df),
                        file_name=f"bertopic_results_{st.session_state.uploaded_file_name}_{unique_topics}topics.csv",
                        mime="text/csv"
                    )
                with c2:
                    topic_out = topic_info[['Topic', 'Human_Label', 'Count', 'Keywords']].copy()
                    st.download_button(
                        label="📥 Download Topic Info (CSV)",
                        data=convert_df_to_csv(topic_out),
                        file_name=f"topic_info_{st.session_state.uploaded_file_name}_{unique_topics}topics.csv",
                        mime="text/csv"
                    )
                with c3:
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
Balance Status: {'Balanced' if balance_analysis['balanced'] else 'Needs Attention'}
Oversized Categories: {len(balance_analysis['oversized_topics'])}
"""
                    st.download_button(
                        label="📥 Download Report (TXT)",
                        data=summary.encode('utf-8'),
                        file_name=f"report_{st.session_state.uploaded_file_name}_{unique_topics}topics.txt",
                        mime="text/plain"
                    )

    elif st.session_state.df is None:
        st.info("👆 Please upload a CSV file in the sidebar to begin.")
        st.header("🚀 Complete Feature Set")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("⚡ Interactive")
            st.markdown("- Dynamic topic slider\n- One-time embedding\n- Fast reclustering\n- Live viz")
        with c2:
            st.subheader("🛡️ Robust")
            st.markdown("- Edge-case preprocessing\n- Memory fallback\n- Safe padding/truncation\n- Progress bars")
        with c3:
            st.subheader("📊 Tools")
            st.markdown("- Split large topics\n- Balance analysis\n- Seed words (optional)\n- Rich exports")

if __name__ == "__main__":
    main()
