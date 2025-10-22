import streamlit as st
import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit to wide mode for better layout
st.set_page_config(layout="wide", page_title="Windows CUDA-Optimized BERTopic", page_icon="🚀")

# Make accelerate optional - not strictly required
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False
    # Don't stop the app - just warn
    st.warning("""
    ⚠️ **Optional: 'accelerate' package not found**
    
    For better GPU performance with some models, consider installing:
    ```bash
    pip install accelerate
    ```
    The app will work without it, but may be slower in some cases.
    """)

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
# CATEGORY BALANCE ANALYZER
# -----------------------------------------------------
class CategoryBalanceAnalyzer:
    """Analyzes topic distribution and suggests splits for oversized categories"""
    
    def __init__(self, min_topic_ratio=0.30, ideal_max_ratio=0.20):
        """
        Args:
            min_topic_ratio: If a topic has more than this ratio of docs, flag it
            ideal_max_ratio: Ideal maximum ratio for any single topic
        """
        self.min_topic_ratio = min_topic_ratio
        self.ideal_max_ratio = ideal_max_ratio
    
    def analyze_distribution(self, labels, include_outliers=False):
        """
        Analyze the distribution of documents across topics
        
        Returns:
            dict with analysis results
        """
        total_docs = len(labels)
        unique_topics = set(labels)
        
        # Remove outliers from analysis if requested
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
        
        # Calculate balance metrics
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
    def suggest_parameters(num_docs, current_outlier_ratio):
        """
        Suggest better parameters to reduce outliers
        
        Args:
            num_docs: Number of documents
            current_outlier_ratio: Current outlier percentage (0-1)
        
        Returns:
            dict of suggested parameters
        """
        suggestions = {
            'method': '',
            'parameters': {},
            'explanation': ''
        }
        
        if current_outlier_ratio > 0.3:
            # High outliers - use K-means
            suggestions['method'] = 'kmeans'
            suggestions['parameters'] = {
                'use_kmeans': True,
                'use_gpu_kmeans': True,
                'nr_topics': max(5, num_docs // 50),  # More topics
                'min_topic_size': max(2, num_docs // 100)  # Smaller min size
            }
            suggestions['explanation'] = "High outliers detected. K-means will assign all documents to topics."
            
        elif current_outlier_ratio > 0.15:
            # Moderate outliers - adjust HDBSCAN
            suggestions['method'] = 'hdbscan'
            suggestions['parameters'] = {
                'use_kmeans': False,
                'min_cluster_size': 3,  # Smaller clusters
                'min_samples': 1,  # More lenient
                'n_components': 15,  # More dimensions
                'min_topic_size': 3
            }
            suggestions['explanation'] = "Moderate outliers. Using lenient HDBSCAN parameters."
            
        else:
            # Low outliers - current settings are good
            suggestions['method'] = 'current'
            suggestions['parameters'] = {}
            suggestions['explanation'] = "Outlier ratio is acceptable. Current settings are working well."
        
        return suggestions
    
    @staticmethod
    def reassign_outliers_to_nearest(model, docs, embeddings, labels, top_k=3):
        """
        Reassign outliers to nearest topics
        
        Args:
            model: BERTopic model
            docs: List of documents
            embeddings: Document embeddings (numpy array)
            labels: Current topic labels
            top_k: Number of nearest topics to consider
        
        Returns:
            Updated labels with outliers reassigned
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        outlier_indices = [i for i, label in enumerate(labels) if label == -1]
        
        if len(outlier_indices) == 0:
            return labels
        
        # Try to get topic embeddings
        try:
            if hasattr(model, 'topic_embeddings_') and model.topic_embeddings_ is not None:
                topic_embeddings = model.topic_embeddings_
                # Skip outlier topic if it's at index 0
                if -1 in model.get_topic_info()['Topic'].values:
                    topic_embeddings = topic_embeddings[1:]
            else:
                # No topic embeddings available
                return labels
        except Exception as e:
            # If anything goes wrong, just return original labels
            return labels
        
        if len(topic_embeddings) == 0:
            return labels
        
        # Ensure embeddings is a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Calculate similarity between outlier docs and topics
        new_labels = labels.copy()
        
        for idx in outlier_indices:
            doc_embedding = embeddings[idx].reshape(1, -1)
            similarities = cosine_similarity(doc_embedding, topic_embeddings)[0]
            
            # Assign to most similar topic if similarity is above threshold
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]
            
            if max_sim > 0.3:  # Threshold for reassignment
                new_labels[idx] = max_sim_idx  # Topic indices start at 0 for non-outliers
        
        return new_labels

# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def safe_extract_embeddings(model, docs, verbose=False):
    """
    Safely extract embeddings from documents, trying multiple methods.
    
    Args:
        model: BERTopic model
        docs: List of documents
        verbose: Print debug information
    
    Returns:
        numpy array of embeddings
    """
    try:
        # Method 1: Use BERTopic's internal method (newest versions)
        if verbose:
            st.info("Trying method 1: BERTopic internal extraction...")
        embeddings = model._extract_embeddings(docs, method="document")
        if verbose:
            st.success("✅ Method 1 succeeded")
        return embeddings
    except Exception as e1:
        if verbose:
            st.warning(f"Method 1 failed: {e1}")
        
        try:
            # Method 2: Access the underlying embedding model
            if verbose:
                st.info("Trying method 2: Direct model access...")
            
            if hasattr(model, 'embedding_model'):
                emb_model = model.embedding_model
                
                # If it's a backend wrapper, get the underlying model
                if hasattr(emb_model, 'embedding_model'):
                    emb_model = emb_model.embedding_model
                
                embeddings = emb_model.encode(docs, show_progress_bar=False)
                if verbose:
                    st.success("✅ Method 2 succeeded")
                return embeddings
        except Exception as e2:
            if verbose:
                st.warning(f"Method 2 failed: {e2}")
        
        # Method 3: Create fresh embedding model (most reliable fallback)
        if verbose:
            st.info("Trying method 3: Fresh embedding model...")
        
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        temp_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        embeddings = temp_model.encode(docs, show_progress_bar=False)
        del temp_model
        
        if verbose:
            st.success("✅ Method 3 succeeded (fallback)")
        
        return embeddings

# -----------------------------------------------------
# GPU-ACCELERATED CLUSTERING FOR WINDOWS
# -----------------------------------------------------
class GPUKMeans:
    """GPU-accelerated K-means using PyTorch for Windows"""
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, device='cuda'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.cluster_centers_ = None
        self.labels_ = None
        
    def fit(self, X):
        """Fit K-means using GPU acceleration"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        if self.device == 'cuda' and not X.is_cuda:
            X = X.cuda()
        
        n_samples = X.shape[0]
        
        # Initialize cluster centers using K-means++
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.cluster_centers_ = X[indices].clone()
        
        for iteration in range(self.max_iter):
            # Assign points to nearest cluster
            distances = torch.cdist(X, self.cluster_centers_)
            self.labels_ = torch.argmin(distances, dim=1)
            
            # Update cluster centers
            new_centers = torch.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if mask.sum() > 0:
                    new_centers[k] = X[mask].mean(dim=0)
                else:
                    # Handle empty clusters
                    new_centers[k] = self.cluster_centers_[k]
            
            # Check convergence
            shift = torch.norm(new_centers - self.cluster_centers_)
            self.cluster_centers_ = new_centers
            
            if shift < self.tol:
                break
        
        # Convert labels back to CPU numpy
        self.labels_ = self.labels_.cpu().numpy()
        return self
    
    def fit_predict(self, X):
        """Fit and return labels"""
        self.fit(X)
        return self.labels_

class WindowsOptimizedHDBSCAN:
    """HDBSCAN with GPU-accelerated distance calculations for Windows"""
    def __init__(self, min_cluster_size=5, min_samples=1, metric='euclidean', **kwargs):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.kwargs = kwargs
        self.labels_ = None
        
    def fit(self, X):
        """Fit HDBSCAN with GPU-accelerated distance matrix if possible"""
        if torch.cuda.is_available() and len(X) < 50000:  # Use GPU for reasonable sizes
            try:
                # Compute distance matrix on GPU
                X_tensor = torch.from_numpy(X).float().cuda()
                # Try to use mixed precision if available
                try:
                    with torch.cuda.amp.autocast():  # Use mixed precision for speed
                        dist_matrix = torch.cdist(X_tensor, X_tensor).cpu().numpy()
                except:
                    # Fallback to regular precision if amp fails
                    dist_matrix = torch.cdist(X_tensor, X_tensor).cpu().numpy()
                
                # Run HDBSCAN on CPU with precomputed distances
                from sklearn.cluster import HDBSCAN as skHDBSCAN
                clusterer = skHDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric='precomputed',
                    **self.kwargs
                )
                self.labels_ = clusterer.fit_predict(dist_matrix)
            except Exception as e:
                st.warning(f"GPU clustering failed, falling back to CPU: {e}")
                # Fall back to regular HDBSCAN
                clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    cluster_selection_method='leaf',
                    **self.kwargs
                )
                self.labels_ = clusterer.fit_predict(X)
        else:
            # Fall back to regular HDBSCAN
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_method='leaf',
                **self.kwargs
            )
            self.labels_ = clusterer.fit_predict(X)
        return self
    
    def fit_predict(self, X):
        """Fit and return labels"""
        self.fit(X)
        return self.labels_

# -----------------------------------------------------
# OPTIMIZED MODEL CREATION WITH BETTER OUTLIER HANDLING
# -----------------------------------------------------
@st.cache_resource(show_spinner=False)
def create_optimized_bertopic_model(
    min_topic_size,
    nr_topics,
    min_cluster_size,
    min_samples,
    n_neighbors,
    n_components,
    num_docs=None,
    use_kmeans=False,
    use_gpu_kmeans=True,
    calculate_probabilities=False,
    embedding_batch_size=32,
    use_fp16=False
):
    """
    Creates a Windows-optimized BERTopic model with better outlier handling.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Embedding model with GPU support
    if use_fp16 and device == "cuda":
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        embedding_model.half()  # Use FP16
    else:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Dimensionality reduction
    if cuml_available and device == "cuda":
        umap_model = cumlUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
    else:
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            low_memory=False
        )
    
    # Clustering - choose based on outlier reduction strategy
    if use_kmeans:
        # K-means has NO outliers - all docs assigned to topics
        if use_gpu_kmeans and device == "cuda":
            n_clusters = nr_topics if isinstance(nr_topics, int) else max(8, num_docs // 50 if num_docs else 8)
            hdbscan_model = GPUKMeans(n_clusters=n_clusters, device=device)
        else:
            n_clusters = nr_topics if isinstance(nr_topics, int) else max(8, num_docs // 50 if num_docs else 8)
            hdbscan_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        # HDBSCAN - optimize to reduce outliers
        if cuml_available and device == "cuda":
            hdbscan_model = cumlHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                gen_min_span_tree=True,
                prediction_data=calculate_probabilities
            )
        else:
            hdbscan_model = WindowsOptimizedHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
    
    # Vectorizer
    vectorizer_model = CountVectorizer(
        min_df=2,
        max_df=0.95,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Representation models for better topic descriptions
    representation_models = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3)
    ]
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_models,
        top_n_words=10,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=calculate_probabilities,
        verbose=False
    )
    
    return topic_model

# -----------------------------------------------------
# GPU CAPABILITY CHECKER
# -----------------------------------------------------
def check_gpu_capabilities():
    """Check what GPU capabilities are available"""
    caps = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cupy_available': cupy_available,
        'cuml_available': cuml_available,
        'faiss_available': faiss_available,
        'faiss_gpu_available': faiss_gpu_available,
        'accelerate_available': accelerate_available,
        'torch_version': torch.__version__
    }
    
    if torch.cuda.is_available():
        caps['gpu_memory_total'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        caps['gpu_memory_allocated'] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
    
    return caps

@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

# -----------------------------------------------------
# MAIN APP
# -----------------------------------------------------
def main():
    st.title("🚀 Windows CUDA-Optimized BERTopic with Smart Category Balancing")
    st.markdown("*Maximum GPU utilization with intelligent outlier reduction and category splitting*")
    
    # Check GPU status
    caps = check_gpu_capabilities()
    gpu_status = "🟢 GPU Enabled" if caps['cuda_available'] else "🔴 CPU Only"
    st.sidebar.markdown(f"### {gpu_status}")
    
    if caps['cuda_available']:
        st.sidebar.success(f"**{caps['device_name']}**\n{caps['gpu_memory_total']}")
    
    # File uploader
    st.sidebar.header("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with a text column to analyze"
    )
    
    if uploaded_file:
        # Store filename in session state
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = uploaded_file.name.replace('.csv', '')
        
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df):,} rows")
            
            # Column selection
            text_col = st.sidebar.selectbox(
                "Select text column",
                options=df.columns.tolist(),
                help="Choose the column containing text to analyze"
            )
            
            # Preview data
            with st.expander("📋 Preview Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Model Parameters
            st.sidebar.header("⚙️ Model Parameters")
            
            # Outlier reduction preset
            outlier_strategy = st.sidebar.selectbox(
                "Outlier Reduction Strategy",
                options=[
                    "Aggressive (K-means, no outliers)",
                    "Moderate (Lenient HDBSCAN)",
                    "Conservative (Standard HDBSCAN)",
                    "Custom"
                ],
                index=0,
                help="Choose how aggressively to reduce outliers"
            )
            
            # Set parameters based on strategy
            if outlier_strategy == "Aggressive (K-means, no outliers)":
                use_kmeans = True
                min_cluster_size = 2
                min_samples = 1
                min_topic_size = max(2, len(df) // 100)
                nr_topics = max(8, len(df) // 50)
                n_components = 10
                n_neighbors = 10
            elif outlier_strategy == "Moderate (Lenient HDBSCAN)":
                use_kmeans = False
                min_cluster_size = 3
                min_samples = 1
                min_topic_size = 3
                nr_topics = "auto"
                n_components = 15
                n_neighbors = 15
            elif outlier_strategy == "Conservative (Standard HDBSCAN)":
                use_kmeans = False
                min_cluster_size = 5
                min_samples = 3
                min_topic_size = 5
                nr_topics = "auto"
                n_components = 5
                n_neighbors = 15
            else:  # Custom
                use_kmeans = st.sidebar.checkbox(
                    "Use K-means (no outliers)",
                    value=False,
                    help="K-means assigns ALL documents to topics (0% outliers)"
                )
                
                if use_kmeans:
                    nr_topics = st.sidebar.number_input(
                        "Number of Topics",
                        min_value=2,
                        max_value=100,
                        value=max(8, len(df) // 50),
                        help="How many topics to create"
                    )
                else:
                    nr_topics_auto = st.sidebar.checkbox("Auto-determine topics", value=True)
                    if not nr_topics_auto:
                        nr_topics = st.sidebar.number_input(
                            "Number of Topics",
                            min_value=2,
                            max_value=100,
                            value=10
                        )
                    else:
                        nr_topics = "auto"
                
                min_topic_size = st.sidebar.slider(
                    "Min Topic Size",
                    min_value=2,
                    max_value=50,
                    value=5,
                    help="Minimum documents per topic"
                )
                
                if not use_kmeans:
                    min_cluster_size = st.sidebar.slider(
                        "Min Cluster Size (HDBSCAN)",
                        min_value=2,
                        max_value=50,
                        value=5,
                        help="Smaller = more topics, fewer outliers"
                    )
                    
                    min_samples = st.sidebar.slider(
                        "Min Samples (HDBSCAN)",
                        min_value=1,
                        max_value=20,
                        value=1,
                        help="Lower = more lenient clustering"
                    )
                else:
                    min_cluster_size = 5
                    min_samples = 1
                
                n_components = st.sidebar.slider(
                    "UMAP Components",
                    min_value=2,
                    max_value=50,
                    value=10,
                    help="More = better separation, fewer outliers"
                )
                
                n_neighbors = st.sidebar.slider(
                    "UMAP Neighbors",
                    min_value=5,
                    max_value=50,
                    value=15
                )
            
            # GPU optimization options
            with st.sidebar.expander("🚀 GPU Optimization"):
                use_gpu_kmeans = st.checkbox(
                    "Use GPU K-means",
                    value=True,
                    disabled=not torch.cuda.is_available(),
                    help="Faster clustering on GPU"
                )
                
                use_fp16 = st.checkbox(
                    "Use FP16 (half precision)",
                    value=False,
                    disabled=not torch.cuda.is_available(),
                    help="2x faster embeddings, may reduce quality slightly"
                )
                
                embedding_batch_size = st.slider(
                    "Embedding Batch Size",
                    min_value=8,
                    max_value=256,
                    value=32,
                    step=8,
                    help="Higher = faster but uses more memory"
                )
            
            # Category balance monitoring
            with st.sidebar.expander("📊 Category Balance Settings"):
                max_category_ratio = st.slider(
                    "Max Category Size (%)",
                    min_value=10,
                    max_value=50,
                    value=30,
                    help="Warn if any category exceeds this % of documents"
                )
                
                auto_suggest_split = st.checkbox(
                    "Auto-suggest splits for large categories",
                    value=True,
                    help="Automatically suggest splitting oversized categories"
                )
            
            # Run analysis
            if st.sidebar.button("🔬 Run Analysis", type="primary"):
                with st.spinner("Processing..."):
                    # Prepare documents
                    docs = df[text_col].fillna("").astype(str).tolist()
                    
                    # Filter out empty docs
                    docs = [doc for doc in docs if len(doc.strip()) > 0]
                    
                    if len(docs) < 10:
                        st.error("Not enough documents to analyze. Need at least 10 non-empty documents.")
                        return
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Create model
                    status_text.text("Creating optimized model...")
                    progress_bar.progress(20)
                    
                    model = create_optimized_bertopic_model(
                        min_topic_size=min_topic_size,
                        nr_topics=nr_topics,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        n_neighbors=n_neighbors,
                        n_components=n_components,
                        num_docs=len(docs),
                        use_kmeans=use_kmeans,
                        use_gpu_kmeans=use_gpu_kmeans,
                        calculate_probabilities=False,
                        embedding_batch_size=embedding_batch_size,
                        use_fp16=use_fp16
                    )
                    
                    # Step 2: Fit model
                    status_text.text("Fitting model and generating topics...")
                    progress_bar.progress(40)
                    
                    topics, probs = model.fit_transform(docs)
                    
                    # Step 3: Get embeddings for outlier reassignment
                    status_text.text("Analyzing document embeddings...")
                    progress_bar.progress(60)
                    
                    # Use safe extraction method
                    embeddings = safe_extract_embeddings(model, docs, verbose=False)
                    
                    # Step 4: Analyze category balance
                    status_text.text("Analyzing category balance...")
                    progress_bar.progress(70)
                    
                    analyzer = CategoryBalanceAnalyzer(
                        min_topic_ratio=max_category_ratio / 100,
                        ideal_max_ratio=0.20
                    )
                    balance_analysis = analyzer.analyze_distribution(topics, include_outliers=False)
                    
                    # Step 5: Try to reduce outliers if needed
                    outlier_count = sum(1 for t in topics if t == -1)
                    outlier_ratio = outlier_count / len(topics)
                    
                    if outlier_ratio > 0.10 and not use_kmeans:
                        status_text.text("Attempting to reassign outliers...")
                        progress_bar.progress(80)
                        
                        try:
                            reducer = OutlierReducer()
                            new_topics = reducer.reassign_outliers_to_nearest(
                                model, docs, embeddings, topics
                            )
                            
                            # Only update if we successfully reduced outliers
                            new_outlier_count = sum(1 for t in new_topics if t == -1)
                            if new_outlier_count < outlier_count:
                                topics = new_topics
                                outlier_count = new_outlier_count
                                outlier_ratio = outlier_count / len(topics)
                                status_text.text(f"✅ Reduced outliers by {outlier_count - new_outlier_count}")
                        except Exception as e:
                            # If reassignment fails, continue with original topics
                            st.warning(f"Could not reassign outliers: {e}. Continuing with original results.")
                            pass
                    
                    # Step 6: Prepare results
                    status_text.text("Preparing results...")
                    progress_bar.progress(90)
                    
                    # Create results dataframe
                    processed_df = df.copy()
                    processed_df['topic'] = topics[:len(df)]
                    
                    # Get topic info
                    topic_info = model.get_topic_info()
                    
                    # Create readable topic labels
                    topic_labels = {}
                    for _, row in topic_info.iterrows():
                        topic_id = row['Topic']
                        if topic_id == -1:
                            topic_labels[topic_id] = "Outliers"
                        else:
                            # Get top 3 words
                            words = row['Representation'][:3] if isinstance(row['Representation'], list) else []
                            topic_labels[topic_id] = f"Topic {topic_id}: {', '.join(words)}"
                    
                    processed_df['topic_label'] = processed_df['topic'].map(topic_labels)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.topics = topics
                    st.session_state.processed_df = processed_df
                    st.session_state.topic_info = topic_info
                    st.session_state.embeddings = embeddings
                    st.session_state.text_col = text_col
                    st.session_state.balance_analysis = balance_analysis
                    st.session_state.outlier_ratio = outlier_ratio
                    
                    # Store run parameters
                    st.session_state.ran_params = {
                        'GPU Used': torch.cuda.is_available(),
                        'Clustering': 'GPU K-means' if use_kmeans and use_gpu_kmeans else 'K-means' if use_kmeans else 'HDBSCAN',
                        'FP16': use_fp16,
                        'Documents': len(docs),
                        'Outlier Strategy': outlier_strategy
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    st.success("Analysis complete!")
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()
    
    # Display results if available
    if 'processed_df' in st.session_state:
        model = st.session_state.model
        processed_df = st.session_state.processed_df
        topic_info = st.session_state.topic_info
        topics = st.session_state.topics
        text_col = st.session_state.text_col
        ran_params = st.session_state.ran_params
        balance_analysis = st.session_state.balance_analysis
        outlier_ratio = st.session_state.outlier_ratio
        
        # Summary metrics
        st.header("📊 Results Summary")
        
        unique_topics = len(set(topics)) - (1 if -1 in topics else 0)
        outlier_count = sum(1 for t in topics if t == -1)
        coverage = ((len(topics) - outlier_count) / len(topics)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents", f"{len(processed_df):,}")
        with col2:
            st.metric("Topics Found", unique_topics)
        with col3:
            # Color code outliers
            outlier_pct = 100 - coverage
            outlier_color = "🟢" if outlier_pct < 10 else "🟡" if outlier_pct < 25 else "🔴"
            st.metric(
                "Outliers",
                f"{outlier_count:,}",
                delta=f"{outlier_pct:.1f}%",
                delta_color="inverse"
            )
        with col4:
            st.metric("Coverage", f"{coverage:.1f}%")
        
        # Category balance warnings
        if not balance_analysis['balanced']:
            st.warning("⚠️ **Category Balance Issues Detected**")
            
            for warning in balance_analysis['warnings']:
                st.warning(f"📌 {warning}")
            
            if auto_suggest_split and balance_analysis['oversized_topics']:
                st.info("""
                💡 **Suggestion:** Some categories are too large and should be split.
                Use the 'Split Topics' tab below to analyze these categories further.
                """)
        else:
            st.success("✅ **Category distribution is well-balanced!**")
        
        # Outlier improvement suggestions
        if outlier_ratio > 0.15:
            reducer = OutlierReducer()
            suggestions = reducer.suggest_parameters(len(topics), outlier_ratio)
            
            if suggestions['method'] != 'current':
                st.info(f"""
                💡 **Outlier Reduction Suggestion:**
                
                {suggestions['explanation']}
                
                Try changing your settings in the sidebar to:
                - Strategy: {'Aggressive (K-means, no outliers)' if suggestions['method'] == 'kmeans' else 'Moderate (Lenient HDBSCAN)'}
                """)
        
        # Run parameters
        with st.expander("⚙️ Run Parameters"):
            cols = st.columns(len(ran_params))
            for col, (key, value) in zip(cols, ran_params.items()):
                with col:
                    st.metric(key, value)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Topic Overview",
            "🔍 Document Explorer",
            "✂️ Split Large Topics",
            "💾 Export"
        ])
        
        with tab1:
            st.subheader("Topic Distribution")
            
            # Topic size distribution
            topic_sizes = processed_df[processed_df['topic'] != -1]['topic'].value_counts().sort_index()
            
            if len(topic_sizes) > 0:
                import plotly.express as px
                
                fig = px.bar(
                    x=topic_sizes.index,
                    y=topic_sizes.values,
                    labels={'x': 'Topic', 'y': 'Document Count'},
                    title='Documents per Topic'
                )
                
                # Add warning line for oversized categories
                if max_category_ratio:
                    max_size = len(processed_df) * (max_category_ratio / 100)
                    fig.add_hline(
                        y=max_size,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Max Recommended ({max_category_ratio}%)"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Topic details table
            st.subheader("Topic Details")
            
            # Enhance topic info with size warnings
            topic_display = topic_info[['Topic', 'Count', 'Name']].copy()
            topic_display['Keywords'] = topic_info['Representation'].apply(
                lambda x: ', '.join(x[:5]) if isinstance(x, list) else str(x)[:100]
            )
            topic_display['% of Docs'] = (topic_display['Count'] / len(processed_df) * 100).round(1)
            
            # Add warning column
            max_size = len(processed_df) * (max_category_ratio / 100)
            topic_display['Status'] = topic_display.apply(
                lambda row: '🔴 Too Large' if row['Count'] > max_size and row['Topic'] != -1
                else '🟢 Good' if row['Topic'] != -1
                else '⚪ Outliers',
                axis=1
            )
            
            # Reorder columns
            topic_display = topic_display[['Status', 'Topic', 'Count', '% of Docs', 'Name', 'Keywords']]
            
            st.dataframe(topic_display, use_container_width=True, height=400)
        
        with tab2:
            st.subheader("Explore Documents by Topic")
            
            # Topic selector
            topic_options = ["All Topics"] + [
                f"{topic_labels[t]}" for t in sorted(set(topics)) if t in topic_labels
            ]
            
            selected_topic_label = st.selectbox(
                "Select Topic",
                options=topic_options
            )
            
            # Filter documents
            if selected_topic_label == "All Topics":
                filtered_df = processed_df
            else:
                filtered_df = processed_df[processed_df['topic_label'] == selected_topic_label]
            
            st.write(f"**{len(filtered_df):,} documents**")
            
            # Display documents
            display_cols = [col for col in [text_col, 'topic_label'] if col in filtered_df.columns]
            st.dataframe(
                filtered_df[display_cols].head(100),
                use_container_width=True,
                height=500
            )
        
        with tab3:
            st.subheader("Split Large Topics into Subtopics")
            
            st.info("""
            Use this tool to analyze large topics and split them into more specific subtopics.
            This is useful when one category contains too many diverse documents.
            """)
            
            # Get topics that are too large
            max_size = len(processed_df) * (max_category_ratio / 100)
            large_topics = processed_df[
                (processed_df['topic'] != -1) &
                (processed_df.groupby('topic')['topic'].transform('count') > max_size)
            ]['topic_label'].unique()
            
            if len(large_topics) > 0:
                st.warning(f"**{len(large_topics)} oversized categories detected:**")
                for topic in large_topics:
                    count = len(processed_df[processed_df['topic_label'] == topic])
                    pct = count / len(processed_df) * 100
                    st.write(f"- {topic}: {count:,} docs ({pct:.1f}%)")
                
                selected_topic_to_split = st.selectbox(
                    "Select topic to split",
                    options=large_topics
                )
                
                # Get suggested number of splits
                topic_count = len(processed_df[processed_df['topic_label'] == selected_topic_to_split])
                topic_ratio = topic_count / len(processed_df)
                suggested_splits = max(2, int(topic_ratio / 0.20))
                
                sub_n_topics = st.slider(
                    "Number of subtopics",
                    min_value=2,
                    max_value=20,
                    value=suggested_splits,
                    help="How many subtopics to create from this large topic"
                )
                
                if st.button(f"🔍 Split '{selected_topic_to_split}' into {sub_n_topics} subtopics"):
                    docs_to_split = processed_df[
                        processed_df['topic_label'] == selected_topic_to_split
                    ][text_col].tolist()
                    
                    if len(docs_to_split) >= 10:
                        with st.spinner(f"Analyzing {len(docs_to_split):,} documents..."):
                            # Use K-means for splitting to ensure all docs are assigned
                            if torch.cuda.is_available():
                                sub_model = BERTopic(
                                    hdbscan_model=GPUKMeans(n_clusters=sub_n_topics),
                                    min_topic_size=2,
                                    calculate_probabilities=False,
                                    verbose=False
                                )
                            else:
                                sub_model = BERTopic(
                                    hdbscan_model=KMeans(n_clusters=sub_n_topics, random_state=42),
                                    min_topic_size=2,
                                    calculate_probabilities=False,
                                    verbose=False
                                )
                            
                            sub_topics, _ = sub_model.fit_transform(docs_to_split)
                            
                            st.success(f"✅ Split into {len(set(sub_topics))} subtopics")
                            
                            st.write("### 📊 Subtopics Found:")
                            sub_topic_info = sub_model.get_topic_info()
                            
                            # Clean display
                            sub_topic_display = sub_topic_info[['Topic', 'Count', 'Name']].copy()
                            sub_topic_display['Keywords'] = sub_topic_info['Representation'].apply(
                                lambda x: ', '.join(x[:3]) if isinstance(x, list) else str(x)[:50]
                            )
                            sub_topic_display['% of Parent'] = (
                                sub_topic_display['Count'] / len(docs_to_split) * 100
                            ).round(1)
                            
                            st.dataframe(sub_topic_display, use_container_width=True)
                            
                            # Summary
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
                        st.warning(f"Not enough documents ({len(docs_to_split)} < 10) for splitting")
            else:
                st.success("✅ No oversized categories detected. All topics are well-balanced!")
        
        with tab4:
            st.subheader("💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=convert_df_to_csv(processed_df),
                    file_name=f"bertopic_results_{st.session_state.uploaded_file_name}.csv",
                    mime="text/csv",
                    help="Full dataset with topic assignments"
                )
            
            with col2:
                st.download_button(
                    label="📥 Download Topic Info (CSV)",
                    data=convert_df_to_csv(topic_info),
                    file_name=f"topic_info_{st.session_state.uploaded_file_name}.csv",
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
GPU Used: {ran_params.get('GPU Used', False)}
Clustering Method: {ran_params.get('Clustering', 'Unknown')}
Balance Status: {'Balanced' if balance_analysis['balanced'] else 'Needs Attention'}
Oversized Categories: {len(balance_analysis['oversized_topics'])}
"""
                st.download_button(
                    label="📥 Download Report (TXT)",
                    data=summary,
                    file_name=f"report_{st.session_state.uploaded_file_name}.txt",
                    mime="text/plain",
                    help="Summary report"
                )
            
    elif not uploaded_file:
        # Welcome screen
        st.info("👆 Please upload a CSV file in the sidebar to begin.")
        
        # Display tips
        st.header("🚀 Smart Category Balancing Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Outlier Reduction")
            st.markdown("""
            **Three strategies to minimize outliers:**
            
            1. **Aggressive (K-means)** - 0% outliers
               - Forces ALL documents into topics
               - Best for balanced categorization
               
            2. **Moderate (Lenient HDBSCAN)** - <15% outliers
               - More flexible clustering
               - Balances quality and coverage
               
            3. **Conservative (Standard HDBSCAN)** - Natural clustering
               - Highest quality topics
               - May have more outliers
            """)
        
        with col2:
            st.subheader("📊 Smart Category Detection")
            st.markdown("""
            **Automatic detection of problems:**
            
            - **Oversized Categories**: Warns when any topic has >30% of docs
            - **Split Suggestions**: Recommends how many subtopics to create
            - **Balance Monitoring**: Tracks distribution across all topics
            - **Visual Warnings**: Red flags for categories needing attention
            
            **Split Large Topics Tool:**
            - Analyze oversized categories
            - Break them into meaningful subtopics
            - Improve overall balance
            """)
        
        # Installation tips
        st.header("📦 Setup & Installation")
        
        st.code("""
# Required packages
pip install streamlit bertopic sentence-transformers torch pandas plotly

# Optional: GPU acceleration (highly recommended!)
pip install accelerate  # For better GPU performance
pip install faiss-gpu   # For faster similarity search

# Optional: CUDA optimization
pip install cupy-cuda11x  # For GPU arrays
        """, language="bash")
        
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
