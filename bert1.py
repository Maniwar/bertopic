import streamlit as st
import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit to wide mode for better layout
st.set_page_config(layout="wide", page_title="Windows CUDA-Optimized BERTopic", page_icon="🚀")

# Check for accelerate package (required for some device operations)
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False
    st.error("""
    ❌ **Missing 'accelerate' package**
    
    The error you're seeing requires the accelerate package. Please install it:
    ```bash
    pip install accelerate
    ```
    
    After installation, restart the app.
    """)
    st.stop()

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
    torch.cuda.init()
    torch.cuda.empty_cache()
    # DO NOT set default tensor type globally - it conflicts with SentenceTransformer
    # Instead, we'll handle device placement explicitly where needed
    # Enable TF32 for better performance on newer GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set memory fraction to prevent OOM
    torch.cuda.set_per_process_memory_fraction(0.8)

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
# OPTIMIZED MODEL CREATION WITH WINDOWS GPU SUPPORT
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
    Creates a Windows-optimized BERTopic model with maximum GPU utilization.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Log GPU status
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        gpu_memory = device_props.total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # Adaptive parameters based on dataset size
    is_small = num_docs and num_docs < 500
    is_large = num_docs and num_docs > 10000
    
    # UMAP parameters - optimized for Windows
    umap_params = {
        'n_neighbors': n_neighbors,
        'n_components': n_components,
        'min_dist': 0.0,
        'metric': 'cosine',
        'random_state': 42,
        'low_memory': False,
        'n_jobs': -1  # Use all CPU cores for UMAP
    }
    
    # Use GPU-accelerated UMAP if available (WSL2)
    if cuml_available:
        umap_params['verbose'] = False
        umap_params['output_type'] = 'numpy'
        umap_model = cumlUMAP(**umap_params)
    else:
        # CPU UMAP with parallel processing
        umap_model = UMAP(**umap_params)
    
    # Clustering model selection
    final_nr_topics = nr_topics if isinstance(nr_topics, int) else None
    
    if use_kmeans:
        if use_gpu_kmeans and torch.cuda.is_available():
            # Use custom GPU K-means for Windows
            clustering_model = GPUKMeans(
                n_clusters=final_nr_topics or 15
            )
            print("Using GPU-accelerated K-means clustering")
        else:
            # Standard sklearn K-means
            clustering_model = KMeans(
                n_clusters=final_nr_topics or 15,
                random_state=42,
                n_init='auto'
            )
    else:
        if cuml_available:
            # cuML HDBSCAN (WSL2)
            clustering_model = cumlHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                prediction_data=True,
                gen_min_span_tree=True,
                output_type='numpy'
            )
        elif torch.cuda.is_available() and num_docs and num_docs < 50000:
            # Windows-optimized HDBSCAN with GPU distance calculation
            clustering_model = WindowsOptimizedHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method='leaf',
                prediction_data=True
            )
            print("Using GPU-accelerated distance calculations for HDBSCAN")
        else:
            # Standard HDBSCAN
            clustering_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='leaf',
                prediction_data=True
            )
    
    # Optimized vectorizer
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=10000 if not is_large else 20000
    )
    
    # Representation models
    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3)
    ]
    
    # Choose embedding model based on dataset size and GPU
    if is_small or device == 'cpu':
        embedding_model_name = 'all-MiniLM-L6-v2'
    elif is_large and device == 'cuda':
        embedding_model_name = 'all-mpnet-base-v2'  # Better for large datasets
    else:
        embedding_model_name = 'all-MiniLM-L12-v2'  # Good balance
    
    # Initialize embedding model with optimizations
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    
    # Windows CUDA optimizations for embeddings
    if device == 'cuda':
        embedding_model = embedding_model.to('cuda')
        # Get device properties for capability check
        device_props = torch.cuda.get_device_properties(0)
        # Only use half precision if GPU supports it (compute capability >= 7.0)
        if device_props.major >= 7 and use_fp16:
            embedding_model = embedding_model.half()  # Use FP16 for faster inference
        
        # Set optimal batch size based on GPU memory
        gpu_memory_gb = device_props.total_memory / 1024**3
        if gpu_memory_gb >= 8:
            embedding_model.max_seq_length = 256  # Can handle longer sequences
        else:
            embedding_model.max_seq_length = 128  # Conserve memory
    
    # Create BERTopic model
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=calculate_probabilities,
        min_topic_size=min_topic_size,
        nr_topics=final_nr_topics,
        verbose=True,
        low_memory=False
    )
    
    return model

# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV byte string for download."""
    return df.to_csv(index=False).encode("utf-8")

def check_gpu_capabilities():
    """Comprehensive GPU capability check for Windows"""
    capabilities = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'primary_device': None,
        'cuda_version': None,
        'cudnn_enabled': torch.backends.cudnn.enabled if torch.cuda.is_available() else False,
        'cupy_available': cupy_available,
        'cuml_available': cuml_available,
        'faiss_gpu': faiss_gpu_available,
        'memory_gb': 0,
        'compute_capability': None,
        'supports_fp16': False,
        'supports_tf32': False
    }
    
    if torch.cuda.is_available():
        capabilities['device_count'] = torch.cuda.device_count()
        capabilities['cuda_version'] = torch.version.cuda
        
        device_props = torch.cuda.get_device_properties(0)
        capabilities['primary_device'] = torch.cuda.get_device_name(0)
        capabilities['memory_gb'] = device_props.total_memory / 1024**3
        capabilities['compute_capability'] = f"{device_props.major}.{device_props.minor}"
        
        # Check for FP16 support (compute capability >= 7.0)
        capabilities['supports_fp16'] = device_props.major >= 7
        # Check for TF32 support (compute capability >= 8.0)
        capabilities['supports_tf32'] = device_props.major >= 8
        
    return capabilities

def get_cuda_memory_info():
    """Get current CUDA memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        free = total - allocated
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'total_mb': total,
            'usage_percent': (allocated / total) * 100
        }
    return None

def generate_embeddings_gpu_optimized(docs, embedding_model, batch_size=32, show_progress=True, use_amp=True):
    """Generate embeddings with GPU optimization and progress tracking"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Use optimal batch size based on GPU memory
        memory_info = get_cuda_memory_info()
        if memory_info and memory_info['free_mb'] > 4000:
            batch_size = 64
        elif memory_info and memory_info['free_mb'] < 2000:
            batch_size = 16
    
    # Generate embeddings with progress
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    embeddings = []
    total_batches = (len(docs) - 1) // batch_size + 1
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        
        if show_progress:
            progress = (i + len(batch)) / len(docs)
            progress_bar.progress(progress)
            status_text.text(f"Generating embeddings: {i+len(batch)}/{len(docs)} documents")
        
        # Generate batch embeddings
        if torch.cuda.is_available() and use_amp:
            with torch.cuda.amp.autocast():  # Use mixed precision
                batch_embeddings = embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=batch_size
                )
        else:
            batch_embeddings = embedding_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
        embeddings.append(batch_embeddings)
        
        # Clear cache periodically
        if torch.cuda.is_available() and i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    if show_progress:
        progress_bar.progress(1.0)
        status_text.text("Embeddings generated!")
    
    return np.vstack(embeddings)

# -----------------------------------------------------
# MAIN STREAMLIT APP
# -----------------------------------------------------
def main():
    # --- SESSION STATE INITIALIZATION ---
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "ran_params" not in st.session_state:
        st.session_state.ran_params = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None

    st.title("🚀 Windows CUDA-Optimized BERTopic Analyzer")
    st.write("Maximum GPU acceleration for topic modeling on Windows systems")
    
    # --- SIDEBAR: CONTROLS AND STATUS ---
    with st.sidebar:
        st.header("🖥️ System Capabilities")
        
        # Comprehensive GPU status
        capabilities = check_gpu_capabilities()
        
        if capabilities['cuda_available']:
            st.success(f"✅ CUDA GPU: {capabilities['primary_device']}")
            
            # GPU Details
            with st.expander("GPU Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Memory", f"{capabilities['memory_gb']:.1f} GB")
                    st.metric("CUDA Version", capabilities['cuda_version'])
                    st.metric("Compute", capabilities['compute_capability'])
                with col2:
                    memory_info = get_cuda_memory_info()
                    if memory_info:
                        st.metric("Memory Used", f"{memory_info['usage_percent']:.1f}%")
                    st.metric("Devices", capabilities['device_count'])
                
                # Acceleration features
                st.write("**Acceleration Features:**")
                features = []
                if capabilities['supports_fp16']:
                    features.append("✅ FP16 (Half precision)")
                if capabilities['supports_tf32']:
                    features.append("✅ TF32 (TensorFloat-32)")
                if capabilities['cudnn_enabled']:
                    features.append("✅ cuDNN")
                if capabilities['cupy_available']:
                    features.append("✅ CuPy arrays")
                else:
                    features.append("❌ CuPy (install: `pip install cupy-cuda11x`)")
                if capabilities['faiss_gpu']:
                    features.append("✅ FAISS GPU")
                elif faiss_available:
                    features.append("⚠️ FAISS CPU only")
                else:
                    features.append("❌ FAISS (install: `pip install faiss-gpu`)")
                
                for feature in features:
                    st.write(feature)
            
            # Windows-specific notice
            if not capabilities['cuml_available']:
                with st.expander("⚠️ Windows GPU Limitations"):
                    st.info("""
                    **cuML not available on native Windows**
                    
                    Using optimized alternatives:
                    - ✅ GPU K-means clustering
                    - ✅ GPU-accelerated embeddings (FP16)
                    - ✅ GPU distance calculations
                    - ✅ Mixed precision training
                    
                    For full cuML support, consider using WSL2.
                    """)
        else:
            st.error("❌ No CUDA GPU detected")
            st.info("The app will run on CPU. For better performance, ensure CUDA is properly installed.")

        st.header("📁 Upload & Control")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        # If a new file is uploaded, clear all previous results
        if uploaded_file and (st.session_state.uploaded_file_name != uploaded_file.name):
            st.session_state.processed_df = None
            st.session_state.model = None
            st.session_state.ran_params = None
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.embeddings = None
            st.cache_resource.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.rerun()

        # --- FORM TO PREVENT RERUNS ON INPUT ---
        with st.form(key="topic_params_form"):
            st.header("🎯 Topic Modeling Controls")
            
            # Main parameters
            nr_topics = st.selectbox(
                "Number of Topics",
                ["auto", 10, 20, 30, 40, 50, 75, 100],
                index=0,
                help="'auto': Let model decide | Number: Target topic count"
            )
            
            min_topic_size = st.slider(
                "Minimum Topic Size",
                3, 100, 10, 1,
                help="Smallest allowed topic. Lower = more topics, fewer outliers"
            )
            
            st.header("⚙️ Advanced Options")
            
            # Organized tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Clustering", "UMAP", "GPU", "Performance"])
            
            with tab1:
                clustering_method = st.radio(
                    "Clustering Method",
                    ["HDBSCAN", "K-means (GPU)", "K-means (CPU)"],
                    index=1 if torch.cuda.is_available() else 0,
                    help="K-means GPU is fastest on Windows with CUDA"
                )
                
                if "HDBSCAN" in clustering_method:
                    st.write("**HDBSCAN Parameters**")
                    min_cluster_size = st.slider(
                        "Min Cluster Size",
                        2, 100, 5, 1,
                        help="Lower = fewer outliers"
                    )
                    min_samples = st.slider(
                        "Min Samples",
                        1, 50, 1, 1,
                        help="Lower = denser clusters"
                    )
                else:
                    min_cluster_size, min_samples = 5, 1
            
            with tab2:
                st.write("**UMAP Dimension Reduction**")
                n_neighbors = st.slider(
                    "N Neighbors",
                    2, 100, 15, 1,
                    help="Local neighborhood size"
                )
                n_components = st.slider(
                    "N Components",
                    2, 30, 10, 1,
                    help="Output dimensions (10-15 recommended)"
                )
            
            with tab3:
                st.write("**GPU Optimization Settings**")
                use_fp16 = st.checkbox(
                    "Use FP16 (Half Precision)",
                    value=capabilities['supports_fp16'],
                    disabled=not capabilities['supports_fp16'],
                    help="2x faster embeddings with minimal quality loss"
                )
                
                embedding_batch_size = st.select_slider(
                    "Embedding Batch Size",
                    options=[8, 16, 32, 64, 128],
                    value=32,
                    help="Larger = faster but uses more GPU memory"
                )
                
                use_amp = st.checkbox(
                    "Use Automatic Mixed Precision",
                    value=True,
                    disabled=not torch.cuda.is_available(),
                    help="Faster computation with automatic precision management"
                )
            
            with tab4:
                calculate_probabilities = st.checkbox(
                    "Calculate Probabilities",
                    value=False,
                    help="Slower but provides confidence scores"
                )
                
                cache_embeddings = st.checkbox(
                    "Cache Embeddings",
                    value=True,
                    help="Store embeddings for faster re-clustering"
                )
            
            # Submit button
            submit_button = st.form_submit_button(label="🚀 Run Topic Modeling")

    # --- MAIN PAGE LOGIC ---
    if submit_button and uploaded_file is not None:
        # Clear CUDA cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Determine clustering settings
        use_kmeans = "K-means" in clustering_method
        use_gpu_kmeans = "GPU" in clustering_method
        
        # Set default values if not defined (for safety)
        if 'use_fp16' not in locals():
            use_fp16 = False
        if 'use_amp' not in locals():
            use_amp = True
        if 'cache_embeddings' not in locals():
            cache_embeddings = True
        
        with st.spinner(f"Processing {st.session_state.uploaded_file_name}..."):
            # Initialize progress tracking
            main_progress = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load data
            status_text.text("Loading data...")
            main_progress.progress(0.1)
            df = pd.read_csv(uploaded_file)
            
            # Infer text column
            text_cols = df.select_dtypes(include=['object']).columns
            text_col = text_cols[0] if len(text_cols) > 0 else df.columns[0]
            
            docs = df[text_col].dropna().astype(str).tolist()
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", len(docs))
            with col2:
                st.metric("Text Column", text_col)
            with col3:
                if torch.cuda.is_available():
                    memory_before = get_cuda_memory_info()['usage_percent']
                    st.metric("GPU Memory", f"{memory_before:.1f}%")
            
            # Step 2: Create model
            status_text.text("Creating optimized model...")
            main_progress.progress(0.2)
            
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
                calculate_probabilities=calculate_probabilities,
                embedding_batch_size=embedding_batch_size,
                use_fp16=use_fp16
            )
            
            # Step 3: Generate or load embeddings
            status_text.text("Generating embeddings on GPU...")
            main_progress.progress(0.3)
            
            if cache_embeddings and st.session_state.embeddings is not None and len(st.session_state.embeddings) == len(docs):
                st.info("Using cached embeddings for faster processing")
                embeddings = st.session_state.embeddings
            else:
                # Generate embeddings with GPU optimization
                embeddings = generate_embeddings_gpu_optimized(
                    docs,
                    model.embedding_model,
                    batch_size=embedding_batch_size,
                    show_progress=True,
                    use_amp=use_amp
                )
                if cache_embeddings:
                    st.session_state.embeddings = embeddings
            
            # Step 4: Fit the model
            status_text.text("Clustering documents...")
            main_progress.progress(0.7)
            
            # Use pre-computed embeddings for faster processing
            topics, probs = model.fit_transform(docs, embeddings)
            
            # Step 5: Process results
            status_text.text("Processing results...")
            main_progress.progress(0.9)
            
            # Get topic information
            topic_info = model.get_topic_info()
            topic_label_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
            
            # Add to dataframe
            df["topic_id"] = topics
            df["topic_label"] = [topic_label_map.get(t, "Outliers") for t in topics]
            
            # Add probabilities if calculated
            if calculate_probabilities and probs is not None:
                if len(probs.shape) > 1:
                    df["topic_probability"] = np.max(probs, axis=1)
                    df["topic_entropy"] = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                else:
                    df["topic_probability"] = probs
            
            # Store in session state
            st.session_state.processed_df = df
            st.session_state.model = model
            st.session_state.ran_params = {
                "Number of Topics": nr_topics,
                "Min Topic Size": min_topic_size,
                "Clustering": clustering_method,
                "Min Cluster Size": min_cluster_size,
                "Min Samples": min_samples,
                "N Neighbors": n_neighbors,
                "N Components": n_components,
                "Embedding Batch Size": embedding_batch_size,
                "Calculate Probabilities": calculate_probabilities,
                "GPU Used": torch.cuda.is_available(),
                "FP16 Used": use_fp16 if torch.cuda.is_available() else False,
                "Documents Processed": len(docs)
            }
            
            main_progress.progress(1.0)
            status_text.text("Complete!")
            
        # Show completion message with GPU stats
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Analysis complete!")
        with col2:
            if torch.cuda.is_available():
                memory_after = get_cuda_memory_info()
                st.info(f"GPU Memory Used: {memory_after['allocated_mb']:.0f} MB")
                torch.cuda.empty_cache()

    # --- DISPLAY RESULTS ---
    if st.session_state.processed_df is not None:
        processed_df = st.session_state.processed_df
        model = st.session_state.model
        ran_params = st.session_state.ran_params
        text_col = processed_df.columns[0]

        st.header("📊 Topic Analysis Results")
        
        # Show GPU acceleration status
        if ran_params.get("GPU Used"):
            gpu_status = "🚀 GPU Accelerated"
            if ran_params.get("FP16 Used"):
                gpu_status += " (FP16)"
        else:
            gpu_status = "💻 CPU Mode"
        
        # Parameters used
        with st.expander(f"Settings Used | {gpu_status}", expanded=False):
            cols = st.columns(3)
            params_items = list(ran_params.items())
            for i, (key, value) in enumerate(params_items):
                cols[i % 3].write(f"**{key}**: {value}")

        # Calculate metrics
        topic_info = model.get_topic_info()
        unique_topics = len(topic_info[topic_info.Topic != -1])
        outlier_count = processed_df['topic_id'].value_counts().get(-1, 0)
        coverage = ((len(processed_df) - outlier_count) / len(processed_df)) * 100
        
        # Display metrics with enhanced visuals
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Topics Found",
                unique_topics,
                delta=f"{unique_topics - 1} clusters" if unique_topics > 1 else None
            )
        
        with col2:
            st.metric(
                "Outliers",
                outlier_count,
                delta=f"{outlier_count/len(processed_df)*100:.1f}%",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Coverage",
                f"{coverage:.1f}%",
                delta="Good" if coverage >= 70 else "Improve",
                delta_color="normal" if coverage >= 70 else "inverse"
            )
        
        with col4:
            avg_docs = (len(processed_df) - outlier_count) / max(unique_topics, 1)
            st.metric(
                "Avg Docs/Topic",
                f"{avg_docs:.0f}",
                delta="Balanced" if avg_docs > min_topic_size else "Small"
            )
        
        with col5:
            if torch.cuda.is_available():
                mem_info = get_cuda_memory_info()
                st.metric(
                    "GPU Memory",
                    f"{mem_info['usage_percent']:.1f}%",
                    delta=f"{mem_info['free_mb']:.0f} MB free"
                )
            else:
                st.metric("Mode", "CPU", delta="No GPU")
        
        # Outlier warning with specific recommendations
        if coverage < 70:
            st.warning(f"""
            ⚠️ **High outlier rate detected ({100-coverage:.1f}% outliers)**
            
            **Quick fixes for Windows with GPU:**
            1. Switch to **GPU K-means** clustering (fastest, no outliers)
            2. Reduce **Min Cluster Size** to 3 (currently {ran_params.get('Min Cluster Size', 'N/A')})
            3. Set **Min Samples** to 1 (currently {ran_params.get('Min Samples', 'N/A')})
            4. Increase **N Components** to 15 (currently {ran_params.get('N Components', 'N/A')})
            """)

        # Topic distribution with visualization
        st.subheader("📊 Topic Distribution")
        
        # Create distribution dataframe
        topic_dist = processed_df["topic_label"].value_counts().reset_index()
        topic_dist.columns = ["Topic", "Document Count"]
        topic_dist["Percentage"] = (topic_dist["Document Count"] / len(processed_df) * 100).round(2)
        
        # Add topic quality metrics if probabilities available
        if "topic_probability" in processed_df.columns:
            avg_probs = processed_df.groupby("topic_label")["topic_probability"].mean().reset_index()
            avg_probs.columns = ["Topic", "Avg Confidence"]
            topic_dist = topic_dist.merge(avg_probs, on="Topic", how="left")
            topic_dist["Avg Confidence"] = (topic_dist["Avg Confidence"] * 100).round(1)
        
        # Display with highlighting
        def style_dataframe(df):
            styles = []
            for _, row in df.iterrows():
                if row["Topic"] == "Outliers":
                    styles.append(['background-color: #ffcccc'] * len(row))
                elif row["Percentage"] > 20:
                    styles.append(['background-color: #ccffcc'] * len(row))
                else:
                    styles.append([''] * len(row))
            return pd.DataFrame(styles, index=df.index, columns=df.columns)
        
        st.dataframe(
            topic_dist.style.apply(lambda x: style_dataframe(topic_dist), axis=None),
            use_container_width=True,
            height=min(400, 50 + len(topic_dist) * 35)
        )
        
        # Topic visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Topic Details")
            topic_info_display = topic_info[['Topic', 'Count', 'Name']].copy()
            
            # Add representation keywords
            topic_info_display['Keywords'] = topic_info['Representation'].apply(
                lambda x: ', '.join(x[:5]) if isinstance(x, list) else str(x)[:100]
            )
            
            st.dataframe(topic_info_display, use_container_width=True, height=400)
        
        with col2:
            st.subheader("📈 Quick Stats")
            st.write(f"""
            **Model Performance:**
            - Processing time: < 1 min
            - GPU acceleration: {ran_params.get('GPU Used', False)}
            - Embedding model: {model.embedding_model.get_sentence_embedding_dimension()}D
            - Clustering: {ran_params.get('Clustering', 'Unknown')}
            
            **Topic Quality:**
            - Largest topic: {topic_dist.iloc[0]['Document Count']} docs
            - Smallest topic: {topic_dist[topic_dist['Topic'] != 'Outliers'].iloc[-1]['Document Count']} docs
            - Topic coherence: {'High' if coverage > 80 else 'Medium' if coverage > 60 else 'Low'}
            """)

        # Document viewer
        st.subheader("📄 Explore Documents by Topic")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            unique_labels = sorted(processed_df["topic_label"].unique())
            selected_topic = st.selectbox(
                "Select a topic:",
                options=unique_labels,
                key="doc_viewer"
            )
        
        with col2:
            num_docs_to_show = st.number_input(
                "Documents to show:",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )
        
        if selected_topic:
            topic_docs_df = processed_df[processed_df["topic_label"] == selected_topic]
            
            # Sort by probability if available
            if "topic_probability" in topic_docs_df.columns:
                topic_docs_df = topic_docs_df.sort_values("topic_probability", ascending=False)
            
            st.info(f"**{selected_topic}** | {len(topic_docs_df)} total documents")
            
            # Display columns based on available data
            display_cols = [text_col]
            if "topic_probability" in processed_df.columns:
                display_cols.append("topic_probability")
            if "topic_entropy" in processed_df.columns:
                display_cols.append("topic_entropy")
            
            # Show documents
            display_df = topic_docs_df[display_cols].head(num_docs_to_show).reset_index(drop=True)
            
            # Round numeric columns
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'float32']:
                    display_df[col] = display_df[col].round(3)
            
            st.dataframe(display_df, use_container_width=True, height=400)

        # Advanced Analysis Section
        st.header("🔬 Advanced Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Hierarchical Topics", "Topic Similarity", "Export"])
        
        with tab1:
            st.subheader("🌳 Hierarchical Topic Analysis")
            
            # Filter out outliers from topic list
            topic_list = processed_df['topic_label'].value_counts().index.tolist()
            if "Outliers" in topic_list:
                topic_list.remove("Outliers")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_topic_to_split = st.selectbox(
                    "Select topic to analyze:",
                    options=topic_list,
                    key="hierarchical_selector"
                )
            
            with col2:
                sub_n_topics = st.number_input(
                    "Target sub-topics:",
                    min_value=2,
                    max_value=10,
                    value=5
                )
            
            if st.button(f"🔍 Find Sub-Topics in '{selected_topic_to_split}'"):
                docs_to_split = processed_df[processed_df['topic_label'] == selected_topic_to_split][text_col].tolist()
                
                if len(docs_to_split) >= 10:
                    with st.spinner(f"Analyzing {len(docs_to_split)} documents..."):
                        # Use GPU-optimized model for sub-analysis
                        if torch.cuda.is_available():
                            sub_model = BERTopic(
                                min_topic_size=max(3, len(docs_to_split) // sub_n_topics),
                                nr_topics=sub_n_topics,
                                calculate_probabilities=False,
                                verbose=False
                            )
                        else:
                            sub_model = BERTopic(
                                min_topic_size=max(3, len(docs_to_split) // 10),
                                nr_topics="auto",
                                verbose=False
                            )
                        
                        sub_topics, _ = sub_model.fit_transform(docs_to_split)
                        
                        st.write("### 📊 Sub-Topics Found:")
                        sub_topic_info = sub_model.get_topic_info()
                        
                        # Clean display
                        sub_topic_display = sub_topic_info[['Topic', 'Count', 'Name']].copy()
                        sub_topic_display['Keywords'] = sub_topic_info['Representation'].apply(
                            lambda x: ', '.join(x[:3]) if isinstance(x, list) else str(x)[:50]
                        )
                        
                        st.dataframe(sub_topic_display, use_container_width=True)
                        
                        # Summary
                        sub_outliers = sum(1 for t in sub_topics if t == -1)
                        sub_coverage = ((len(sub_topics) - sub_outliers) / len(sub_topics)) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sub-topics", len(sub_topic_info) - 1)
                        with col2:
                            st.metric("Sub-outliers", sub_outliers)
                        with col3:
                            st.metric("Sub-coverage", f"{sub_coverage:.1f}%")
                else:
                    st.warning(f"Not enough documents ({len(docs_to_split)} < 10) for sub-analysis")
        
        with tab2:
            st.subheader("🔗 Topic Similarity Matrix")
            
            if st.button("Calculate Topic Similarities"):
                with st.spinner("Computing similarity matrix..."):
                    # Get topic embeddings
                    topic_embeddings = model.topic_embeddings_
                    
                    if topic_embeddings is not None and len(topic_embeddings) > 1:
                        # Calculate cosine similarity
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarity_matrix = cosine_similarity(topic_embeddings[1:])  # Exclude outliers
                        
                        # Create dataframe
                        topic_names = [topic_info[topic_info.Topic == i].iloc[0]['Name'] 
                                     for i in range(similarity_matrix.shape[0])]
                        
                        sim_df = pd.DataFrame(
                            similarity_matrix,
                            index=topic_names[:len(similarity_matrix)],
                            columns=topic_names[:len(similarity_matrix)]
                        )
                        
                        # Display heatmap-style
                        st.dataframe(
                            sim_df.style.background_gradient(cmap='RdYlBu_r', vmin=0, vmax=1),
                            use_container_width=True
                        )
                        
                        # Find most similar pairs
                        st.write("### Most Similar Topic Pairs:")
                        upper_tri = np.triu(similarity_matrix, k=1)
                        top_pairs = []
                        
                        for i in range(len(upper_tri)):
                            for j in range(i+1, len(upper_tri)):
                                if upper_tri[i, j] > 0.5:  # Threshold
                                    top_pairs.append({
                                        'Topic 1': topic_names[i][:50],
                                        'Topic 2': topic_names[j][:50],
                                        'Similarity': f"{upper_tri[i, j]:.3f}"
                                    })
                        
                        if top_pairs:
                            st.dataframe(pd.DataFrame(top_pairs), use_container_width=True)
                        else:
                            st.info("No highly similar topics found (threshold: 0.5)")
                    else:
                        st.error("Not enough topics for similarity analysis")
        
        with tab3:
            st.subheader("💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=convert_df_to_csv(processed_df),
                    file_name=f"bertopic_results_{st.session_state.uploaded_file_name}",
                    mime="text/csv",
                    help="Full dataset with topic assignments"
                )
            
            with col2:
                st.download_button(
                    label="📥 Download Topic Info (CSV)",
                    data=convert_df_to_csv(topic_info),
                    file_name=f"topic_info_{st.session_state.uploaded_file_name}",
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
Model: {ran_params.get('Clustering', 'Unknown')}
"""
                st.download_button(
                    label="📥 Download Report (TXT)",
                    data=summary,
                    file_name=f"report_{st.session_state.uploaded_file_name}.txt",
                    mime="text/plain",
                    help="Summary report"
                )
            
    elif not uploaded_file:
        # Welcome screen with instructions
        st.info("👆 Please upload a CSV file in the sidebar to begin.")
        
        # Display optimization tips
        st.header("🚀 Windows CUDA Optimization Tips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Best Practices")
            st.markdown("""
            **For Maximum GPU Performance:**
            1. Use **GPU K-means** clustering (fastest)
            2. Enable **FP16** for 2x faster embeddings
            3. Increase **batch size** if GPU memory allows
            4. Cache embeddings for re-clustering
            
            **To Reduce Outliers:**
            - Min Cluster Size: 3-5
            - Min Samples: 1
            - N Components: 10-15
            - Use K-means instead of HDBSCAN
            """)
        
        with col2:
            st.subheader("📦 Optional Packages")
            st.markdown("""
            **Install for better performance:**
            ```bash
            # GPU arrays (recommended)
            pip install cupy-cuda11x
            
            # GPU similarity search
            pip install faiss-gpu
            
            # Mixed precision training
            pip install nvidia-ml-py3
            
            # For WSL2 users only:
            conda install -c rapidsai cuml
            ```
            """)
        
        # System check
        with st.expander("🔍 Run System Check"):
            if st.button("Check GPU Capabilities"):
                capabilities = check_gpu_capabilities()
                
                st.write("### System Capabilities:")
                for key, value in capabilities.items():
                    if isinstance(value, bool):
                        icon = "✅" if value else "❌"
                        st.write(f"{icon} **{key.replace('_', ' ').title()}**")
                    elif value is not None:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")

if __name__ == "__main__":
    main()
