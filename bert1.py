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
# TOPIC MERGER FOR MINIMUM SIZE ENFORCEMENT
# -----------------------------------------------------
class TopicMerger:
    """Merge small topics to enforce minimum topic size"""
    
    @staticmethod
    def merge_small_topics(topics, embeddings, min_size=10):
        """
        Merge topics that are smaller than min_size with their nearest neighbors
        
        Args:
            topics: Array of topic assignments
            embeddings: Document embeddings
            min_size: Minimum number of documents per topic
            
        Returns:
            Merged topic assignments
        """
        from collections import Counter
        from scipy.spatial.distance import cosine
        
        topics = np.array(topics)
        topic_counts = Counter(topics)
        
        # Identify small topics (excluding outliers -1)
        small_topics = [t for t, count in topic_counts.items() 
                       if t != -1 and count < min_size]
        
        if not small_topics:
            return topics
        
        # Calculate topic centroids
        topic_centroids = {}
        for topic in set(topics):
            if topic != -1:
                topic_mask = topics == topic
                topic_centroids[topic] = embeddings[topic_mask].mean(axis=0)
        
        # Merge small topics with nearest larger topic
        merged_topics = topics.copy()
        
        for small_topic in small_topics:
            if small_topic not in topic_centroids:
                continue
                
            small_centroid = topic_centroids[small_topic]
            
            # Find nearest larger topic
            min_distance = float('inf')
            best_merge_topic = None
            
            for topic, centroid in topic_centroids.items():
                if topic != small_topic and topic_counts[topic] >= min_size:
                    distance = cosine(small_centroid, centroid)
                    if distance < min_distance:
                        min_distance = distance
                        best_merge_topic = topic
            
            # Merge if found suitable topic
            if best_merge_topic is not None:
                merged_topics[merged_topics == small_topic] = best_merge_topic
                topic_counts[best_merge_topic] += topic_counts[small_topic]
                del topic_counts[small_topic]
        
        # Renumber topics to be sequential
        unique_topics = sorted(set(merged_topics[merged_topics != -1]))
        topic_mapping = {old: new for new, old in enumerate(unique_topics)}
        topic_mapping[-1] = -1  # Keep outliers as -1
        
        final_topics = np.array([topic_mapping.get(t, t) for t in merged_topics])
        
        return final_topics

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
    def suggest_parameters(num_docs, current_outlier_ratio, min_topic_size=10):
        """
        Suggest better parameters to reduce outliers
        
        Args:
            num_docs: Number of documents
            current_outlier_ratio: Current outlier percentage (0-1)
            min_topic_size: User-defined minimum topic size
        
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
            # Adjust number of topics based on min_topic_size
            max_topics = max(5, num_docs // min_topic_size)
            suggested_topics = min(max_topics, max(5, num_docs // 50))
            
            suggestions['parameters'] = {
                'use_kmeans': True,
                'use_gpu_kmeans': True,
                'nr_topics': suggested_topics,
                'min_topic_size': min_topic_size,
                'merge_small_topics': True  # New parameter
            }
            suggestions['explanation'] = f"High outliers detected. K-means will assign all documents to topics. Topics smaller than {min_topic_size} will be merged."
            
        elif current_outlier_ratio > 0.15:
            # Moderate outliers - adjust HDBSCAN
            suggestions['method'] = 'hdbscan'
            suggestions['parameters'] = {
                'use_kmeans': False,
                'min_cluster_size': max(3, min_topic_size // 3),  # Dynamic based on min_topic_size
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

# GPU-accelerated K-means wrapper
class GPUKMeans:
    """K-means clustering with GPU acceleration if available"""
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Try to use GPU-accelerated version
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
# MAIN APPLICATION
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
        
        # Get memory info
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
    if torch.cuda.is_available():
        model = SentenceTransformer(model_name, device='cuda')
        # Optimize for GPU
        model.max_seq_length = 512
        model.encode = torch.compile(model.encode) if hasattr(torch, 'compile') else model.encode
    else:
        model = SentenceTransformer(model_name, device='cpu')
    return model

def main():
    st.title("🚀 Windows CUDA-Optimized BERTopic with Smart Balancing")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    
    # Check GPU capabilities
    gpu_capabilities = check_gpu_capabilities()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # GPU Status Display
        if gpu_capabilities['cuda_available']:
            st.success(f"✅ GPU Detected: {gpu_capabilities['device_name']}")
            st.info(f"Memory: {gpu_capabilities['gpu_memory_free']} free / {gpu_capabilities['gpu_memory_total']} total")
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
            # Store filename in session state
            st.session_state.uploaded_file_name = uploaded_file.name.replace('.csv', '')
            
            # Load and preview data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df):,} rows")
            
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
            
            # NEW: Minimum Topic Size Control
            st.subheader("📏 Topic Size Control")
            min_topic_size = st.slider(
                "Minimum Topic Size",
                min_value=2,
                max_value=min(100, len(df) // 10),
                value=min(10, len(df) // 50),
                help="Minimum number of documents per topic. Topics smaller than this will be merged with similar topics."
            )
            
            st.info(f"ℹ️ Topics with fewer than {min_topic_size} documents will be merged with their most similar larger topic.")
            
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
                    # Calculate reasonable max topics based on min_topic_size
                    max_topics = max(5, len(df) // min_topic_size)
                    default_topics = min(max_topics, max(5, len(df) // 50))
                    
                    nr_topics = st.number_input(
                        "Number of Topics",
                        min_value=2,
                        max_value=max_topics,
                        value=default_topics,
                        help=f"Number of topics to create (limited by min topic size of {min_topic_size})"
                    )
                    
                    # Validate that nr_topics * min_topic_size doesn't exceed document count
                    if nr_topics * min_topic_size > len(df):
                        st.warning(f"⚠️ With {nr_topics} topics and min size {min_topic_size}, you need at least {nr_topics * min_topic_size} documents. Adjusting...")
                        nr_topics = len(df) // min_topic_size
                else:
                    nr_topics = None
                
                # UMAP parameters
                n_neighbors = st.slider("UMAP n_neighbors", 2, 50, 15)
                n_components = st.slider("UMAP n_components", 2, 10, 5)
                
                # Representation model
                use_mmr = st.checkbox("Use MMR for diverse keywords", value=True)
                
                # Seed words
                st.subheader("🎯 Seed Words (Optional)")
                seed_words_input = st.text_area(
                    "Enter seed words (one set per line)",
                    placeholder="Example:\nfinance, money, budget, cost\nmarketing, campaign, advertising",
                    help="Guide topic discovery with predefined keyword sets"
                )
                
    # Main content area
    if uploaded_file is not None and 'text_col' in locals():
        # Process button
        if st.button("🚀 Run Topic Modeling", type="primary"):
            # Clear any existing results
            st.session_state.model = None
            st.session_state.processed_df = None
            
            # Set up parameters based on strategy and min_topic_size
            if "Aggressive" in outlier_strategy:
                use_kmeans = True
                if torch.cuda.is_available() and cuml_available:
                    hdbscan_model = GPUKMeans(n_clusters=nr_topics)
                else:
                    hdbscan_model = KMeans(n_clusters=nr_topics, random_state=42)
                clustering_method = "K-means"
            elif "Moderate" in outlier_strategy:
                use_kmeans = False
                min_cluster_size = max(3, min_topic_size // 3)
                if torch.cuda.is_available() and cuml_available:
                    hdbscan_model = cumlHDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=max(1, min_topic_size // 5),
                        prediction_data=True
                    )
                else:
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=max(1, min_topic_size // 5),
                        prediction_data=True
                    )
                clustering_method = "Lenient HDBSCAN"
            else:  # Conservative
                use_kmeans = False
                if torch.cuda.is_available() and cuml_available:
                    hdbscan_model = cumlHDBSCAN(
                        min_cluster_size=min_topic_size,
                        min_samples=max(1, min_topic_size // 2)
                    )
                else:
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=min_topic_size,
                        min_samples=max(1, min_topic_size // 2)
                    )
                clustering_method = "Standard HDBSCAN"
            
            # UMAP configuration
            if torch.cuda.is_available() and cuml_available:
                umap_model = cumlUMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    random_state=42
                )
            else:
                umap_model = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    random_state=42
                )
            
            # Representation model
            representation_model = []
            if use_mmr:
                representation_model.append(MaximalMarginalRelevance())
            representation_model.append(KeyBERTInspired())
            
            # Parse seed words if provided
            seed_topic_list = []
            if seed_words_input:
                for line in seed_words_input.strip().split('\n'):
                    if line.strip():
                        words = [w.strip() for w in line.split(',')]
                        if words:
                            seed_topic_list.append(words)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load embedding model (20%)
                status_text.text("📚 Loading embedding model...")
                progress_bar.progress(20)
                
                sentence_model = load_embedding_model(embedding_model)
                
                # Step 2: Generate embeddings (40%)
                status_text.text(f"🔤 Generating embeddings for {len(df):,} documents...")
                progress_bar.progress(40)
                
                documents = df[text_col].tolist()
                
                # GPU-optimized batch encoding
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        embeddings = sentence_model.encode(
                            documents,
                            show_progress_bar=True,
                            batch_size=32,
                            convert_to_numpy=True
                        )
                else:
                    embeddings = sentence_model.encode(
                        documents,
                        show_progress_bar=True,
                        batch_size=32,
                        convert_to_numpy=True
                    )
                
                # Store embeddings in session state for later use
                st.session_state.embeddings = embeddings
                
                # Step 3: Configure BERTopic (50%)
                status_text.text("🔧 Configuring BERTopic model...")
                progress_bar.progress(50)
                
                # Create BERTopic model with appropriate min_topic_size
                topic_model = BERTopic(
                    embedding_model=sentence_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    representation_model=representation_model,
                    min_topic_size=min_topic_size,  # Use user-defined min_topic_size
                    seed_topic_list=seed_topic_list if seed_topic_list else None,
                    calculate_probabilities=False if use_kmeans else True,
                    verbose=True
                )
                
                # Step 4: Fit model (70%)
                status_text.text(f"🎯 Fitting topics using {clustering_method}...")
                progress_bar.progress(70)
                
                topics, probs = topic_model.fit_transform(documents, embeddings)
                
                # Step 5: Post-processing - Merge small topics if needed
                if min_topic_size > 2:
                    status_text.text(f"🔄 Merging topics smaller than {min_topic_size} documents...")
                    progress_bar.progress(80)
                    
                    # Get original topic counts
                    from collections import Counter
                    original_counts = Counter(topics)
                    small_topics = [t for t, count in original_counts.items() 
                                  if t != -1 and count < min_topic_size]
                    
                    if small_topics:
                        # Merge small topics
                        merged_topics = TopicMerger.merge_small_topics(
                            topics, embeddings, min_topic_size
                        )
                        
                        # Update the model with merged topics
                        topics = merged_topics
                        
                        # Re-fit the topic representations for merged topics
                        topic_model._update_topic_size(topics)
                        topic_model._extract_topics(documents, embeddings)
                        
                        st.info(f"ℹ️ Merged {len(small_topics)} small topics into larger ones")
                
                # Step 6: Prepare results (90%)
                status_text.text("📊 Preparing results...")
                progress_bar.progress(90)
                
                # Get topic information
                topic_info = topic_model.get_topic_info()
                
                # Add results to dataframe
                df_results = df.copy()
                df_results['topic'] = topics
                df_results['topic_label'] = df_results['topic'].apply(
                    lambda x: f"Topic {x}" if x != -1 else "Outlier"
                )
                
                # Add keywords for each topic
                df_results['keywords'] = df_results['topic'].apply(
                    lambda x: ', '.join([word for word, _ in topic_model.get_topic(x)[:5]]) 
                    if x != -1 else "No keywords"
                )
                
                # Store in session state
                st.session_state.model = topic_model
                st.session_state.processed_df = df_results
                st.session_state.topic_info = topic_info
                st.session_state.clustering_method = clustering_method
                st.session_state.gpu_used = torch.cuda.is_available()
                st.session_state.min_topic_size_used = min_topic_size
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Topic modeling complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try reducing the number of topics or adjusting parameters")
        
        # Display results if available
        if st.session_state.model is not None:
            model = st.session_state.model
            processed_df = st.session_state.processed_df
            topic_info = st.session_state.topic_info
            
            # Summary metrics
            st.header("📊 Results Summary")
            
            # Calculate metrics
            unique_topics = len(processed_df['topic'].unique()) - 1  # Exclude outliers
            outlier_count = len(processed_df[processed_df['topic'] == -1])
            total_docs = len(processed_df)
            coverage = ((total_docs - outlier_count) / total_docs) * 100
            
            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Documents", f"{total_docs:,}")
            with col2:
                st.metric("Topics Found", unique_topics)
            with col3:
                st.metric("Outliers", f"{outlier_count:,} ({100-coverage:.1f}%)")
            with col4:
                st.metric("Coverage", f"{coverage:.1f}%")
            with col5:
                st.metric("Min Topic Size", st.session_state.min_topic_size_used)
            
            # Run parameters used
            ran_params = {
                'Clustering': st.session_state.clustering_method,
                'GPU Used': st.session_state.gpu_used,
                'Min Topic Size': st.session_state.min_topic_size_used
            }
            
            with st.expander("🔍 Parameters Used"):
                for key, value in ran_params.items():
                    st.write(f"**{key}:** {value}")
            
            # Balance Analysis
            balance_analyzer = CategoryBalanceAnalyzer()
            balance_analysis = balance_analyzer.analyze_distribution(
                processed_df['topic'].tolist(),
                include_outliers=False
            )
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Topics Overview", "📈 Distribution Analysis", 
                                              "🔍 Split Large Topics", "💾 Export"])
            
            with tab1:
                st.subheader("Topic Information")
                
                # Clean up topic display
                display_df = topic_info.copy()
                display_df = display_df[display_df['Topic'] != -1]  # Exclude outliers from main display
                
                # Add percentage column
                display_df['Percentage'] = (display_df['Count'] / total_docs * 100).round(2)
                
                # Format the representation column
                display_df['Keywords'] = display_df['Representation'].apply(
                    lambda x: ', '.join(x[:5]) if isinstance(x, list) else str(x)[:100]
                )
                
                # Select columns to display
                display_df = display_df[['Topic', 'Count', 'Percentage', 'Name', 'Keywords']]
                
                # Highlight oversized topics
                def highlight_oversized(row):
                    if row['Percentage'] > 30:
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['Percentage'] > 20:
                        return ['background-color: #fff4cc'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = display_df.style.apply(highlight_oversized, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Show outliers separately if any
                if outlier_count > 0:
                    st.warning(f"⚠️ {outlier_count} documents ({100-coverage:.1f}%) were classified as outliers")
                
                # Balance warnings
                if not balance_analysis['balanced']:
                    st.error("⚠️ **Topic Distribution Imbalance Detected!**")
                    for warning in balance_analysis['warnings']:
                        st.warning(warning)
                        
            with tab2:
                st.subheader("📈 Topic Distribution Analysis")
                
                # Create distribution chart
                import plotly.express as px
                
                # Prepare data for visualization
                viz_df = processed_df['topic'].value_counts().reset_index()
                viz_df.columns = ['Topic', 'Count']
                viz_df = viz_df[viz_df['Topic'] != -1]  # Exclude outliers
                viz_df['Topic'] = viz_df['Topic'].apply(lambda x: f"Topic {x}")
                viz_df = viz_df.sort_values('Count', ascending=False)
                
                # Bar chart
                fig = px.bar(viz_df, x='Topic', y='Count', 
                           title='Document Distribution Across Topics',
                           color='Count',
                           color_continuous_scale='Viridis')
                
                # Add threshold line
                threshold_count = total_docs * 0.3
                fig.add_hline(y=threshold_count, line_dash="dash", line_color="red",
                            annotation_text="30% threshold")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Balance score
                st.subheader("📊 Balance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    balance_score = balance_analysis['balance_score']
                    st.metric("Balance Score", f"{balance_score:.2f}", 
                            help="1.0 = perfectly balanced, 0 = highly imbalanced")
                with col2:
                    st.metric("Oversized Topics", len(balance_analysis['oversized_topics']))
                with col3:
                    avg_size = total_docs / unique_topics if unique_topics > 0 else 0
                    st.metric("Avg Topic Size", f"{avg_size:.0f} docs")
                    
            with tab3:
                st.subheader("🔍 Split Large Topics")
                
                if balance_analysis['oversized_topics']:
                    st.warning(f"Found {len(balance_analysis['oversized_topics'])} oversized topic(s)")
                    
                    # Select topic to split
                    oversized_options = [
                        f"Topic {t['topic']}: {t['count']} docs ({t['ratio']*100:.1f}%)"
                        for t in balance_analysis['oversized_topics']
                    ]
                    
                    selected_topic_to_split = st.selectbox(
                        "Select topic to split",
                        oversized_options
                    )
                    
                    # Extract topic number
                    topic_to_split = int(selected_topic_to_split.split(':')[0].replace('Topic ', ''))
                    
                    # Get suggested splits
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
                        docs_to_split = processed_df[
                            processed_df['topic'] == topic_to_split
                        ][text_col].tolist()
                        
                        min_docs_for_split = max(10, sub_n_topics * 2)
                        if len(docs_to_split) >= min_docs_for_split:
                            with st.spinner(f"Analyzing {len(docs_to_split):,} documents..."):
                                # Use K-means for splitting to ensure all docs are assigned
                                if torch.cuda.is_available():
                                    sub_model = BERTopic(
                                        hdbscan_model=GPUKMeans(n_clusters=sub_n_topics),
                                        min_topic_size=max(2, len(docs_to_split) // (sub_n_topics * 2)),
                                        calculate_probabilities=False,
                                        verbose=False
                                    )
                                else:
                                    sub_model = BERTopic(
                                        hdbscan_model=KMeans(n_clusters=sub_n_topics, random_state=42),
                                        min_topic_size=max(2, len(docs_to_split) // (sub_n_topics * 2)),
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
                            st.warning(f"Not enough documents ({len(docs_to_split)} < {min_docs_for_split}) for splitting")
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
Min Topic Size: {st.session_state.min_topic_size_used}
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
            st.subheader("✅ Improved Topic Size Control")
            st.markdown("""
            **NEW: Configurable Minimum Topic Size**
            
            - **Set minimum topic size** for ALL clustering modes
            - **Automatic merging** of small topics with similar larger ones
            - **Smart topic number limits** based on your minimum size
            - **Post-processing** ensures no tiny topics (1-2 documents)
            
            **Three Outlier Reduction Strategies:**
            
            1. **Aggressive (K-means)** - 0% outliers
               - Forces ALL documents into topics
               - Respects minimum topic size via merging
               
            2. **Moderate (Lenient HDBSCAN)** - <15% outliers
               - Flexible clustering with size constraints
               - Balances quality and coverage
               
            3. **Conservative (Standard HDBSCAN)** - Natural clustering
               - Highest quality topics
               - Enforces minimum size naturally
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
            - Maintains minimum size in subtopics
            """)
        
        # Installation tips
        st.header("📦 Setup & Installation")
        
        st.code("""
# Required packages
pip install streamlit bertopic sentence-transformers torch pandas plotly scipy

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
