import streamlit as st
import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit to wide mode for better layout
st.set_page_config(layout="wide", page_title="Interactive BERTopic with Dynamic Adjustment", page_icon="🎯")

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
from collections import Counter
from scipy.spatial.distance import cosine
import plotly.express as px
import plotly.graph_objects as go

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

# Force CUDA initialization if available
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
# TOPIC MERGER FOR MINIMUM SIZE ENFORCEMENT
# -----------------------------------------------------
class TopicMerger:
    """Merge small topics to enforce minimum topic size"""
    
    @staticmethod
    def merge_small_topics(topics, embeddings, min_size=10):
        """
        Merge topics that are smaller than min_size with their nearest neighbors
        """
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
# FAST RECLUSTERING ENGINE
# -----------------------------------------------------
class FastReclusterer:
    """Fast reclustering using pre-computed embeddings and reduced embeddings"""
    
    def __init__(self, documents, embeddings, umap_embeddings=None):
        self.documents = documents
        self.embeddings = embeddings
        self.umap_embeddings = umap_embeddings
        self.use_gpu = torch.cuda.is_available() and cuml_available
        
    def recluster(self, n_topics, min_topic_size=10, use_reduced=True, method='kmeans'):
        """
        Quickly recluster documents into new topics
        
        Args:
            n_topics: Number of topics to create
            min_topic_size: Minimum size per topic
            use_reduced: Whether to use UMAP-reduced embeddings (faster)
            method: 'kmeans' or 'hdbscan'
        
        Returns:
            topics: Array of topic assignments
            topic_info: DataFrame with topic information
        """
        # Choose embeddings to use
        clustering_embeddings = self.umap_embeddings if (use_reduced and self.umap_embeddings is not None) else self.embeddings
        
        # Perform clustering
        if method == 'kmeans':
            if self.use_gpu:
                try:
                    from cuml.cluster import KMeans as cuKMeans
                    clusterer = cuKMeans(n_clusters=n_topics, random_state=42, max_iter=300)
                except:
                    clusterer = KMeans(n_clusters=n_topics, random_state=42, max_iter=300)
            else:
                clusterer = KMeans(n_clusters=n_topics, random_state=42, max_iter=300)
            
            topics = clusterer.fit_predict(clustering_embeddings)
        else:
            # HDBSCAN with dynamic parameters
            min_cluster_size = max(min_topic_size, len(self.documents) // (n_topics * 2))
            
            if self.use_gpu:
                clusterer = cumlHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(1, min_cluster_size // 5),
                    prediction_data=True
                )
            else:
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(1, min_cluster_size // 5),
                    prediction_data=True
                )
            
            topics = clusterer.fit_predict(clustering_embeddings)
        
        # Merge small topics if necessary
        if min_topic_size > 2:
            topics = TopicMerger.merge_small_topics(topics, self.embeddings, min_topic_size)
        
        # Extract topic keywords using c-TF-IDF
        topic_info = self._extract_topic_keywords(topics)
        
        return topics, topic_info
    
    def _extract_topic_keywords(self, topics, top_n_words=10):
        """Extract keywords for each topic using c-TF-IDF"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Prepare documents per topic
        topics_dict = {}
        for idx, topic in enumerate(topics):
            if topic not in topics_dict:
                topics_dict[topic] = []
            topics_dict[topic].append(self.documents[idx])
        
        # Vectorize
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=100
        )
        
        # Calculate c-TF-IDF for each topic
        topic_info_list = []
        for topic_id in sorted(topics_dict.keys()):
            if topic_id == -1:
                continue
                
            topic_docs = topics_dict[topic_id]
            
            # Get top words for this topic
            try:
                # Join documents for this topic
                topic_text = ' '.join(topic_docs[:100])  # Sample for speed
                
                # Simple keyword extraction
                words = topic_text.lower().split()
                word_counts = Counter(words)
                
                # Filter common words
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                               'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                               'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                               'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'it', 'this',
                               'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they'}
                
                filtered_counts = {w: c for w, c in word_counts.items() 
                                 if w not in common_words and len(w) > 2}
                
                # Get top words
                top_words = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
                keywords = [word for word, _ in top_words]
                
                if not keywords:
                    keywords = ['topic', str(topic_id)]
                
            except Exception as e:
                keywords = [f'topic_{topic_id}']
            
            topic_info_list.append({
                'Topic': topic_id,
                'Count': len(topic_docs),
                'Keywords': ', '.join(keywords[:5]),
                'Name': f"Topic {topic_id}: {', '.join(keywords[:3])}"
            })
        
        # Add outliers if present
        if -1 in topics_dict:
            topic_info_list.append({
                'Topic': -1,
                'Count': len(topics_dict[-1]),
                'Keywords': 'Outliers',
                'Name': 'Outliers'
            })
        
        return pd.DataFrame(topic_info_list)

# -----------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------
def check_gpu_capabilities():
    """Check and return GPU/CUDA capabilities"""
    capabilities = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'device_name': None,
        'cuml_available': cuml_available,
    }
    
    if torch.cuda.is_available():
        capabilities['device_count'] = torch.cuda.device_count()
        capabilities['device_name'] = torch.cuda.get_device_name(0)
    
    return capabilities

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def load_embedding_model(model_name):
    """Load and cache the embedding model"""
    if torch.cuda.is_available():
        model = SentenceTransformer(model_name, device='cuda')
        model.max_seq_length = 512
        model.encode = torch.compile(model.encode) if hasattr(torch, 'compile') else model.encode
    else:
        model = SentenceTransformer(model_name, device='cpu')
    return model

@st.cache_data
def compute_embeddings(_model, documents, batch_size=32):
    """Compute and cache embeddings"""
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            embeddings = _model.encode(
                documents,
                show_progress_bar=True,
                batch_size=batch_size,
                convert_to_numpy=True
            )
    else:
        embeddings = _model.encode(
            documents,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True
        )
    return embeddings

@st.cache_data
def compute_umap_embeddings(embeddings, n_neighbors=15, n_components=5):
    """Compute and cache UMAP reduced embeddings"""
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
    
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

def main():
    st.title("🎯 Interactive BERTopic with Dynamic Topic Adjustment")
    
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
    
    # Check GPU capabilities
    gpu_capabilities = check_gpu_capabilities()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # GPU Status Display
        if gpu_capabilities['cuda_available']:
            st.success(f"✅ GPU: {gpu_capabilities['device_name']}")
        else:
            st.warning("⚠️ No GPU detected. Using CPU (slower)")
        
        # File upload
        st.header("📄 Data Input")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df):,} rows")
            
            # Column selection
            text_col = st.selectbox(
                "Select text column", 
                df.columns.tolist(),
                help="Column containing the text to analyze"
            )
            
            # Embedding Settings
            st.header("🧮 Embedding Settings")
            
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
            
            # UMAP Settings
            with st.expander("🔧 UMAP Settings"):
                n_neighbors = st.slider("n_neighbors", 5, 50, 15,
                                       help="Affects local vs global structure")
                n_components = st.slider("n_components", 2, 10, 5,
                                       help="Dimensions for clustering")
            
            # Initial clustering settings
            st.header("🎯 Initial Clustering")
            
            initial_method = st.selectbox(
                "Clustering Method",
                ["K-means (Fast, No Outliers)", "HDBSCAN (Quality, May Have Outliers)"]
            )
            
            if "K-means" in initial_method:
                initial_n_topics = st.slider(
                    "Initial Number of Topics",
                    min_value=2,
                    max_value=min(50, len(df) // 10),
                    value=min(10, len(df) // 50),
                    help="Starting point - you can adjust this dynamically later!"
                )
            else:
                initial_n_topics = None
                
            min_topic_size = st.slider(
                "Minimum Topic Size",
                min_value=2,
                max_value=min(100, len(df) // 10),
                value=min(10, len(df) // 50),
                help="Topics smaller than this will be merged"
            )
            
            # Compute embeddings button
            if st.button("🚀 Compute Embeddings & Initial Topics", type="primary"):
                with st.spinner("Computing embeddings... This is the slow part, but only needs to be done once!"):
                    # Load model and compute embeddings
                    model = load_embedding_model(embedding_model)
                    documents = df[text_col].tolist()
                    
                    # Compute full embeddings
                    embeddings = compute_embeddings(model, documents)
                    
                    # Compute UMAP embeddings for faster reclustering
                    with st.spinner("Computing UMAP reduction for fast reclustering..."):
                        umap_embeddings = compute_umap_embeddings(embeddings, n_neighbors, n_components)
                    
                    # Store in session state
                    st.session_state.embeddings = embeddings
                    st.session_state.umap_embeddings = umap_embeddings
                    st.session_state.documents = documents
                    st.session_state.embeddings_computed = True
                    st.session_state.text_col = text_col
                    
                    # Create reclusterer
                    st.session_state.reclusterer = FastReclusterer(
                        documents, embeddings, umap_embeddings
                    )
                    
                    # Perform initial clustering
                    method = 'kmeans' if "K-means" in initial_method else 'hdbscan'
                    topics, topic_info = st.session_state.reclusterer.recluster(
                        n_topics=initial_n_topics if initial_n_topics else 10,
                        min_topic_size=min_topic_size,
                        use_reduced=True,
                        method=method
                    )
                    
                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.min_topic_size = min_topic_size
                    
                    st.success("✅ Embeddings computed! Now you can adjust topics dynamically below.")
                    st.balloons()
    
    # Main content area
    if st.session_state.embeddings_computed:
        st.success("✅ Embeddings ready! Use the slider below to dynamically adjust the number of topics.")
        
        # Interactive controls section
        st.header("🎚️ Dynamic Topic Adjustment")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Determine max topics based on min_topic_size
            max_topics = min(100, len(st.session_state.documents) // st.session_state.min_topic_size)
            
            # Interactive slider for number of topics
            n_topics_slider = st.slider(
                "🎯 **Number of Topics** (Adjust in real-time!)",
                min_value=2,
                max_value=max_topics,
                value=min(10, len(st.session_state.documents) // 50),
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
                value=True,
                help="Faster reclustering using reduced dimensions"
            )
        
        # Add a debounce mechanism using a form
        with st.form(key='recluster_form'):
            st.write(f"Click below to recluster with **{n_topics_slider} topics**")
            recluster_button = st.form_submit_button("🔄 Recluster Now", use_container_width=True)
        
        if recluster_button or (st.session_state.current_topics is None):
            with st.spinner(f"Reclustering into {n_topics_slider} topics... (This is fast!)"):
                # Perform reclustering
                method = 'kmeans' if "K-means" in clustering_method else 'hdbscan'
                topics, topic_info = st.session_state.reclusterer.recluster(
                    n_topics=n_topics_slider,
                    min_topic_size=st.session_state.min_topic_size,
                    use_reduced=use_reduced,
                    method=method
                )
                
                st.session_state.current_topics = topics
                st.session_state.current_topic_info = topic_info
                
                st.success(f"✅ Reclustered into {len(topic_info[topic_info['Topic'] != -1])} topics!")
        
        # Display results
        if st.session_state.current_topics is not None:
            topics = st.session_state.current_topics
            topic_info = st.session_state.current_topic_info
            
            # Calculate metrics
            total_docs = len(topics)
            unique_topics = len(set(topics)) - (1 if -1 in topics else 0)
            outlier_count = sum(1 for t in topics if t == -1)
            coverage = ((total_docs - outlier_count) / total_docs) * 100 if total_docs > 0 else 0
            
            # Metrics display
            st.header("📊 Current Results")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Documents", f"{total_docs:,}")
            with col2:
                st.metric("Topics", unique_topics)
            with col3:
                st.metric("Outliers", f"{outlier_count:,}")
            with col4:
                st.metric("Coverage", f"{coverage:.1f}%")
            with col5:
                # Calculate balance score
                topic_counts = Counter(topics)
                counts = [c for t, c in topic_counts.items() if t != -1]
                balance = 1 - (np.std(counts) / np.mean(counts)) if counts else 0
                st.metric("Balance", f"{balance:.2f}")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Topic Overview", 
                "📈 Interactive Visualization", 
                "🔍 Document Explorer",
                "📉 Distribution Analysis",
                "💾 Export"
            ])
            
            with tab1:
                st.subheader("Topic Information")
                
                # Display topic info
                display_df = topic_info.copy()
                display_df = display_df[display_df['Topic'] != -1] if -1 in display_df['Topic'].values else display_df
                display_df['Percentage'] = (display_df['Count'] / total_docs * 100).round(2)
                
                # Color code by size
                def color_code_size(row):
                    if row['Percentage'] > 30:
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['Percentage'] > 20:
                        return ['background-color: #fff4cc'] * len(row)
                    elif row['Percentage'] < 2:
                        return ['background-color: #ccf4ff'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = display_df.style.apply(color_code_size, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Legend
                st.caption("🔴 Red: >30% (Too Large) | 🟡 Yellow: >20% (Large) | 🔵 Blue: <2% (Small)")
            
            with tab2:
                st.subheader("Interactive Topic Visualization")
                
                # Create scatter plot of topics
                if len(set(topics)) > 1:
                    # Prepare data
                    viz_df = pd.DataFrame({
                        'Topic': topics,
                        'Document': st.session_state.documents
                    })
                    
                    # If we have 2D UMAP, use it for visualization
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
                    
                    # Create interactive scatter plot
                    fig = px.scatter(
                        viz_df_sample,
                        x='X', y='Y',
                        color='Topic_Label',
                        hover_data={'Document': True, 'Topic': True},
                        title='Document Clusters in 2D Space',
                        height=600
                    )
                    
                    fig.update_layout(
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 topics for visualization")
            
            with tab3:
                st.subheader("Document Explorer")
                
                # Select topic to explore
                topic_options = [f"Topic {t}" for t in sorted(set(topics)) if t != -1]
                if -1 in topics:
                    topic_options.append("Outliers")
                
                selected_topic_str = st.selectbox("Select topic to explore", topic_options)
                
                # Parse topic number
                if selected_topic_str == "Outliers":
                    selected_topic = -1
                else:
                    selected_topic = int(selected_topic_str.replace("Topic ", ""))
                
                # Get documents for this topic
                topic_docs_idx = [i for i, t in enumerate(topics) if t == selected_topic]
                topic_docs = [st.session_state.documents[i] for i in topic_docs_idx[:100]]  # Limit to 100
                
                st.write(f"**Showing up to 100 documents from {selected_topic_str} (Total: {len(topic_docs_idx)})**")
                
                # Display sample documents
                for i, doc in enumerate(topic_docs[:10]):
                    with st.expander(f"Document {i+1}"):
                        st.write(doc[:500] + "..." if len(doc) > 500 else doc)
                
            with tab4:
                st.subheader("Topic Distribution Analysis")
                
                # Prepare distribution data
                topic_counts = Counter([t for t in topics if t != -1])
                dist_df = pd.DataFrame(
                    list(topic_counts.items()),
                    columns=['Topic', 'Count']
                )
                dist_df = dist_df.sort_values('Count', ascending=False)
                dist_df['Topic_Label'] = dist_df['Topic'].apply(lambda x: f"Topic {x}")
                
                # Bar chart
                fig_bar = px.bar(
                    dist_df,
                    x='Topic_Label',
                    y='Count',
                    title='Document Distribution Across Topics',
                    color='Count',
                    color_continuous_scale='Viridis',
                    height=400
                )
                
                # Add mean line
                mean_count = dist_df['Count'].mean()
                fig_bar.add_hline(
                    y=mean_count,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean"
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie chart
                fig_pie = px.pie(
                    dist_df.head(10),  # Top 10 topics
                    values='Count',
                    names='Topic_Label',
                    title='Top 10 Topics Distribution'
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab5:
                st.subheader("💾 Export Results")
                
                # Prepare export dataframe
                export_df = st.session_state.df.copy()
                export_df['Topic'] = topics
                export_df['Topic_Label'] = [f"Topic {t}" if t != -1 else "Outliers" for t in topics]
                
                # Add topic keywords
                topic_keywords = {}
                for _, row in topic_info.iterrows():
                    topic_keywords[row['Topic']] = row['Keywords']
                export_df['Topic_Keywords'] = export_df['Topic'].map(topic_keywords)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=convert_df_to_csv(export_df),
                        file_name=f"interactive_bertopic_results_{n_topics_slider}_topics.csv",
                        mime="text/csv",
                        help="Full dataset with topic assignments"
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download Topic Info (CSV)",
                        data=convert_df_to_csv(topic_info),
                        file_name=f"topic_info_{n_topics_slider}_topics.csv",
                        mime="text/csv",
                        help="Topic descriptions and statistics"
                    )
                
                # Settings export
                settings = {
                    'n_topics': n_topics_slider,
                    'min_topic_size': st.session_state.min_topic_size,
                    'method': clustering_method,
                    'coverage': f"{coverage:.1f}%",
                    'outliers': outlier_count,
                    'balance_score': f"{balance:.2f}"
                }
                
                st.download_button(
                    label="📥 Download Settings (JSON)",
                    data=pd.Series(settings).to_json(),
                    file_name=f"settings_{n_topics_slider}_topics.json",
                    mime="application/json",
                    help="Current configuration"
                )
    
    elif not uploaded_file:
        # Welcome screen
        st.info("👆 Please upload a CSV file in the sidebar to begin.")
        
        # Feature highlights
        st.header("✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚡ One-Time Embedding")
            st.write("""
            - Compute embeddings **only once**
            - Embeddings are the slow part (30-60s)
            - Cached for entire session
            - No need to recompute
            """)
        
        with col2:
            st.subheader("🎚️ Dynamic Adjustment")
            st.write("""
            - **Real-time reclustering** (<1 second)
            - Interactive slider control
            - Try different topic numbers
            - Find your sweet spot quickly
            """)
        
        with col3:
            st.subheader("📊 Instant Feedback")
            st.write("""
            - See results immediately
            - Balance score updates
            - Coverage metrics
            - Visual distribution charts
            """)
        
        # Workflow
        st.header("🔄 Workflow")
        
        st.write("""
        1. **Upload Data** → Load your CSV file
        2. **Compute Embeddings** → One-time computation (30-60 seconds)
        3. **Adjust Dynamically** → Use slider to change topic count instantly
        4. **Find Sweet Spot** → Experiment until you get the perfect balance
        5. **Export Results** → Download when satisfied
        """)
        
        # Performance tips
        with st.expander("🚀 Performance Tips"):
            st.write("""
            **For fastest interactive adjustment:**
            - ✅ Use "K-means (Fast)" method
            - ✅ Keep "Use UMAP reduction" checked
            - ✅ Use GPU if available (auto-detected)
            
            **Reclustering speed:**
            - With UMAP reduction + K-means: <1 second
            - Without UMAP reduction: 2-5 seconds
            - HDBSCAN: 3-10 seconds
            
            **Embedding computation (one-time):**
            - Small dataset (1-5k docs): 10-30 seconds
            - Medium dataset (5-20k docs): 30-90 seconds
            - Large dataset (20k+ docs): 2-5 minutes
            """)

if __name__ == "__main__":
    main()
