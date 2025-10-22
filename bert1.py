import streamlit as st
import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Try to import GPU-accelerated libraries
try:
    from cuml.manifold import UMAP
    from cuml.cluster import HDBSCAN
    cuml_ready = True
except ImportError:
    from umap import UMAP
    from hdbscan import HDBSCAN
    cuml_ready = False

# Set Streamlit to wide mode for better layout
st.set_page_config(layout="wide", page_title="Advanced BERTopic Analyzer", page_icon="🔍")

# Force CUDA initialization if available
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()
    # Set default tensor type to CUDA
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# -----------------------------------------------------
# 1. CREATE BERTOPIC MODEL (Cached Resource)
# -----------------------------------------------------
@st.cache_resource(show_spinner=False)
def create_bertopic_model(
    min_topic_size,
    nr_topics,
    min_cluster_size,
    min_samples,
    n_neighbors,
    n_components,
    num_docs=None,
    use_kmeans=False,
    calculate_probabilities=True
):
    """
    Creates a scalable BERTopic model with improved parameters.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Log CUDA status
    if device == 'cuda':
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Adaptive parameters based on dataset size
    is_small = num_docs and num_docs < 500
    
    # UMAP parameters - adjusted for better clustering
    umap_params = {
        'n_neighbors': n_neighbors,  # Now configurable
        'n_components': n_components,  # Now configurable
        'min_dist': 0.0,
        'metric': 'cosine',
    }
    
    if cuml_ready:
        # cuML specific parameters
        umap_params['verbose'] = False
        umap_params['output_type'] = 'numpy'
    else:
        # CPU UMAP parameters
        umap_params['random_state'] = 42
        umap_params['low_memory'] = False
        
    umap_model = UMAP(**umap_params)
    
    # Clustering model
    final_nr_topics = nr_topics if isinstance(nr_topics, int) else None
    
    if use_kmeans:
        clustering_model = KMeans(
            n_clusters=final_nr_topics or 15, 
            random_state=42, 
            n_init='auto'
        )
    else:
        hdbscan_params = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'metric': 'euclidean',
            'prediction_data': True,
        }
        
        if cuml_ready:
            # cuML HDBSCAN parameters
            hdbscan_params['gen_min_span_tree'] = True
            hdbscan_params['output_type'] = 'numpy'
        else:
            # CPU HDBSCAN parameters - use 'leaf' for less outliers
            hdbscan_params['cluster_selection_method'] = 'leaf'  # Changed from 'eom'
            hdbscan_params['cluster_selection_epsilon'] = 0.0
            
        clustering_model = HDBSCAN(**hdbscan_params)
    
    # Improved vectorizer with less restrictive parameters
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,  # Changed from 2 to be less restrictive
        max_df=0.95,  # Added to remove very common terms
        max_features=10000  # Limit features for performance
    )
    
    # Representation models for better topic descriptions
    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3)
    ]
    
    # Choose embedding model based on dataset size and GPU availability
    if is_small or device == 'cpu':
        embedding_model_name = 'all-MiniLM-L6-v2'
    else:
        embedding_model_name = 'all-mpnet-base-v2'
    
    # Initialize embedding model with explicit device setting
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    
    # Force model to device
    if device == 'cuda':
        embedding_model = embedding_model.to('cuda')
    
    # Create BERTopic model with optimized parameters
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
        low_memory=False,  # Use full memory for better performance
    )
    
    return model

# -----------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV byte string for download."""
    return df.to_csv(index=False).encode("utf-8")

def check_cuda_status():
    """Detailed CUDA status check"""
    status = {}
    if torch.cuda.is_available():
        status['available'] = True
        status['device_count'] = torch.cuda.device_count()
        status['current_device'] = torch.cuda.current_device()
        status['device_name'] = torch.cuda.get_device_name(0)
        status['memory_allocated'] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
        status['memory_reserved'] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
    else:
        status['available'] = False
    return status

# -----------------------------------------------------
# 3. MAIN STREAMLIT APP
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

    st.title("🔍 Advanced BERTopic Analyzer")
    st.write("A powerful tool for discovering topics in text data, with GPU acceleration and hierarchical analysis.")
    
    # --- SIDEBAR: CONTROLS AND STATUS ---
    with st.sidebar:
        st.header("System Status")
        
        # Detailed CUDA status
        cuda_status = check_cuda_status()
        if cuda_status['available']:
            st.success(f"✅ CUDA GPU: {cuda_status['device_name']}")
            with st.expander("GPU Details"):
                st.write(f"- Device Count: {cuda_status['device_count']}")
                st.write(f"- Current Device: {cuda_status['current_device']}")
                st.write(f"- Memory Allocated: {cuda_status['memory_allocated']}")
                st.write(f"- Memory Reserved: {cuda_status['memory_reserved']}")
            
            if cuml_ready:
                st.success("✅ cuML ready (GPU-accelerated UMAP/HDBSCAN)")
            else:
                st.warning("⚠️ cuML not found. Install rapids-cuml for full GPU acceleration")
                st.code("conda install -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.11", language="bash")
        else:
            st.error("❌ No CUDA GPU detected. All operations will use CPU.")

        st.header("Upload & Control")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        # If a new file is uploaded, clear all previous results
        if uploaded_file and (st.session_state.uploaded_file_name != uploaded_file.name):
            st.session_state.processed_df = None
            st.session_state.model = None
            st.session_state.ran_params = None
            st.session_state.uploaded_file_name = uploaded_file.name
            st.cache_resource.clear()
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.rerun()

        # --- FORM TO PREVENT RERUNS ON INPUT ---
        with st.form(key="topic_params_form"):
            st.header("🎯 Topic Modeling Controls")
            st.info("Adjust parameters here, then click 'Run Topic Modeling' at the bottom.")
            
            # Main parameters
            nr_topics = st.selectbox(
                "Number of Topics",
                ["auto", 10, 20, 30, 40, 50, 75, 100],
                index=0,
                help="'auto': Let the model decide. Number: Merge clusters to reach this target."
            )
            
            min_topic_size = st.slider(
                "Minimum Topic Size",
                3, 100, 10, 1,
                help="The smallest number of documents allowed in a topic. Lower = more topics, fewer outliers."
            )
            
            st.header("Advanced Options")
            
            # Add tabs for better organization
            tab1, tab2, tab3 = st.tabs(["Clustering", "UMAP", "Performance"])
            
            with tab1:
                force_kmeans = st.checkbox(
                    "Force K-means Clustering",
                    value=False,
                    help="Use K-means for deterministic results (less outliers but might miss nuances)"
                )
                
                if not force_kmeans:
                    st.write("**HDBSCAN Parameters** (lower values = fewer outliers)")
                    min_cluster_size = st.slider(
                        "Min Cluster Size",
                        2, 100, 5, 1,  # Changed default from 15 to 5
                        help="Minimum size of clusters. Lower = more clusters, fewer outliers"
                    )
                    min_samples = st.slider(
                        "Min Samples",
                        1, 50, 1, 1,  # Changed default from 5 to 1
                        help="Core points in a neighborhood. Lower = denser clusters, fewer outliers"
                    )
                else:
                    min_cluster_size, min_samples = 5, 1
            
            with tab2:
                st.write("**UMAP Dimension Reduction** (affects clustering quality)")
                n_neighbors = st.slider(
                    "N Neighbors",
                    2, 100, 15, 1,
                    help="Size of local neighborhood. Lower = more local structure"
                )
                n_components = st.slider(
                    "N Components",
                    2, 30, 10, 1,  # Changed default from 5 to 10
                    help="Dimensions for reduction. Higher = more information preserved"
                )
            
            with tab3:
                calculate_probabilities = st.checkbox(
                    "Calculate Probabilities",
                    value=False,  # Changed to False by default for speed
                    help="Calculate topic probabilities (slower but more detailed)"
                )
            
            # The submit button for the form
            submit_button = st.form_submit_button(label="🚀 Run Topic Modeling")

    # --- MAIN PAGE LOGIC ---

    # This block executes ONLY when the form is submitted
    if submit_button and uploaded_file is not None:
        # Clear CUDA cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        with st.spinner(f"Analyzing {st.session_state.uploaded_file_name}..."):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Read data
            status_text.text("Loading data...")
            progress_bar.progress(10)
            df = pd.read_csv(uploaded_file)
            
            # Infer text column (first string column)
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                text_col = text_cols[0]
            else:
                text_col = df.columns[0]
            
            docs = df[text_col].dropna().astype(str).tolist()
            
            # Show data statistics
            st.info(f"📊 Processing {len(docs)} documents from column '{text_col}'")
            
            # Create and fit the model
            status_text.text("Creating model...")
            progress_bar.progress(30)
            
            model = create_bertopic_model(
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                n_neighbors=n_neighbors,
                n_components=n_components,
                num_docs=len(docs),
                use_kmeans=force_kmeans,
                calculate_probabilities=calculate_probabilities
            )
            
            status_text.text("Fitting model (this may take a while)...")
            progress_bar.progress(50)
            
            # Fit the model
            topics, probs = model.fit_transform(docs)
            
            status_text.text("Processing results...")
            progress_bar.progress(80)
            
            # Store results
            topic_info = model.get_topic_info()
            topic_label_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
            df["topic_id"] = topics
            df["topic_label"] = [topic_label_map.get(t, "Outliers") for t in topics]
            
            # Add probabilities if calculated
            if calculate_probabilities and probs is not None:
                df["topic_probability"] = np.max(probs, axis=1) if len(probs.shape) > 1 else probs
            
            st.session_state.processed_df = df
            st.session_state.model = model
            st.session_state.ran_params = {
                "Number of Topics": nr_topics,
                "Min Topic Size": min_topic_size,
                "Clustering": "K-means" if force_kmeans else "HDBSCAN",
                "Min Cluster Size": min_cluster_size,
                "Min Samples": min_samples,
                "N Neighbors": n_neighbors,
                "N Components": n_components,
                "Calculate Probabilities": calculate_probabilities,
                "CUDA Used": torch.cuda.is_available(),
                "cuML Used": cuml_ready
            }
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
        st.success("✅ Analysis complete! View results below.")
        
        # Show memory usage if CUDA
        if torch.cuda.is_available():
            st.info(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            torch.cuda.empty_cache()

    # --- DISPLAY RESULTS ---
    if st.session_state.processed_df is not None:
        processed_df = st.session_state.processed_df
        model = st.session_state.model
        ran_params = st.session_state.ran_params
        text_col = processed_df.columns[0]

        st.header("📊 Topic Analysis Results")
        
        # Show parameters used
        with st.expander("Show Settings Used for This Analysis"):
            cols = st.columns(3)
            params_items = list(ran_params.items())
            for i, (key, value) in enumerate(params_items):
                cols[i % 3].write(f"**{key}**: {value}")

        # Metrics
        topic_info = model.get_topic_info()
        unique_topics = len(topic_info[topic_info.Topic != -1])
        outlier_count = processed_df['topic_id'].value_counts().get(-1, 0)
        coverage = ((len(processed_df) - outlier_count) / len(processed_df)) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Topics Found", unique_topics)
        col2.metric("Outliers", outlier_count)
        col3.metric("Coverage", f"{coverage:.1f}%")
        col4.metric("Avg Docs/Topic", f"{(len(processed_df) - outlier_count) / max(unique_topics, 1):.0f}")
        
        # Warning if too many outliers
        if coverage < 70:
            st.warning(f"""
            ⚠️ **High outlier rate detected ({100-coverage:.1f}% outliers)**
            
            Try these adjustments to reduce outliers:
            - Decrease 'Min Cluster Size' (currently {ran_params.get('Min Cluster Size', 'N/A')})
            - Decrease 'Min Samples' (currently {ran_params.get('Min Samples', 'N/A')})
            - Increase 'N Components' for UMAP (currently {ran_params.get('N Components', 'N/A')})
            - Consider using K-means clustering instead of HDBSCAN
            """)

        # Topic distribution
        st.subheader("📋 Topic Distribution")
        topic_dist = processed_df["topic_label"].value_counts().reset_index()
        topic_dist.columns = ["Topic", "Document Count"]
        topic_dist["Percentage"] = (topic_dist["Document Count"] / len(processed_df) * 100).round(2)
        
        # Highlight outliers in red
        def highlight_outliers(row):
            if row["Topic"] == "Outliers":
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            topic_dist.style.apply(highlight_outliers, axis=1),
            use_container_width=True
        )

        # Topic details
        st.subheader("🔍 Topic Details")
        topic_info_display = topic_info[['Topic', 'Count', 'Name', 'Representation']].copy()
        topic_info_display['Representation'] = topic_info_display['Representation'].apply(lambda x: ', '.join(x[:5]) if isinstance(x, list) else str(x))
        st.dataframe(topic_info_display, use_container_width=True)

        # Documents by topic
        st.subheader("📄 Documents by Topic")
        unique_labels = sorted(processed_df["topic_label"].unique())
        selected_topic_docs = st.selectbox(
            "Select a topic to view documents:",
            options=unique_labels,
            key="doc_viewer"
        )
        
        if selected_topic_docs:
            topic_docs_df = processed_df[processed_df["topic_label"] == selected_topic_docs]
            st.markdown(f"**{selected_topic_docs}** ({len(topic_docs_df)} documents)")
            
            # Show with probabilities if available
            if "topic_probability" in processed_df.columns:
                display_cols = [text_col, "topic_probability"]
                topic_docs_display = topic_docs_df[display_cols].reset_index(drop=True)
                topic_docs_display["topic_probability"] = topic_docs_display["topic_probability"].round(3)
            else:
                topic_docs_display = topic_docs_df[[text_col]].reset_index(drop=True)
            
            st.dataframe(topic_docs_display, use_container_width=True, height=400)

        # Hierarchical analysis
        st.header("🔎 Hierarchical Analysis: Split a Large Topic")
        topic_list = processed_df['topic_label'].value_counts().index.tolist()
        if "Outliers" in topic_list:
            topic_list.remove("Outliers")
        
        selected_topic_to_split = st.selectbox(
            "Select a topic to split:",
            options=topic_list,
            key="hierarchical_selector"
        )
        
        if st.button(f"Analyze Sub-Topics within '{selected_topic_to_split}'"):
            docs_to_split = processed_df[processed_df['topic_label'] == selected_topic_to_split][text_col].tolist()
            if len(docs_to_split) >= 10:
                with st.spinner(f"Running sub-analysis on {len(docs_to_split)} documents..."):
                    # Use more lenient parameters for sub-analysis
                    sub_topic_model = BERTopic(
                        min_topic_size=max(3, len(docs_to_split) // 10),
                        nr_topics="auto",
                        verbose=False
                    )
                    sub_topics, _ = sub_topic_model.fit_transform(docs_to_split)
                    st.write("### Sub-Topics Found:")
                    sub_topic_info = sub_topic_model.get_topic_info()
                    st.dataframe(sub_topic_info)
                    
                    # Show sub-topic distribution
                    sub_outliers = sum(1 for t in sub_topics if t == -1)
                    st.info(f"Sub-analysis: {len(sub_topic_info)-1} sub-topics found, {sub_outliers} outliers")
            else:
                st.warning("Not enough documents in this topic to perform a sub-analysis (minimum: 10)")
        
        # Download section
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Main Results (CSV)",
                data=convert_df_to_csv(processed_df),
                file_name="bertopic_main_results.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📥 Download Topic Info (CSV)",
                data=convert_df_to_csv(topic_info),
                file_name="bertopic_topic_info.csv",
                mime="text/csv"
            )
            
    elif not uploaded_file:
        st.info("👆 Please upload a CSV file in the sidebar to begin.")
        
        # Show example usage
        with st.expander("📚 Usage Guide"):
            st.markdown("""
            ### How to use this tool:
            
            1. **Upload your CSV file** containing text data
            2. **Adjust parameters** to control topic discovery:
               - **Min Topic Size**: Lower values create more granular topics
               - **Min Cluster Size & Min Samples**: Lower values reduce outliers
               - **N Components**: Higher values preserve more information
            3. **Click 'Run Topic Modeling'** to start analysis
            4. **Review results** and download the enriched dataset
            
            ### Tips for reducing outliers:
            - Start with lower Min Cluster Size (3-5)
            - Set Min Samples to 1
            - Increase N Components (10-15)
            - Try K-means if HDBSCAN produces too many outliers
            """)

if __name__ == "__main__":
    main()
