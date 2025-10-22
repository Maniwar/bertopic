import streamlit as st
import pandas as pd
import torch

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


# -----------------------------------------------------
# 1. CREATE BERTOPIC MODEL (Cached Resource)
# -----------------------------------------------------
@st.cache_resource(show_spinner=False)
def create_bertopic_model(
    min_topic_size,
    nr_topics,
    min_cluster_size,
    min_samples,
    num_docs=None,
    use_kmeans=False
):
    """
    Creates a scalable BERTopic model. Cached to avoid recreating the object.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_small = num_docs and num_docs < 500

    umap_params = {'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'}
    if not cuml_ready:
        umap_params['random_state'] = 42
    umap_model = UMAP(**umap_params)

    final_nr_topics = nr_topics if isinstance(nr_topics, int) else None
    if use_kmeans:
        clustering_model = KMeans(n_clusters=final_nr_topics or 15, random_state=42, n_init='auto')
    else:
        hdbscan_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples, 'metric': 'euclidean', 'prediction_data': True}
        if cuml_ready:
            hdbscan_params['gen_min_span_tree'] = True
        else:
            hdbscan_params['cluster_selection_method'] = 'eom'
        clustering_model = HDBSCAN(**hdbscan_params)

    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    representation_model = [KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.3)]
    embedding_model_name = 'all-MiniLM-L6-v2' if is_small else 'all-mpnet-base-v2'
    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    model = BERTopic(
        embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model, representation_model=representation_model,
        calculate_probabilities=True, min_topic_size=min_topic_size,
        nr_topics=final_nr_topics, verbose=True
    )
    return model

# -----------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV byte string for download."""
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------------------------------
# 3. MAIN STREAMLIT APP
# -----------------------------------------------------
def main():
    # --- SESSION STATE INITIALIZATION ---
    # Define all keys used in the app's state for clarity and to prevent errors
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
        if torch.cuda.is_available():
            st.success(f"CUDA GPU Detected: {torch.cuda.get_device_name(0)}")
            if cuml_ready:
                st.success("cuML is ready. UMAP/HDBSCAN are GPU-accelerated.")
            else:
                st.warning("cuML not found. UMAP/HDBSCAN will use CPU.")
        else:
            st.error("No CUDA GPU detected. All operations will use CPU.")

        st.header("Upload & Control")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        # If a new file is uploaded, clear all previous results
        if uploaded_file and (st.session_state.uploaded_file_name != uploaded_file.name):
            st.session_state.processed_df = None
            st.session_state.model = None
            st.session_state.ran_params = None
            st.session_state.uploaded_file_name = uploaded_file.name
            st.cache_resource.clear()
            st.rerun()

        # --- FORM TO PREVENT RERUNS ON INPUT ---
        with st.form(key="topic_params_form"):
            st.header("🎯 Topic Modeling Controls")
            st.info("Adjust parameters here, then click 'Run Topic Modeling' at the bottom.")
            
            nr_topics = st.selectbox("Number of Topics", ["auto", 10, 20, 30, 40, 50, 75, 100], index=0, help="**'auto'**: Let the model decide. **Number**: Merge clusters to reach this target.")
            min_topic_size = st.slider("Minimum Topic Size", 3, 100, 10, 1, help="The smallest number of documents allowed in a topic.")
            
            st.header("Advanced Options")
            force_kmeans = st.checkbox("Force K-means Clustering", value=False, help="Use K-means instead of HDBSCAN.")
            if not force_kmeans:
                min_cluster_size = st.slider("Min Cluster Size (HDBSCAN)", 2, 100, 15, 1)
                min_samples = st.slider("Min Samples (HDBSCAN)", 1, 100, 5, 1)
            else:
                min_cluster_size, min_samples = 15, 5
            
            # The submit button for the form
            submit_button = st.form_submit_button(label="🚀 Run Topic Modeling")

    # --- MAIN PAGE LOGIC ---

    # This block executes ONLY when the form is submitted
    if submit_button and uploaded_file is not None:
        with st.spinner(f"Analyzing {st.session_state.uploaded_file_name}..."):
            df = pd.read_csv(uploaded_file)
            text_col = df.columns[0] # Simple inference for the text column
            docs = df[text_col].dropna().astype(str).tolist()
            
            # Create and fit the model
            model = create_bertopic_model(
                min_topic_size=min_topic_size, nr_topics=nr_topics,
                min_cluster_size=min_cluster_size, min_samples=min_samples,
                num_docs=len(docs), use_kmeans=force_kmeans
            )
            topics, _ = model.fit_transform(docs)
            
            # Store results and the parameters used in session state
            topic_info = model.get_topic_info()
            topic_label_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
            df["topic_id"] = topics
            df["topic_label"] = [topic_label_map.get(t, "Outliers") for t in topics]
            
            st.session_state.processed_df = df
            st.session_state.model = model
            st.session_state.ran_params = {
                "Number of Topics": nr_topics, "Min Topic Size": min_topic_size,
                "Clustering": "K-means" if force_kmeans else "HDBSCAN",
                "Min Cluster Size": min_cluster_size, "Min Samples": min_samples
            }
        st.success("Analysis complete! View results below.")

    # --- DISPLAY RESULTS (only if they exist in the session state) ---
    if st.session_state.processed_df is not None:
        processed_df = st.session_state.processed_df
        model = st.session_state.model
        ran_params = st.session_state.ran_params
        text_col = processed_df.columns[0]

        st.header("📊 Topic Analysis Results")
        with st.expander("Show Settings Used for This Analysis"):
            st.json(ran_params)

        topic_info = model.get_topic_info()
        unique_topics = len(topic_info[topic_info.Topic != -1])
        outlier_count = processed_df['topic_id'].value_counts().get(-1, 0)
        coverage = ((len(processed_df) - outlier_count) / len(processed_df)) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Topics Found", unique_topics)
        col2.metric("Outliers", outlier_count)
        col3.metric("Coverage", f"{coverage:.1f}%")

        st.subheader("📋 Topic Distribution")
        st.dataframe(processed_df["topic_label"].value_counts().reset_index(), use_container_width=True)

        st.subheader("📄 Documents by Topic")
        unique_labels = sorted(processed_df["topic_label"].unique())
        selected_topic_docs = st.selectbox("Select a topic to view documents:", options=unique_labels, key="doc_viewer")
        if selected_topic_docs:
            topic_docs_df = processed_df[processed_df["topic_label"] == selected_topic_docs][[text_col]].reset_index(drop=True)
            st.markdown(f"**{selected_topic_docs}** ({len(topic_docs_df)} documents)")
            st.dataframe(topic_docs_df, use_container_width=True, height=400)

        # HIERARCHICAL ANALYSIS
        st.header("🔎 Hierarchical Analysis: Split a Large Topic")
        topic_list = processed_df['topic_label'].value_counts().index.tolist()
        if "Outliers" in topic_list: topic_list.remove("Outliers")
        
        selected_topic_to_split = st.selectbox("Select a topic to split:", options=topic_list, key="hierarchical_selector")
        if st.button(f"Analyze Sub-Topics within '{selected_topic_to_split}'"):
            # This part will still cause a rerun, but the main results above are preserved
            docs_to_split = processed_df[processed_df['topic_label'] == selected_topic_to_split][text_col].tolist()
            if len(docs_to_split) >= 10:
                with st.spinner(f"Running sub-analysis on {len(docs_to_split)} documents..."):
                    sub_topic_model = BERTopic(min_topic_size=5, nr_topics="auto", verbose=False)
                    sub_topics, _ = sub_topic_model.fit_transform(docs_to_split)
                    st.write("### Sub-Topics Found:")
                    st.dataframe(sub_topic_model.get_topic_info())
            else:
                st.warning("Not enough documents in this topic to perform a sub-analysis.")
        
        # DOWNLOAD
        st.header("Download Results")
        st.download_button(
            label="Download Main CSV (with topic assignments)",
            data=convert_df_to_csv(processed_df),
            file_name="bertopic_main_results.csv",
            mime="text/csv"
        )
    elif not uploaded_file:
        st.info("Please upload a CSV file in the sidebar to begin.")

if __name__ == "__main__":
    main()
