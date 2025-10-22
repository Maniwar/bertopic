import streamlit as st
import pandas as pd
import torch

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
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
# 1. CREATE BERTOPIC MODEL (FIXED)
# -----------------------------------------------------
@st.cache_resource(show_spinner=False)
def create_bertopic_model(
    min_topic_size,
    nr_topics,
    n_neighbors,
    n_components,
    min_dist,
    min_cluster_size,
    min_samples,
    seed_words=None,
    seed_multiplier=1.0,
    use_kmeans=False,
    num_docs=None,
    use_guided_sentiment=False
):
    """
    Creates a scalable BERTopic model optimized for both small and large datasets.
    This version uses a robust keyword-based representation model to avoid errors.
    """
    # Check for CUDA availability for GPU acceleration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine dataset size category
    is_small = num_docs and num_docs < 500
    is_medium = num_docs and 500 <= num_docs < 5000
    is_large = num_docs and num_docs >= 5000

    # Adaptive UMAP parameters
    if is_small:
        adaptive_neighbors, adaptive_components, adaptive_min_dist = min(15, max(2, num_docs // 10)), min(5, n_components), 0.0
    elif is_medium:
        adaptive_neighbors, adaptive_components, adaptive_min_dist = min(30, n_neighbors), min(10, n_components), 0.05
    else:
        adaptive_neighbors, adaptive_components, adaptive_min_dist = n_neighbors, n_components, min_dist

    umap_params = {'n_neighbors': adaptive_neighbors, 'n_components': adaptive_components, 'min_dist': adaptive_min_dist, 'metric': 'cosine'}
    if not cuml_ready:
        umap_params['random_state'] = 42
        umap_params['low_memory'] = is_large
    umap_model = UMAP(**umap_params)

    # Adaptive clustering
    final_nr_topics = nr_topics if isinstance(nr_topics, int) else None
    if use_kmeans or (is_small and final_nr_topics is not None):
        clustering_model = KMeans(n_clusters=final_nr_topics or 10, random_state=42, n_init='auto')
    else:
        hdbscan_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples, 'metric': 'euclidean', 'prediction_data': True}
        if cuml_ready:
            hdbscan_params['gen_min_span_tree'] = True
        else:
            hdbscan_params['cluster_selection_method'] = 'eom'
        clustering_model = HDBSCAN(**hdbscan_params)

    # Vectorizer
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.9)

    # Embedding model
    embedding_model_name = 'all-MiniLM-L6-v2' if is_small or is_medium else 'all-mpnet-base-v2'
    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    # ** FIX IS HERE: Using a robust, keyword-based representation model **
    # This combination generates clear topic labels without crashing.
    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3)
    ]

    # c-TF-IDF with optional seeding
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, seed_words=seed_words, seed_multiplier=seed_multiplier) if seed_words else ClassTfidfTransformer(reduce_frequent_words=True)

    # Guided topics for sentiment
    seed_topic_list = None
    if use_guided_sentiment:
        seed_topic_list = [
            ["excellent", "great", "amazing", "outstanding", "wonderful", "fantastic", "superb", "perfect"],
            ["terrible", "awful", "horrible", "poor", "bad", "worst", "disappointed", "frustrating"],
        ]

    # Assemble the BERTopic model
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        min_topic_size=min_topic_size,
        nr_topics=final_nr_topics,
        seed_topic_list=seed_topic_list,
        verbose=True
    )
    return model

# -----------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------------------------------
# 3. MAIN STREAMLIT APP (FIXED)
# -----------------------------------------------------
def main():
    # Session state initialization
    for key in ['topics', 'processed_df', 'model', 'topic_info']:
        if key not in st.session_state:
            st.session_state[key] = None

    st.title("🔍 Advanced BERTopic Analyzer")
    st.write("Scalable topic modeling that works equally well on small (50 docs) and large (100K+ docs) datasets with high-quality topic categories.")
    
    # --- System Diagnostics ---
    st.sidebar.header("System Status")
    if torch.cuda.is_available():
        st.sidebar.success(f"CUDA GPU Detected: {torch.cuda.get_device_name(0)}")
        if cuml_ready:
            st.sidebar.success("cuML is ready. UMAP/HDBSCAN are GPU-accelerated.")
        else:
            st.sidebar.warning("cuML not found. UMAP/HDBSCAN will use CPU.")
    else:
        st.sidebar.error("No CUDA GPU detected. All operations will use CPU.")

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    st.sidebar.header("🎯 Main Controls")
    nr_topics = st.sidebar.selectbox("Number of Topics", ["auto", 5, 8, 10, 12, 15, 20, 25, 30, 50], index=0, help="**'auto'**: Let the model decide. **Number**: Aim for a specific number of topics.")
    min_topic_size = st.sidebar.slider("Minimum Topic Size", 1, 100, 10, 1, help="Smallest allowable topic size. Higher values merge smaller topics.")

    st.sidebar.header("Advanced Options")
    use_sentiment_separation = st.sidebar.checkbox("🎭 Separate by Sentiment", value=True, help="Guide the model to separate positive and negative feedback.")
    force_kmeans = st.sidebar.checkbox("Force K-means Clustering", value=False, help="Use K-means instead of HDBSCAN. Guarantees no outliers but requires a fixed number of topics.")

    # Only show advanced params if relevant
    if not force_kmeans:
        min_cluster_size = st.sidebar.slider("Min Cluster Size (HDBSCAN)", 2, 500, 15, 1, help="Minimum points needed to form a cluster.")
        min_samples = st.sidebar.slider("Min Samples (HDBSCAN)", 1, 100, 5, 1, help="How conservative to be with clustering.")
    else:
        min_cluster_size, min_samples = 15, 5
        
    # Default UMAP params (usually don't need changing)
    n_neighbors, n_components, min_dist = 15, 5, 0.1

    st.sidebar.header("Seed Words (Optional)")
    seed_words_str = st.sidebar.text_input("Seed Words (comma-separated)", "", help="e.g., 'price, delivery, support'")
    seed_words = [w.strip() for w in seed_words_str.split(",") if w.strip()] if seed_words_str else None

    # --- MAIN CONTENT ---
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())

        text_col = st.selectbox("Select the text column", df.columns)

        if st.button("Run Topic Modeling"):
            docs = df[text_col].dropna().astype(str).tolist()

            with st.spinner(f"Creating and fitting model for {len(docs)} documents... This may take a while."):
                model = create_bertopic_model(
                    min_topic_size=min_topic_size, nr_topics=nr_topics,
                    n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
                    min_cluster_size=min_cluster_size, min_samples=min_samples,
                    seed_words=seed_words, use_kmeans=force_kmeans,
                    num_docs=len(docs), use_guided_sentiment=use_sentiment_separation
                )

                topics, _ = model.fit_transform(docs)

                # ** FIX IS HERE: Using the default 'Name' column for labels **
                topic_info = model.get_topic_info()
                st.session_state.topic_info = topic_info
                st.session_state.model = model

                # Create a mapping from topic ID to the keyword-based label
                topic_label_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}

                # Assign labels to the original DataFrame
                df["topic_id"] = topics
                df["topic_label"] = [topic_label_map.get(t, "Outliers") for t in topics]
                st.session_state.processed_df = df

            st.success("Topic modeling complete!")

        if st.session_state.processed_df is not None:
            processed_df = st.session_state.processed_df
            topic_info = st.session_state.topic_info

            st.subheader("📊 Topic Analysis Results")
            unique_topics = len(topic_info[topic_info.Topic != -1])
            outlier_count = processed_df['topic_id'].value_counts().get(-1, 0)
            coverage = ((len(processed_df) - outlier_count) / len(processed_df)) * 100 if len(processed_df) > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Topics Found", unique_topics)
            col2.metric("Outliers", outlier_count)
            col3.metric("Coverage", f"{coverage:.1f}%")

            st.subheader("📋 Topic Distribution")
            st.dataframe(processed_df["topic_label"].value_counts().reset_index(), use_container_width=True)

            st.subheader("📄 Documents by Topic")
            unique_labels = sorted(processed_df["topic_label"].unique())
            selected_topic = st.selectbox("Select a topic to view documents:", options=unique_labels)

            if selected_topic:
                topic_docs_df = processed_df[processed_df["topic_label"] == selected_topic][[text_col]]
                st.markdown(f"**{selected_topic}** ({len(topic_docs_df)} documents)")
                st.dataframe(topic_docs_df, use_container_width=True, height=400)

            st.subheader("Download Final Results")
            csv_bytes = convert_df_to_csv(processed_df)
            st.download_button(
                label="Download CSV (with topic assignments)",
                data=csv_bytes,
                file_name="bertopic_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
