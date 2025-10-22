import streamlit as st
import pandas as pd
import torch

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline

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
# 1. CREATE BERTOPIC MODEL
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
    Automatically adapts parameters based on dataset size and leverages GPU if available.
    """
    # Check for CUDA availability for GPU acceleration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_enabled = (device == 'cuda') and cuml_ready

    if gpu_enabled:
        st.success("✅ CUDA-enabled GPU detected. Using cuML for accelerated UMAP and HDBSCAN.")
    else:
        st.info("ℹ️ No CUDA-enabled GPU with cuML detected. Using CPU-based UMAP and HDBSCAN.")


    # Determine dataset size category
    is_small = num_docs and num_docs < 500
    is_medium = num_docs and 500 <= num_docs < 5000
    is_large = num_docs and num_docs >= 5000

    # Adaptive UMAP parameters based on dataset size
    if is_small:
        adaptive_neighbors = min(15, max(2, num_docs // 10))
        adaptive_components = min(5, n_components)
        adaptive_min_dist = 0.0
    elif is_medium:
        adaptive_neighbors = min(30, n_neighbors)
        adaptive_components = min(10, n_components)
        adaptive_min_dist = 0.05
    else: # Large dataset
        adaptive_neighbors = n_neighbors
        adaptive_components = n_components
        adaptive_min_dist = min_dist

    umap_params = {
        'n_neighbors': adaptive_neighbors,
        'n_components': adaptive_components,
        'min_dist': adaptive_min_dist,
        'metric': 'cosine'
    }
    if not gpu_enabled:
        umap_params['random_state'] = 42
        umap_params['low_memory'] = is_large

    umap_model = UMAP(**umap_params)

    # Adaptive clustering
    if use_kmeans or (is_small and not gpu_enabled): # Force K-means if requested or if dataset is small (on CPU)
        if nr_topics and nr_topics != "auto":
            n_clusters = min(nr_topics, max(2, num_docs // 2)) if num_docs else nr_topics
        else: # Automatic topic number for small datasets
            if num_docs:
                if num_docs < 30: n_clusters = 3
                elif num_docs < 100: n_clusters = min(8, num_docs // 6)
                else: n_clusters = min(15, num_docs // 10)
            else:
                n_clusters = 5
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    else: # HDBSCAN with adaptive parameters (GPU or medium/large CPU)
        if is_medium:
            adaptive_min_cluster = max(5, min(50, min_cluster_size))
            adaptive_min_samples = max(3, min(10, min_samples))
        else:
            adaptive_min_cluster = min_cluster_size
            adaptive_min_samples = min_samples

        hdbscan_params = {
            'min_cluster_size': adaptive_min_cluster,
            'min_samples': adaptive_min_samples,
            'metric': 'euclidean',
            'prediction_data': True
        }
        if gpu_enabled:
            hdbscan_params['gen_min_span_tree'] = True
        else:
            hdbscan_params['cluster_selection_method'] = 'eom'

        clustering_model = HDBSCAN(**hdbscan_params)


    # Adaptive vectorizer
    if is_small: max_features, min_df, max_df = 500, 1, 0.95
    elif is_medium: max_features, min_df, max_df = 2000, 2, 0.95
    else: max_features, min_df, max_df = 5000, 5, 0.95

    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3), max_features=max_features, min_df=min_df, max_df=max_df)

    # Create embedding model with appropriate size and move to GPU if available
    embedding_model_name = 'all-MiniLM-L6-v2' if is_small or is_medium else 'all-mpnet-base-v2'
    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    # Improved representation model for more readable labels
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == 'cuda' else -1)
    generator = TextGeneration(summarizer)

    representation_models = {
        "Main": KeyBERTInspired(),
        "Summary": generator,
        "MMR": MaximalMarginalRelevance(diversity=0.3)
    }

    # Adaptive minimum topic size
    adaptive_min_topic_size = max(2, min(10, min_topic_size)) if is_small else min_topic_size

    # Configure c-TF-IDF with seed words if provided
    ctfidf_model = ClassTfidfTransformer(
        seed_words=seed_words,
        seed_multiplier=seed_multiplier,
        reduce_frequent_words=True
    ) if seed_words else ClassTfidfTransformer(reduce_frequent_words=True)

    # Guided topics for sentiment separation
    seed_topic_list = None
    if use_guided_sentiment:
        seed_topic_list = [
            ["excellent", "great", "amazing", "outstanding", "wonderful", "fantastic", "superb", "perfect"],
            ["terrible", "awful", "horrible", "poor", "bad", "worst", "disappointed", "frustrating"],
            ["quality", "durable", "reliable", "solid", "well-made", "excellent quality"],
            ["broken", "defective", "cheap", "flimsy", "failed", "poor quality"],
            ["fast", "quick", "early", "prompt", "speedy delivery"],
            ["delayed", "late", "slow", "lost", "damaged", "missing"]
        ]

    # Handle nr_topics parameter correctly
    # If a number is given, it's a request to reduce to that many topics after clustering
    # If 'auto' or None, the clustering algorithm decides.
    final_nr_topics = nr_topics if isinstance(nr_topics, int) else None

    # Create BERTopic model
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_models,
        calculate_probabilities=True,
        min_topic_size=adaptive_min_topic_size,
        nr_topics=final_nr_topics,
        seed_topic_list=seed_topic_list,
        verbose=True
    )
    return model

# -----------------------------------------------------
# 2. CONVERT DF TO CSV
# -----------------------------------------------------
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------------------------------
# 3. SENTIMENT PREPROCESSING
# -----------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer():
    """Load sentiment analysis pipeline"""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

def analyze_sentiment(docs):
    """Analyze sentiment of documents for labeling purposes."""
    sentiment_analyzer = get_sentiment_analyzer()
    sentiment_labels = []
    with st.spinner("Analyzing sentiment..."):
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            sentiments = sentiment_analyzer(batch, truncation=True, max_length=512)
            for sent in sentiments:
                if sent['label'] == 'POSITIVE' and sent['score'] > 0.75: sentiment_labels.append("positive")
                elif sent['label'] == 'NEGATIVE' and sent['score'] > 0.75: sentiment_labels.append("negative")
                else: sentiment_labels.append("neutral")
    return sentiment_labels

# -----------------------------------------------------
# 4. MAIN STREAMLIT APP
# -----------------------------------------------------
def main():
    # Session state initialization
    for key in ['topics', 'topic_labels', 'processed_df', 'model', 'topic_info']:
        if key not in st.session_state:
            st.session_state[key] = None

    st.title("🔍 Advanced BERTopic Analyzer")
    st.write("Scalable topic modeling that works equally well on small (50 docs) and large (100K+ docs) datasets with high-quality topic categories.")

    # --- SIDEBAR ---
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    st.sidebar.header("🎯 Main Controls")
    nr_topics_options = ["auto", 3, 5, 8, 10, 12, 15, 20, 25, 30]
    nr_topics = st.sidebar.selectbox(
        "Number of Topics",
        nr_topics_options,
        index=3,
        help="**'auto'**: Let the model decide the optimal number of topics. **Number**: Aim for a specific number of topics."
    )

    min_topic_size = st.sidebar.slider("Minimum Topic Size", 1, 50, 5, 1, help="Smallest group size. Lower = more small topics.")

    st.sidebar.header("Advanced Options")
    use_sentiment_separation = st.sidebar.checkbox("🎭 Separate by Sentiment", value=True, help="Guide the model to separate positive and negative feedback.")
    force_kmeans = st.sidebar.checkbox("Force K-means Clustering", value=False, help="Use K-means instead of HDBSCAN. Guarantees no outliers but requires a fixed number of topics.")

    show_advanced = st.sidebar.checkbox("Show Advanced Parameters", value=False, help="⚠️ Only affects LARGE datasets (>5000 docs) or when Force K-means is OFF")

    if show_advanced:
        n_neighbors = st.sidebar.slider("Neighbors (UMAP)", 2, 100, 15, 1)
        n_components = st.sidebar.slider("Components (UMAP)", 2, 10, 5, 1)
        min_dist = st.sidebar.slider("Min Distance (UMAP)", 0.0, 1.0, 0.1, 0.01)
        min_cluster_size = st.sidebar.slider("Min Cluster Size (HDBSCAN)", 2, 500, 10, 1)
        min_samples = st.sidebar.slider("Min Samples (HDBSCAN)", 1, 100, 5, 1)
    else:
        n_neighbors, n_components, min_dist, min_cluster_size, min_samples = 15, 5, 0.1, 10, 5

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
            docs = df[text_col].astype(str).tolist()

            with st.spinner(f"Creating optimized topic model for {len(docs)} documents..."):
                model = create_bertopic_model(
                    min_topic_size=min_topic_size,
                    nr_topics=nr_topics,
                    n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
                    min_cluster_size=min_cluster_size, min_samples=min_samples,
                    seed_words=seed_words, use_kmeans=force_kmeans,
                    num_docs=len(docs), use_guided_sentiment=use_sentiment_separation
                )

                topics, probabilities = model.fit_transform(docs)

                # Use the generated summary as the main topic label
                model.set_topic_labels(model.get_topic_info()['Summary'])

                topic_info = model.get_topic_info()
                st.session_state.topic_info = topic_info
                st.session_state.model = model

                # Create a mapping from topic ID to the new custom label
                topic_label_map = {row['Topic']: row['CustomName'] for index, row in topic_info.iterrows()}

                df["topic_id"] = topics
                df["topic_label"] = [topic_label_map.get(t, "Outliers") for t in topics]

                # Add sentiment if analyzed
                if use_sentiment_separation:
                    sentiment_labels = analyze_sentiment(docs)
                    df["sentiment"] = sentiment_labels[:len(df)]

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
        st.info("Please upload a CSV file in the sidebar to begin.")

if __name__ == "__main__":
    main()
