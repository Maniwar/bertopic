import streamlit as st
import pandas as pd

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from umap import UMAP
from hdbscan import HDBSCAN
from transformers import pipeline

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
    Automatically adapts parameters based on dataset size.
    """

    # Determine dataset size category
    is_small = num_docs and num_docs < 500
    is_medium = num_docs and 500 <= num_docs < 5000
    is_large = num_docs and num_docs >= 5000

    # Adaptive UMAP parameters based on dataset size
    if is_small:
        # Small dataset optimizations
        adaptive_neighbors = min(15, max(2, num_docs // 10))
        adaptive_components = min(5, n_components)
        adaptive_min_dist = 0.0
    elif is_medium:
        # Medium dataset parameters
        adaptive_neighbors = min(30, n_neighbors)
        adaptive_components = min(10, n_components)
        adaptive_min_dist = 0.05
    else:
        # Large dataset parameters
        adaptive_neighbors = n_neighbors
        adaptive_components = n_components
        adaptive_min_dist = min_dist

    umap_model = UMAP(
        n_neighbors=adaptive_neighbors,
        n_components=adaptive_components,
        min_dist=adaptive_min_dist,
        metric='cosine',
        random_state=42,
        low_memory=is_large  # Use low memory mode for large datasets
    )

    # Adaptive clustering based on dataset size
    # Force K-means if explicitly requested OR if dataset is small
    if use_kmeans or is_small:
        # Use K-means for small datasets to guarantee no outliers
        if nr_topics and nr_topics != "auto":
            # Respect user's choice but cap at reasonable maximum (half the documents)
            n_clusters = min(nr_topics, max(2, num_docs // 2)) if num_docs else nr_topics
        else:
            # Automatic: create reasonable number of topics based on dataset size
            if num_docs:
                if num_docs < 30:
                    n_clusters = 3  # Very small dataset
                elif num_docs < 100:
                    n_clusters = min(8, num_docs // 6)  # Small dataset: ~6 docs per topic
                else:
                    n_clusters = min(15, num_docs // 10)  # Medium dataset: ~10 docs per topic
            else:
                n_clusters = 5
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        # HDBSCAN with adaptive parameters for medium/large datasets
        if is_medium:
            adaptive_min_cluster = max(5, min(50, min_cluster_size))
            adaptive_min_samples = max(3, min(10, min_samples))
        else:
            adaptive_min_cluster = min_cluster_size
            adaptive_min_samples = min_samples

        clustering_model = HDBSCAN(
            min_cluster_size=adaptive_min_cluster,
            min_samples=adaptive_min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

    # Adaptive vectorizer based on dataset size
    if is_small:
        max_features = 500
        min_df = 1
        max_df = 0.95
    elif is_medium:
        max_features = 2000
        min_df = 2
        max_df = 0.95
    else:
        max_features = 5000
        min_df = 5
        max_df = 0.95

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),  # Include trigrams for better context
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )

    # Create embedding model with appropriate size
    if is_small or is_medium:
        # Lighter model for smaller datasets
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        # More powerful model for large datasets
        embedding_model = SentenceTransformer('all-mpnet-base-v2')

    # Create representation model chain for better topic quality
    representation_models = [
        KeyBERTInspired(),  # Extract keywords that best represent topics
        MaximalMarginalRelevance(diversity=0.3)  # Balance relevance and diversity
    ]

    # Adaptive minimum topic size
    if is_small:
        adaptive_min_topic_size = max(2, min(10, min_topic_size))
    else:
        adaptive_min_topic_size = min_topic_size

    # Configure c-TF-IDF with seed words if provided
    if seed_words:
        ctfidf_model = ClassTfidfTransformer(
            seed_words=seed_words,
            seed_multiplier=seed_multiplier,
            reduce_frequent_words=True
        )
    else:
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Add guided topics for sentiment separation if requested
    seed_topic_list = None
    if use_guided_sentiment:
        # Create seed topics that guide sentiment separation
        seed_topic_list = [
            ["excellent", "great", "amazing", "outstanding", "wonderful", "fantastic", "superb", "perfect"],  # Positive
            ["terrible", "awful", "horrible", "poor", "bad", "worst", "disappointed", "frustrating"],  # Negative
            ["good", "helpful", "friendly", "responsive", "professional", "efficient", "satisfied"],  # Positive service
            ["unhelpful", "rude", "slow", "incompetent", "useless", "waste", "unresponsive"],  # Negative service
            ["quality", "durable", "reliable", "solid", "well-made", "excellent quality"],  # Positive product
            ["broken", "defective", "cheap", "flimsy", "failed", "poor quality"],  # Negative product
            ["fast", "quick", "early", "prompt", "speedy delivery"],  # Positive delivery
            ["delayed", "late", "slow", "lost", "damaged", "missing"]  # Negative delivery
        ]

    # Create BERTopic model with all configurations
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_models,
        calculate_probabilities=True,
        min_topic_size=adaptive_min_topic_size,
        nr_topics=nr_topics,
        seed_topic_list=seed_topic_list,
        verbose=False
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
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(docs):
    """
    Analyze sentiment of documents for labeling purposes.
    Returns sentiment labels for each document.
    """
    sentiment_analyzer = get_sentiment_analyzer()
    sentiment_labels = []

    with st.spinner("Analyzing sentiment..."):
        # Process in batches for efficiency
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]

            # Get sentiments for batch
            sentiments = sentiment_analyzer(batch, truncation=True, max_length=512)

            for sent in sentiments:
                if sent['label'] == 'POSITIVE' and sent['score'] > 0.75:
                    sentiment_labels.append("positive")
                elif sent['label'] == 'NEGATIVE' and sent['score'] > 0.75:
                    sentiment_labels.append("negative")
                else:
                    sentiment_labels.append("neutral")

    return sentiment_labels

# -----------------------------------------------------
# 4. MAIN STREAMLIT APP
# -----------------------------------------------------
def main():
    # Initialize session state
    if 'topics' not in st.session_state:
        st.session_state.topics = None
    if 'topic_labels' not in st.session_state:
        st.session_state.topic_labels = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'topic_info' not in st.session_state:
        st.session_state.topic_info = None

    st.title("🔍 Advanced BERTopic Analyzer")
    st.write("Scalable topic modeling that works equally well on small (50 docs) and large (100K+ docs) datasets with high-quality topic categories.")

    # Display dataset size indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Status", "Ready", delta="Optimized")
    with col2:
        st.metric("Embedding Model", "Adaptive")
    with col3:
        st.metric("Clustering", "Auto-Selected")

    # --- SIDEBAR ---
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    st.sidebar.header("🎯 Main Controls")
    st.sidebar.info("💡 **These always affect your results:**")

    nr_topics = st.sidebar.selectbox(
        "Number of Topics",
        [None, "auto", 3, 5, 8, 10, 12, 15, 20, 25, 30],
        index=4,  # Default to 8 topics
        help="**This directly controls how many topics you get!** Pick a specific number for guaranteed topics."
    )

    min_topic_size = st.sidebar.slider(
        "Minimum Topic Size",
        1, 50, 5, 1,
        help="Smallest group size. Lower = more small topics, Higher = fewer but bigger topics."
    )

    st.sidebar.header("Advanced Options")

    use_sentiment_separation = st.sidebar.checkbox(
        "🎭 Separate by Sentiment",
        value=True,
        help="Automatically separate positive and negative feedback into different topics. Essential for customer feedback analysis!"
    )

    force_kmeans = st.sidebar.checkbox(
        "Force K-means Clustering",
        value=False,
        help="Always use K-means clustering. Guarantees no outliers but requires specifying number of topics."
    )

    # Only show HDBSCAN/UMAP parameters if they'll actually be used
    show_advanced = st.sidebar.checkbox(
        "Show Advanced Parameters",
        value=False,
        help="⚠️ Only affects LARGE datasets (500+ docs) or when Force K-means is OFF"
    )

    if show_advanced:
        st.sidebar.warning("⚠️ These DON'T affect small datasets (<500 docs) unless you turn OFF 'Force K-means'")

        n_neighbors = st.sidebar.slider(
            "Neighbors (UMAP)",
            2, 100, 15, 1,
            help="How many nearby points to consider for dimensionality reduction."
        )
        n_components = st.sidebar.slider(
            "Components (UMAP)",
            2, 10, 5, 1,
            help="Number of dimensions to reduce to before clustering."
        )
        min_dist = st.sidebar.slider(
            "Min Distance (UMAP)",
            0.0, 1.0, 0.1, 0.01,
            help="How tightly to pack points together."
        )
        min_cluster_size = st.sidebar.slider(
            "Min Cluster Size (HDBSCAN)",
            2, 500, 10, 1,
            help="Minimum points needed to form a cluster."
        )
        min_samples = st.sidebar.slider(
            "Min Samples (HDBSCAN)",
            1, 100, 5, 1,
            help="Minimum points needed to be considered a core point."
        )
    else:
        # Use default values when hidden
        n_neighbors = 15
        n_components = 5
        min_dist = 0.1
        min_cluster_size = 10
        min_samples = 5

    st.sidebar.header("Seed Words (Optional)")
    st.sidebar.write("Boost important words by listing them here (comma-separated).")
    seed_words_str = st.sidebar.text_input(
        "Seed Words",
        "",
        help="Add key words like 'price, delivery, support' to guide the app. It's like telling it to spotlight comments about shipping. **Example:** Type 'refund,' and refund-related feedback gets more focus. **Why it's different:** It steers the topics you care about, unlike settings that tweak grouping mechanics."
    )
    seed_words = [w.strip() for w in seed_words_str.split(",") if w.strip()] if seed_words_str else None
    seed_multiplier = 1.0
    if seed_words:
        seed_multiplier = st.sidebar.slider(
            "Seed Word Boost",
            1.0, 5.0, 2.0, 0.5,
            help="This makes your seed words more important. Think of it as turning up the volume on 'delivery' complaints. **Example:** Set to 3.0, and 'support' in feedback gets triple the attention. **Why it's different:** It only boosts your chosen words, not the whole process like other settings."
        )

    st.sidebar.markdown("""
**Steps:**
1. Upload a CSV with your text data.
2. Select the text column.
3. Adjust topic settings and seed words.
4. Click "Run Topic Modeling."
5. View topics and see all documents grouped by topic.
6. Download the results.
    """)

    # --- MAIN CONTENT ---
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())

        text_col = st.selectbox("Select the text column", df.columns)

        if st.button("Run Topic Modeling"):
            docs = df[text_col].astype(str).tolist()

            # Analyze sentiment if enabled
            sentiment_labels = None
            if use_sentiment_separation:
                sentiment_labels = analyze_sentiment(docs)
                st.success("✅ Analyzed sentiment to guide topic separation")

            # Clear previous results
            st.session_state.topics = None
            st.session_state.topic_labels = None
            st.session_state.processed_df = None

            # Create optimized topic model
            with st.spinner(f"Creating optimized topic model for {len(docs)} documents..."):
                # Show dataset size classification
                if len(docs) < 500:
                    if nr_topics and nr_topics != "auto":
                        st.info(f"📊 Small dataset ({len(docs)} docs) → Creating {nr_topics} topics using K-means")
                        if nr_topics > len(docs) // 2:
                            actual_topics = min(nr_topics, len(docs) // 2)
                            st.warning(f"⚠️ {nr_topics} topics is too many for {len(docs)} docs. Creating {actual_topics} topics instead (max 2 docs/topic).")
                    else:
                        estimated_topics = min(8, len(docs) // 6) if len(docs) < 100 else min(15, len(docs) // 10)
                        st.info(f"📊 Small dataset ({len(docs)} docs) → Auto-detecting ~{estimated_topics} topics")
                elif len(docs) < 5000:
                    if nr_topics and nr_topics != "auto":
                        st.info(f"📊 Medium dataset ({len(docs)} docs) → Creating {nr_topics} topics using HDBSCAN")
                    else:
                        st.info(f"📊 Medium dataset ({len(docs)} docs) → Auto-detecting optimal topics")
                else:
                    st.info(f"📊 Large dataset ({len(docs)} docs) → Using scalable HDBSCAN clustering")

                model = create_bertopic_model(
                    min_topic_size=min_topic_size,
                    nr_topics=nr_topics,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    seed_words=seed_words,
                    seed_multiplier=seed_multiplier,
                    use_kmeans=force_kmeans,
                    num_docs=len(docs),
                    use_guided_sentiment=use_sentiment_separation
                )

                # Fit the model
                topics, probabilities = model.fit_transform(docs)

                # Get initial statistics
                initial_outliers = topics.count(-1)
                total_docs = len(topics)
                outlier_percentage = (initial_outliers / total_docs) * 100

                st.write(f"Initial results: {len(set(topics)) - (1 if -1 in topics else 0)} topics, {initial_outliers} outliers ({outlier_percentage:.1f}%)")

                # Apply intelligent outlier reduction based on dataset size
                if initial_outliers > 0 and len(docs) >= 100:
                    with st.spinner("Applying intelligent outlier reduction..."):
                        # Try probability-based reduction first
                        if probabilities is not None:
                            topics = model.reduce_outliers(
                                docs, topics,
                                probabilities=probabilities,
                                strategy="probabilities",
                                threshold=0.1 if len(docs) < 500 else 0.05
                            )

                        # If still have outliers, try c-TF-IDF
                        remaining_outliers = topics.count(-1)
                        if remaining_outliers > 0 and remaining_outliers > (0.1 * total_docs):
                            topics = model.reduce_outliers(
                                docs, topics,
                                strategy="c-tf-idf",
                                threshold=0.1
                            )

                        # Final statistics
                        final_outliers = topics.count(-1)
                        final_percentage = (final_outliers / total_docs) * 100
                        reduction = initial_outliers - final_outliers

                        if reduction > 0:
                            st.success(f"✅ Reduced outliers from {initial_outliers} to {final_outliers} (-{reduction} docs, now {final_percentage:.1f}%)")
                        else:
                            st.info(f"Outliers: {final_outliers} documents ({final_percentage:.1f}%)")

                topic_info = model.get_topic_info()
                topic_labels = [topic_info[topic_info["Topic"] == t]["Name"].iloc[0] if t in topic_info["Topic"].values else "Outliers" for t in topics]

                # Store results in session state
                st.session_state.topics = topics
                st.session_state.topic_labels = topic_labels
                st.session_state.model = model
                st.session_state.topic_info = topic_info

                # Process and store DataFrame
                df["topic_id"] = topics

                # Add sentiment indicators to labels if sentiment was analyzed
                if use_sentiment_separation and sentiment_labels:
                    # Create labels with sentiment indicators
                    enhanced_labels = []
                    for topic in topics:
                        if topic == -1:
                            enhanced_labels.append("Outliers")
                        else:
                            # Get the base label
                            base_label = topic_info[topic_info["Topic"] == topic]["Name"].iloc[0]

                            # Get sentiment distribution for this topic
                            topic_docs_idx = [j for j, t in enumerate(topics) if t == topic]
                            topic_sentiments = [sentiment_labels[j] for j in topic_docs_idx if j < len(sentiment_labels)]

                            if topic_sentiments:
                                pos_count = topic_sentiments.count("positive")
                                neg_count = topic_sentiments.count("negative")
                                total = len(topic_sentiments)

                                # Add sentiment indicator based on majority
                                if pos_count / total > 0.6:
                                    enhanced_labels.append(f"😊 {base_label}")
                                elif neg_count / total > 0.6:
                                    enhanced_labels.append(f"😞 {base_label}")
                                else:
                                    enhanced_labels.append(f"😐 {base_label}")
                            else:
                                enhanced_labels.append(base_label)

                    df["topic_label"] = enhanced_labels
                    df["sentiment"] = sentiment_labels[:len(df)]  # Add sentiment column
                else:
                    df["topic_label"] = topic_labels

                st.session_state.processed_df = df

            st.success("Topic modeling complete!")

        # Display results from session state if available
        if st.session_state.topics is not None and st.session_state.processed_df is not None:
            topics = st.session_state.topics
            topic_labels = st.session_state.topic_labels
            df = st.session_state.processed_df

            # Display comprehensive results
            st.subheader("📊 Topic Analysis Results")

            # Show topic quality metrics
            unique_topics = len(set(topics)) - (1 if -1 in topics else 0)
            avg_docs_per_topic = len([t for t in topics if t != -1]) / unique_topics if unique_topics > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Topics Found", unique_topics)
            with col2:
                st.metric("Avg Docs/Topic", f"{avg_docs_per_topic:.0f}")
            with col3:
                outlier_count = topics.count(-1)
                st.metric("Outliers", outlier_count)
            with col4:
                coverage = ((len(topics) - outlier_count) / len(topics)) * 100
                st.metric("Coverage", f"{coverage:.1f}%")

            # Show what settings were actually used
            with st.expander("⚙️ Settings Used for This Analysis", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Clustering Method:**", "K-means" if len(df) < 500 or force_kmeans else "HDBSCAN")
                    st.write("**Number of Topics:**", unique_topics)
                    st.write("**Min Topic Size:**", min_topic_size)
                with col2:
                    st.write("**Dataset Size:**", len(df), "documents")
                    st.write("**Avg Docs/Topic:**", f"{avg_docs_per_topic:.0f}")
                    st.write("**Coverage:**", f"{coverage:.1f}%")

            # Show topic distribution
            st.subheader("📋 Topic Distribution")
            topic_counts = df["topic_label"].value_counts()

            # Create a more informative display
            topic_df = pd.DataFrame({
                'Topic': topic_counts.index,
                'Document Count': topic_counts.values,
                'Percentage': (topic_counts.values / len(df) * 100).round(1)
            })
            st.dataframe(topic_df, use_container_width=True, hide_index=True)

            st.subheader("📄 Documents by Topic")

            # Add topic selector for better navigation
            selected_topic = st.selectbox(
                "Select a topic to view documents:",
                options=sorted(df["topic_label"].unique()),
                format_func=lambda x: f"{x} ({len(df[df['topic_label']==x])} docs)"
            )

            # Display documents for selected topic
            if selected_topic:
                topic_docs = df[df["topic_label"] == selected_topic]
                st.markdown(f"### {selected_topic}")
                st.caption(f"Total documents: {len(topic_docs)}")

                # Create a clean display with sentiment if available
                if use_sentiment_separation and "sentiment" in df.columns:
                    display_df = topic_docs[[text_col, "sentiment"]].copy()
                    display_df.columns = ['Document', 'Sentiment']
                    # Color code by sentiment
                    display_df['Sentiment'] = display_df['Sentiment'].apply(
                        lambda x: f"🟢 {x}" if x == "positive" else (f"🔴 {x}" if x == "negative" else f"⚪ {x}")
                    )
                else:
                    display_df = topic_docs[[text_col]].copy()
                    display_df.columns = ['Document']

                # Show documents in a scrollable container with theme-aware colors
                html_table = display_df.to_html(index=False, border=0, escape=False)
                html = f'''
                <div style="width:100%; height:400px; overflow-y:scroll;
                            border: 1px solid var(--border-color, #e0e0e0);
                            border-radius:8px;
                            padding:10px;
                            background-color: var(--background-secondary, rgba(0,0,0,0.02));">
                    <style>
                        @media (prefers-color-scheme: dark) {{
                            div table {{ color: #fafafa; }}
                        }}
                    </style>
                    {html_table}
                </div>
                '''
                st.markdown(html, unsafe_allow_html=True)

            # Hierarchical topic exploration
            st.subheader("🎯 Interactive Topic Explorer")

            # Create tabs for different views
            tab1, tab2 = st.tabs(["Topic List", "Grouped View"])

            with tab1:
                # Regular expandable view with better UX
                for topic in sorted(df["topic_label"].unique()):
                    topic_docs = df[df["topic_label"] == topic]
                    doc_count = len(topic_docs)

                    # Create preview text
                    if doc_count > 0:
                        preview = topic_docs[text_col].iloc[0][:100] + "..."
                    else:
                        preview = "No documents"

                    with st.expander(f"{topic} ({doc_count} documents)"):
                        st.caption(f"Preview: {preview}")
                        st.divider()

                        # Show all documents with better formatting
                        for i, (_, row) in enumerate(topic_docs.iterrows(), 1):
                            st.write(f"{i}. {row[text_col]}")
                            if i < doc_count:
                                st.caption("─" * 50)

            with tab2:
                st.write("### Grouped Topics")
                st.caption("Topics are grouped by keyword similarity")

                # Group topics by shared keywords
                topic_groups = {}
                for topic in df["topic_label"].unique():
                    if "Outliers" in topic:
                        group_key = "Outliers"
                    else:
                        # Use first significant word as group key
                        words = topic.replace("0_", "").replace("1_", "").replace("2_", "").split("_")
                        group_key = words[0] if words else "Other"

                    if group_key not in topic_groups:
                        topic_groups[group_key] = []
                    topic_groups[group_key].append(topic)

                # Display grouped topics
                for group_name, topics_list in sorted(topic_groups.items()):
                    total_docs = sum(len(df[df["topic_label"] == t]) for t in topics_list)

                    with st.expander(f"📁 {group_name} ({len(topics_list)} topics, {total_docs} docs)"):
                        for topic in topics_list:
                            topic_docs = df[df["topic_label"] == topic]
                            st.write(f"**{topic}**")
                            st.caption(f"{len(topic_docs)} documents")

                            # Show first 2 docs as examples
                            for doc in topic_docs[text_col].head(2):
                                st.write(f"  • {doc[:150]}...")
                            st.divider()

            st.subheader("Download Final Results")
            csv_bytes = convert_df_to_csv(df)
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