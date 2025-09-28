import streamlit as st
import pandas as pd
import tensorflow as tf

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from umap import UMAP
from hdbscan import HDBSCAN

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
    num_docs=None
):
    """
    Creates a scalable BERTopic model optimized for both small and large datasets.
    Automatically adapts parameters based on dataset size.
    """
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

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
            n_clusters = min(nr_topics, max(2, num_docs // 10)) if num_docs else nr_topics
        else:
            n_clusters = min(10, max(2, num_docs // 15)) if num_docs else 5
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
        verbose=False
    )

    return model

# -----------------------------------------------------
# 2. SENTIMENT ANALYSIS FOR TOPIC SEPARATION
# -----------------------------------------------------
def analyze_sentiment(documents):
    """Analyze sentiment of documents for topic separation"""
    try:
        from transformers import pipeline

        # Use reliable sentiment model
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        sentiment_labels = []
        for doc in documents:
            try:
                # Clean and truncate document
                clean_doc = doc.strip()
                if len(clean_doc) < 3:
                    sentiment_labels.append('neutral')
                    continue

                # Get sentiment
                result = sentiment_pipeline(clean_doc[:512])
                label = result[0]['label'].upper()
                confidence = result[0]['score']

                # Only assign sentiment if confidence is good
                if confidence > 0.65:
                    if label == 'POSITIVE':
                        sentiment_labels.append('positive')
                    elif label == 'NEGATIVE':
                        sentiment_labels.append('negative')
                    else:
                        sentiment_labels.append('neutral')
                else:
                    sentiment_labels.append('neutral')

            except Exception:
                sentiment_labels.append('neutral')

        return sentiment_labels

    except ImportError:
        st.warning("⚠️ Transformers library not available. Sentiment separation disabled.")
        return ['neutral'] * len(documents)

# -----------------------------------------------------
# 3. CONVERT DF TO CSV
# -----------------------------------------------------
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------------------------------
# 4. MAIN STREAMLIT APP
# -----------------------------------------------------
def main():
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

    st.sidebar.header("Topic Settings")
    min_topic_size = st.sidebar.slider(
        "Minimum Topic Size",
        1, 500, 10, 1,
        help="This sets the smallest number of customer comments needed to keep a topic. Imagine sorting feedback into piles like 'slow delivery.' If you set this to 10, only piles with 10 or more comments stay—smaller ones get mixed into bigger piles. **Example:** A group with 3 comments about 'bad packaging' gets merged if this is 10. **Why it's different:** It cleans up topics at the end, unlike Min Cluster Size, which sets group sizes at the start."
    )
    nr_topics = st.sidebar.selectbox(
        "Number of Topics",
        [None, "auto", 5, 10, 15, 20],
        index=0,
        help="This picks how many main topics you get, like choosing to focus on 5 big customer issues. **Example:** Set it to 5, and you might get 'pricing,' 'quality,' 'shipping,' 'service,' and 'returns.' Pick 'auto' to let the app decide. **Why it's different:** It locks in the exact number of topics, unlike other settings that just adjust how groups form or merge."
    )
    n_neighbors = st.sidebar.slider(
        "Neighbors (UMAP)",
        2, 100, 15, 1,
        help="This controls how many similar comments the app checks to spot patterns. Imagine looking at 15 nearby complaints to decide if 'fast delivery' is a trend. **Example:** Set to 15, and it groups comments by comparing them to 15 others. **Why it's different:** It shapes the big picture of feedback, unlike Min Distance, which only sets how far apart groups are."
    )
    n_components = st.sidebar.slider(
        "Components (UMAP)",
        2, 10, 5, 1,
        help="This chooses how many angles the app uses to sort feedback, like looking at tone and topic. **Example:** Set to 5, and it might sort 'angry refund comments' by urgency, product, and wording. **Why it's different:** It controls the map's depth, unlike Min Distance or Neighbors, which focus on spacing or comparisons."
    )
    min_dist = st.sidebar.slider(
        "Min Distance (UMAP)",
        0.0, 1.0, 0.1, 0.01,
        help="This sets how separate feedback groups stay. Think of it like keeping 'price issues' and 'service complaints' apart on a map. **Example:** A low value (0.1) keeps 'poor quality' distinct from 'high cost.' **Why it's different:** It adjusts spacing between groups, unlike Neighbors, which decides how many comments to compare for patterns."
    )
    min_cluster_size = st.sidebar.slider(
        "Min Cluster Size (HDBSCAN)",
        2, 500, 10, 1,
        help="This decides how many comments are needed to start a group when sorting begins. Think of it like needing at least 5 complaints to open a new category, such as 'rude staff.' **Example:** If set to 5, 4 comments about 'late orders' won't form a group yet. **Why it's different:** It works at the beginning to create groups, unlike Minimum Topic Size, which trims them later."
    )
    min_samples = st.sidebar.slider(
        "Min Samples (HDBSCAN)",
        1, 100, 5, 1,
        help="This sets how similar comments must be to form a group. Imagine needing 5 customers to say 'great support' almost the same way to make it a category. **Example:** If set to 5, scattered comments like 'nice staff' won't group unless they match closely. **Why it's different:** It's about consistency, not just size, unlike Min Cluster Size."
    )

    st.sidebar.header("Advanced Options")
    force_kmeans = st.sidebar.checkbox(
        "Force K-means Clustering",
        value=False,
        help="Always use K-means clustering regardless of dataset size. **Example:** Guarantees no outliers but may need to specify number of topics. **Why it's useful:** Ensures every document gets assigned to a topic."
    )

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

            # Create optimized topic model
            with st.spinner(f"Creating optimized topic model for {len(docs)} documents..."):
                # Show dataset size classification
                if len(docs) < 500:
                    st.info(f"📊 Small dataset detected ({len(docs)} docs) - Using K-means clustering for guaranteed topics")
                elif len(docs) < 5000:
                    st.info(f"📊 Medium dataset detected ({len(docs)} docs) - Using balanced parameters")
                else:
                    st.info(f"📊 Large dataset detected ({len(docs)} docs) - Using scalable configuration")

                model = create_bertopic_model(
                    positive_docs = [doc for doc, sentiment in zip(docs, sentiment_labels) if sentiment == 'positive']
                    negative_docs = [doc for doc, sentiment in zip(docs, sentiment_labels) if sentiment == 'negative']
                    neutral_docs = [doc for doc, sentiment in zip(docs, sentiment_labels) if sentiment == 'neutral']

                    st.info(f"Positive docs: {len(positive_docs)}, Negative docs: {len(negative_docs)}, Neutral docs: {len(neutral_docs)}")

                    # Calculate minimum required documents for sentiment separation
                    min_required = max(5, min_topic_size // 2)  # More lenient requirement

                    if len(positive_docs) < min_required or len(negative_docs) < min_required:
                        st.warning(f"⚠️ Not enough documents for sentiment separation. Need at least {min_required} docs per sentiment.")
                        st.info(f"Consider lowering the 'Minimum Topic Size' parameter to {min_required} or less.")
                        enable_sentiment = False
                        sentiment_labels = None
                    else:
                        # Show parameter adjustments that will be made
                        pos_topic_size = max(3, min(min_topic_size, len(positive_docs) // 3))
                        neg_topic_size = max(3, min(min_topic_size, len(negative_docs) // 3))
                        st.info(f"Auto-adjusting parameters: Positive min_topic_size={pos_topic_size}, Negative min_topic_size={neg_topic_size}")

            if enable_sentiment and sentiment_labels:
                # Step 2: Two-Stage Sequential Modeling (Proven 2024 Approach)
                st.subheader("🎭 Two-Stage Sentiment-Aware Topic Modeling")

                all_topics = []
                all_topic_labels = []
                topic_offset = 0

                # Model positive documents
                if len(positive_docs) >= max(5, min_topic_size // 2):
                    with st.spinner("Modeling positive sentiment topics..."):
                        # Adjust parameters for smaller sentiment-specific datasets
                        sentiment_min_topic_size = max(3, min(min_topic_size, len(positive_docs) // 3))
                        sentiment_min_cluster_size = max(2, min(min_cluster_size, len(positive_docs) // 4))

                        # Debug logging
                        st.write("🔍 **DEBUG: Positive Topic Modeling**")
                        st.write(f"- Documents: {len(positive_docs)}")
                        st.write(f"- Adjusted min_topic_size: {sentiment_min_topic_size}")
                        st.write(f"- Adjusted min_cluster_size: {sentiment_min_cluster_size}")
                        st.write(f"- Original min_topic_size: {min_topic_size}")
                        st.write(f"- Original min_cluster_size: {min_cluster_size}")
                        st.write(f"- Other params: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}")

                        # Show sample positive documents
                        with st.expander("📄 Sample Positive Documents"):
                            for i, doc in enumerate(positive_docs[:5]):
                                st.write(f"{i+1}. {doc[:100]}...")
                            if len(positive_docs) > 5:
                                st.write(f"... and {len(positive_docs) - 5} more")

                        pos_model = create_bertopic_model(
                            min_topic_size=sentiment_min_topic_size,
                            nr_topics=3,  # Force 3 topics for positive sentiment
                            n_neighbors=n_neighbors,
                            n_components=n_components,
                            min_dist=min_dist,
                            min_cluster_size=sentiment_min_cluster_size,
                            min_samples=min_samples,
                            seed_words=seed_words,
                            seed_multiplier=seed_multiplier,
                            use_kmeans=True,  # Use K-means for guaranteed topics
                            num_docs=len(positive_docs)
                        )
                        pos_topics, pos_probs = pos_model.fit_transform(positive_docs)

                        # Apply outlier reduction if using HDBSCAN and outliers exist
                        if pos_topics.count(-1) > 0:
                            st.write("🔄 Applying outlier reduction...")
                            pos_topics = pos_model.reduce_outliers(
                                positive_docs, pos_topics,
                                probabilities=pos_probs,
                                strategy="probabilities",
                                threshold=0.05
                            )
                            # Second pass with c-TF-IDF if still outliers
                            if pos_topics.count(-1) > 0:
                                pos_topics = pos_model.reduce_outliers(
                                    positive_docs, pos_topics,
                                    strategy="c-tf-idf"
                                )
                        pos_topic_info = pos_model.get_topic_info()

                        # Debug logging
                        st.write(f"- Raw topics found: {len(set(pos_topics))}")
                        st.write(f"- Unique topic IDs: {sorted(list(set(pos_topics)))}")
                        st.write(f"- Outliers (-1): {pos_topics.count(-1)}")
                        st.write(f"- Non-outlier topics: {len(set(pos_topics)) - (1 if -1 in pos_topics else 0)}")

                        # Show topic info details
                        if len(pos_topic_info) > 0:
                            st.write("- Topic info:")
                            st.dataframe(pos_topic_info[['Topic', 'Count', 'Name']], use_container_width=True)


                        # Adjust topic IDs and add sentiment prefix
                        adjusted_pos_topics = [t + topic_offset if t != -1 else -1 for t in pos_topics]
                        pos_labels = [f"😊 Positive: {name}" if topic_id != -1 else "😊 Positive Outliers"
                                     for topic_id, name in zip(pos_topic_info["Topic"], pos_topic_info["Name"])]

                        all_topics.extend(adjusted_pos_topics)
                        all_topic_labels.extend(pos_labels)
                        topic_offset += len([t for t in pos_topics if t != -1]) + 1

                        st.success(f"✅ Found {len(set(pos_topics)) - (1 if -1 in pos_topics else 0)} positive topics")

                # Model negative documents
                if len(negative_docs) >= max(5, min_topic_size // 2):
                    with st.spinner("Modeling negative sentiment topics..."):
                        # Adjust parameters for smaller sentiment-specific datasets
                        sentiment_min_topic_size = max(3, min(min_topic_size, len(negative_docs) // 3))
                        sentiment_min_cluster_size = max(2, min(min_cluster_size, len(negative_docs) // 4))

                        # Debug logging
                        st.write("🔍 **DEBUG: Negative Topic Modeling**")
                        st.write(f"- Documents: {len(negative_docs)}")
                        st.write(f"- Adjusted min_topic_size: {sentiment_min_topic_size}")
                        st.write(f"- Adjusted min_cluster_size: {sentiment_min_cluster_size}")

                        neg_model = create_bertopic_model(
                            min_topic_size=sentiment_min_topic_size,
                            nr_topics=3,  # Force 3 topics for negative sentiment
                            n_neighbors=n_neighbors,
                            n_components=n_components,
                            min_dist=min_dist,
                            min_cluster_size=sentiment_min_cluster_size,
                            min_samples=min_samples,
                            seed_words=seed_words,
                            seed_multiplier=seed_multiplier,
                            use_kmeans=True,  # Use K-means for guaranteed topics
                            num_docs=len(negative_docs)
                        )
                        neg_topics, neg_probs = neg_model.fit_transform(negative_docs)

                        # Apply outlier reduction if using HDBSCAN and outliers exist
                        if neg_topics.count(-1) > 0:
                            st.write("🔄 Applying outlier reduction...")
                            neg_topics = neg_model.reduce_outliers(
                                negative_docs, neg_topics,
                                probabilities=neg_probs,
                                strategy="probabilities",
                                threshold=0.05
                            )
                            # Second pass with c-TF-IDF if still outliers
                            if neg_topics.count(-1) > 0:
                                neg_topics = neg_model.reduce_outliers(
                                    negative_docs, neg_topics,
                                    strategy="c-tf-idf"
                                )
                        neg_topic_info = neg_model.get_topic_info()

                        # Debug logging
                        st.write(f"- Raw topics found: {len(set(neg_topics))}")
                        st.write(f"- Unique topic IDs: {sorted(list(set(neg_topics)))}")
                        st.write(f"- Outliers (-1): {neg_topics.count(-1)}")
                        st.write(f"- Non-outlier topics: {len(set(neg_topics)) - (1 if -1 in neg_topics else 0)}")

                        # Show topic info details
                        if len(neg_topic_info) > 0:
                            st.write("- Topic info:")
                            st.dataframe(neg_topic_info[['Topic', 'Count', 'Name']], use_container_width=True)


                        # Adjust topic IDs and add sentiment prefix
                        adjusted_neg_topics = [t + topic_offset if t != -1 else -1 for t in neg_topics]
                        neg_labels = [f"😞 Negative: {name}" if topic_id != -1 else "😞 Negative Outliers"
                                     for topic_id, name in zip(neg_topic_info["Topic"], neg_topic_info["Name"])]

                        all_topics.extend(adjusted_neg_topics)
                        all_topic_labels.extend(neg_labels)
                        topic_offset += len(set(neg_topics)) - (1 if -1 in neg_topics else 0) + 1

                        st.success(f"✅ Found {len(set(neg_topics)) - (1 if -1 in neg_topics else 0)} negative topics")

                # Handle neutral documents if any
                if len(neutral_docs) >= max(5, min_topic_size // 2):
                    with st.spinner("Modeling neutral sentiment topics..."):
                        # Adjust parameters for smaller sentiment-specific datasets
                        sentiment_min_topic_size = max(3, min(min_topic_size, len(neutral_docs) // 3))
                        sentiment_min_cluster_size = max(2, min(min_cluster_size, len(neutral_docs) // 4))

                        neutral_model = create_bertopic_model(
                            min_topic_size=sentiment_min_topic_size,
                            nr_topics=nr_topics,
                            n_neighbors=n_neighbors,
                            n_components=n_components,
                            min_dist=min_dist,
                            min_cluster_size=sentiment_min_cluster_size,
                            min_samples=min_samples,
                            seed_words=seed_words,
                            seed_multiplier=seed_multiplier
                        )
                        neutral_topics, _ = neutral_model.fit_transform(neutral_docs)
                        neutral_topic_info = neutral_model.get_topic_info()

                        # Adjust topic IDs and add sentiment prefix
                        adjusted_neutral_topics = [t + topic_offset if t != -1 else -1 for t in neutral_topics]
                        neutral_labels = [f"😐 Neutral: {name}" if topic_id != -1 else "😐 Neutral Outliers"
                                         for topic_id, name in zip(neutral_topic_info["Topic"], neutral_topic_info["Name"])]

                        all_topics.extend(adjusted_neutral_topics)
                        all_topic_labels.extend(neutral_labels)

                        st.success(f"✅ Found {len(set(neutral_topics)) - (1 if -1 in neutral_topics else 0)} neutral topics")
                else:
                    # Add neutral docs as outliers
                    all_topics.extend([-1] * len(neutral_docs))
                    all_topic_labels.extend(["😐 Neutral Outliers"] * len(neutral_docs))

                # Reconstruct full results by mapping back to original document order
                topics = []
                topic_labels = []

                pos_idx = 0
                neg_idx = 0
                neutral_idx = 0

                for doc, sentiment in zip(docs, sentiment_labels):
                    if sentiment == 'positive' and len(positive_docs) >= min_topic_size:
                        if pos_idx < len(pos_topics):
                            topics.append(adjusted_pos_topics[pos_idx])
                            topic_labels.append(pos_labels[pos_topics[pos_idx]] if pos_topics[pos_idx] != -1 else "😊 Positive Outliers")
                        else:
                            topics.append(-1)
                            topic_labels.append("😊 Positive Outliers")
                        pos_idx += 1
                    elif sentiment == 'negative' and len(negative_docs) >= min_topic_size:
                        if neg_idx < len(neg_topics):
                            topics.append(adjusted_neg_topics[neg_idx])
                            topic_labels.append(neg_labels[neg_topics[neg_idx]] if neg_topics[neg_idx] != -1 else "😞 Negative Outliers")
                        else:
                            topics.append(-1)
                            topic_labels.append("😞 Negative Outliers")
                        neg_idx += 1
                    elif sentiment == 'neutral' and len(neutral_docs) >= min_topic_size:
                        if neutral_idx < len(neutral_topics):
                            topics.append(adjusted_neutral_topics[neutral_idx])
                            topic_labels.append(neutral_labels[neutral_topics[neutral_idx]] if neutral_topics[neutral_idx] != -1 else "😐 Neutral Outliers")
                        else:
                            topics.append(-1)
                            topic_labels.append("😐 Neutral Outliers")
                        neutral_idx += 1
                    else:
                        # Fallback for insufficient docs or neutral
                        topics.append(-1)
                        if sentiment == 'positive':
                            topic_labels.append("😊 Positive Outliers")
                        elif sentiment == 'negative':
                            topic_labels.append("😞 Negative Outliers")
                        else:
                            topic_labels.append("😐 Neutral Outliers")

            else:
                # Step 2: Standard Topic Modeling (No Sentiment Separation)
                with st.spinner(f"Creating optimized topic model for {len(docs)} documents..."):
                    # Show dataset size classification
                    if len(docs) < 500:
                        st.info(f"📊 Small dataset detected ({len(docs)} docs) - Using K-means clustering for guaranteed topics")
                    elif len(docs) < 5000:
                        st.info(f"📊 Medium dataset detected ({len(docs)} docs) - Using balanced parameters")
                    else:
                        st.info(f"📊 Large dataset detected ({len(docs)} docs) - Using scalable configuration")

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
                        num_docs=len(docs)
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

            st.success("Topic modeling complete!")

            # Step 4: Process Results
            df["topic_id"] = topics
            if enable_sentiment and sentiment_labels:
                df["topic_label"] = topic_labels
            else:
                df["topic_label"] = topic_labels

            # Add sentiment data if available
            if sentiment_labels:
                df["sentiment"] = sentiment_labels

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

            if enable_sentiment and sentiment_labels:
                st.success("✅ Topics successfully separated by sentiment!")

                # Show sentiment-based topic breakdown
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 😊 Positive Topics")
                    pos_topics = df[df["topic_label"].str.contains("😊", na=False)]["topic_label"].value_counts()
                    if len(pos_topics) > 0:
                        st.write(pos_topics)
                    else:
                        st.write("No positive topics found")

                with col2:
                    st.markdown("### 😞 Negative Topics")
                    neg_topics = df[df["topic_label"].str.contains("😞", na=False)]["topic_label"].value_counts()
                    if len(neg_topics) > 0:
                        st.write(neg_topics)
                    else:
                        st.write("No negative topics found")

                st.markdown("### 😐 Neutral/Mixed Topics")
                neutral_topics = df[~df["topic_label"].str.contains("😊|😞", na=False)]["topic_label"].value_counts()
                if len(neutral_topics) > 0:
                    st.write(neutral_topics)
                else:
                    st.write("No neutral topics found")
            else:
                st.subheader("Topic Counts")
                counts = df["topic_label"].value_counts()
                st.write(counts)

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

                # Create a clean display
                display_df = topic_docs[[text_col]].copy()
                display_df.columns = ['Document']

                # Show documents in a scrollable container
                html_table = display_df.to_html(index=False, border=0, escape=False)
                html = f'''
                <div style="width:100%; height:400px; overflow-y:scroll;
                            border: 1px solid #e0e0e0; border-radius:8px;
                            padding:10px; background-color: #f9f9f9;">
                    {html_table}
                </div>
                '''
                st.markdown(html, unsafe_allow_html=True)

            # Option to view all topics at once
            if st.checkbox("Show all topics at once"):
                for topic in sorted(df["topic_label"].unique()):
                    with st.expander(f"{topic} ({len(df[df['topic_label']==topic])} documents)"):
                        topic_docs = df[df["topic_label"] == topic]
                        st.write(topic_docs[[text_col]].to_dict('records'))

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