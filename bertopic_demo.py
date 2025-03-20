import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px

# Set Streamlit to wide mode for better layout
st.set_page_config(layout="wide")

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
    seed_multiplier=1.0
):
    """
    Creates a BERTopic model to cluster documents into topics.
    Optionally boosts specific words (seed_words).
    Uses default single-word tokenization.
    """
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1,1))

    if seed_words:
        ctfidf_model = ClassTfidfTransformer(
            seed_words=seed_words,
            seed_multiplier=seed_multiplier
        )
        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
        )
    else:
        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
        )

    return model

# -----------------------------------------------------
# 2. CONVERT DF TO CSV
# -----------------------------------------------------
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------------------------------
# 3. MAIN STREAMLIT APP
# -----------------------------------------------------
def main():
    st.title("BERTopic with Seed Words & Topic Grouping")
    st.write("This app clusters your free-text data into topics. You can boost important words (seed words) and view all documents for each topic.")

    # --- SIDEBAR ---
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    st.sidebar.header("Topic Settings")
    min_topic_size = st.sidebar.slider(
        "Minimum Topic Size",
        1, 500, 10, 1,
        help="This sets the smallest number of customer comments needed to keep a topic. Imagine sorting feedback into piles like 'slow delivery.' If you set this to 10, only piles with 10 or more comments stay—smaller ones get mixed into bigger piles. **Example:** A group with 3 comments about 'bad packaging' gets merged if this is 10. **Why it’s different:** It cleans up topics at the end, unlike Min Cluster Size, which sets group sizes at the start."
    )
    nr_topics = st.sidebar.selectbox(
        "Number of Topics",
        [None, "auto", 5, 10, 15, 20],
        index=0,
        help="This picks how many main topics you get, like choosing to focus on 5 big customer issues. **Example:** Set it to 5, and you might get 'pricing,' 'quality,' 'shipping,' 'service,' and 'returns.' Pick 'auto' to let the app decide. **Why it’s different:** It locks in the exact number of topics, unlike other settings that just adjust how groups form or merge."
    )
    n_neighbors = st.sidebar.slider(
        "Neighbors (UMAP)",
        2, 100, 15, 1,
        help="This controls how many similar comments the app checks to spot patterns. Imagine looking at 15 nearby complaints to decide if 'fast delivery' is a trend. **Example:** Set to 15, and it groups comments by comparing them to 15 others. **Why it’s different:** It shapes the big picture of feedback, unlike Min Distance, which only sets how far apart groups are."
    )
    n_components = st.sidebar.slider(
        "Components (UMAP)",
        2, 10, 5, 1,
        help="This chooses how many angles the app uses to sort feedback, like looking at tone and topic. **Example:** Set to 5, and it might sort 'angry refund comments' by urgency, product, and wording. **Why it’s different:** It controls the map’s depth, unlike Min Distance or Neighbors, which focus on spacing or comparisons."
    )
    min_dist = st.sidebar.slider(
        "Min Distance (UMAP)",
        0.0, 1.0, 0.1, 0.01,
        help="This sets how separate feedback groups stay. Think of it like keeping 'price issues' and 'service complaints' apart on a map. **Example:** A low value (0.1) keeps 'poor quality' distinct from 'high cost.' **Why it’s different:** It adjusts spacing between groups, unlike Neighbors, which decides how many comments to compare for patterns."
    )
    min_cluster_size = st.sidebar.slider(
        "Min Cluster Size (HDBSCAN)",
        2, 500, 10, 1,
        help="This decides how many comments are needed to start a group when sorting begins. Think of it like needing at least 5 complaints to open a new category, such as 'rude staff.' **Example:** If set to 5, 4 comments about 'late orders' won’t form a group yet. **Why it’s different:** It works at the beginning to create groups, unlike Minimum Topic Size, which trims them later."
    )
    min_samples = st.sidebar.slider(
        "Min Samples (HDBSCAN)",
        1, 100, 5, 1,
        help="This sets how similar comments must be to form a group. Imagine needing 5 customers to say 'great support' almost the same way to make it a category. **Example:** If set to 5, scattered comments like 'nice staff' won’t group unless they match closely. **Why it’s different:** It’s about consistency, not just size, unlike Min Cluster Size."
    )

    st.sidebar.header("Seed Words (Optional)")
    st.sidebar.write("Boost important words by listing them here (comma-separated).")
    seed_words_str = st.sidebar.text_input(
        "Seed Words",
        "",
        help="Add key words like 'price, delivery, support' to guide the app. It’s like telling it to spotlight comments about shipping. **Example:** Type 'refund,' and refund-related feedback gets more focus. **Why it’s different:** It steers the topics you care about, unlike settings that tweak grouping mechanics."
    )
    seed_words = [w.strip() for w in seed_words_str.split(",") if w.strip()] if seed_words_str else None
    seed_multiplier = 1.0
    if seed_words:
        seed_multiplier = st.sidebar.slider(
            "Seed Word Boost",
            1.0, 5.0, 2.0, 0.5,
            help="This makes your seed words more important. Think of it as turning up the volume on 'delivery' complaints. **Example:** Set to 3.0, and 'support' in feedback gets triple the attention. **Why it’s different:** It only boosts your chosen words, not the whole process like other settings."
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
            with st.spinner("Running BERTopic..."):
                model = create_bertopic_model(
                    min_topic_size=min_topic_size,
                    nr_topics=nr_topics,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    seed_words=seed_words,
                    seed_multiplier=seed_multiplier
                )
                docs = df[text_col].astype(str).tolist()
                topics, _ = model.fit_transform(docs)

            st.success("Topic modeling complete!")

            # Insert topic assignments into the DataFrame
            df["topic_id"] = topics
            topic_info = model.get_topic_info()
            label_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
            df["topic_label"] = df["topic_id"].map(label_map).fillna("Unknown")

            st.subheader("Topic Counts")
            counts = df["topic_label"].value_counts()
            st.write(counts)

            st.subheader("Documents Grouped by Topic")
            # For each unique topic, display a scrollable table of the free text documents.
            for topic in sorted(df["topic_label"].unique()):
                st.markdown(f"### {topic}")
                topic_docs = df[df["topic_label"] == topic][[text_col]]
                # Convert DataFrame to HTML without index and wrap in a div with inline styles for vertical scrolling.
                html_table = topic_docs.to_html(index=False, border=0)
                html = f'<div style="width:100%; height:400px; overflow-y:scroll; border: 1px solid #e0e0e0; border-radius:8px; padding:10px;">{html_table}</div>'
                st.markdown(html, unsafe_allow_html=True)

            st.subheader("Download Final Results")
            csv_bytes = convert_df_to_csv(df)
            st.download_button(
                label="Download CSV (with topic assignments)",
                data=csv_bytes,
                file_name="bertopic_topic_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Please upload a CSV file in the sidebar to begin.")

if __name__ == "__main__":
    main()
