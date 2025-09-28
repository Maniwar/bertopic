#!/usr/bin/env python3
"""
Pure BERTopic Implementation 2025 - Zero Custom Logic
Uses ONLY BERTopic's built-in capabilities for topic discovery and labeling
No hardcoded patterns, semantic assumptions, or custom categorization logic
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging

# BERTopic pure imports - using ONLY library features
from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    ZeroShotClassification,
    PartOfSpeech
)
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    layout="wide",
    page_title="Pure BERTopic 2025",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

class PureBERTopicService:
    """Pure BERTopic implementation using ONLY library-native capabilities"""

    def __init__(self):
        self.model = None
        self.topic_info = None
        self.hierarchical_topics = None

    def create_pure_bertopic_model(
        self,
        documents: List[str],
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None,
        representation_strategy: str = "keybert_mmr",
        zero_shot_topics: Optional[List[str]] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_state: int = 42
    ) -> BERTopic:
        """Create pure BERTopic model using only library features"""

        logger.info(f"Creating pure BERTopic model for {len(documents)} documents")

        # Pure UMAP configuration (library defaults with minimal tuning)
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=random_state
        )

        # Pure HDBSCAN clustering (library defaults)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Pure CountVectorizer (minimal configuration)
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )

        # Pure representation models - ONLY library implementations
        representation_model = self._get_pure_representation_model(representation_strategy, zero_shot_topics)

        # Create pure BERTopic model
        if zero_shot_topics:
            model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                zeroshot_topic_list=zero_shot_topics,
                zeroshot_min_similarity=0.7,
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                verbose=True
            )
        else:
            model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                verbose=True
            )

        return model

    def _get_pure_representation_model(self, strategy: str, zero_shot_topics: Optional[List[str]] = None):
        """Get pure representation model using ONLY BERTopic implementations"""

        if strategy == "keybert_only":
            return KeyBERTInspired(top_n_words=10, nr_repr_docs=3)

        elif strategy == "mmr_only":
            return MaximalMarginalRelevance(diversity=0.3)

        elif strategy == "pos_only":
            return PartOfSpeech("en_core_web_sm")

        elif strategy == "zero_shot" and zero_shot_topics:
            return ZeroShotClassification(
                zero_shot_topics,
                model="facebook/bart-large-mnli"
            )

        elif strategy == "keybert_mmr":
            # Chain KeyBERT + MMR for optimal keyword diversity
            return [
                KeyBERTInspired(top_n_words=15, nr_repr_docs=5),
                MaximalMarginalRelevance(diversity=0.3)
            ]

        elif strategy == "multi_aspect":
            # Multiple representation aspects
            return {
                "KeyBERT": KeyBERTInspired(top_n_words=10),
                "MMR": MaximalMarginalRelevance(diversity=0.3),
                "POS": PartOfSpeech("en_core_web_sm")
            }

        else:
            # Default: KeyBERT inspired
            return KeyBERTInspired(top_n_words=10)

    def discover_topics_pure(self, documents: List[str], model: BERTopic) -> Dict[str, Any]:
        """Pure topic discovery using ONLY BERTopic capabilities"""

        logger.info(f"Starting pure topic discovery for {len(documents)} documents")

        # Core BERTopic discovery - no custom logic
        topics, probabilities = model.fit_transform(documents)
        topic_info = model.get_topic_info()

        # BERTopic's built-in outlier reduction
        outlier_ratio = sum(1 for t in topics if t == -1) / len(topics)
        if outlier_ratio > 0.3:
            logger.info(f"Using BERTopic's built-in outlier reduction: {outlier_ratio:.1%}")
            topics = model.reduce_outliers(documents, topics, strategy="embeddings")
            topic_info = model.get_topic_info()

        # BERTopic's native hierarchical topics
        hierarchical_topics = None
        try:
            hierarchical_topics = model.hierarchical_topics(documents)
            logger.info("BERTopic hierarchical structure created")
        except Exception as e:
            logger.warning(f"Hierarchical generation failed: {e}")

        # Store model
        self.model = model
        self.topic_info = topic_info
        self.hierarchical_topics = hierarchical_topics

        return {
            'topics': topics,
            'probabilities': probabilities,
            'topic_info': topic_info,
            'hierarchical_topics': hierarchical_topics,
            'model': model
        }

    def get_pure_topic_labels(self, topic_id: int) -> Dict[str, Any]:
        """Get topic labels using ONLY BERTopic's native methods"""

        if self.model is None:
            return {'label': f'Topic {topic_id}', 'keywords': [], 'representation_source': 'none'}

        # Get standard BERTopic topic representation
        topic_words = self.model.get_topic(topic_id)
        default_keywords = [word for word, score in topic_words[:8]] if topic_words else []

        # Try enhanced representation if available
        enhanced_keywords = []
        representation_source = 'c_tf_idf'

        if hasattr(self.model, 'topic_aspects_') and self.model.topic_aspects_:
            aspects = self.model.topic_aspects_

            # Priority: MMR > KeyBERT > POS
            if topic_id in aspects.get('MMR', {}):
                mmr_results = aspects['MMR'][topic_id]
                enhanced_keywords = [word[0] if isinstance(word, tuple) else str(word) for word in mmr_results[:8]]
                representation_source = 'mmr'

            elif topic_id in aspects.get('MaximalMarginalRelevance', {}):
                mmr_results = aspects['MaximalMarginalRelevance'][topic_id]
                enhanced_keywords = [word[0] if isinstance(word, tuple) else str(word) for word in mmr_results[:8]]
                representation_source = 'mmr'

            elif topic_id in aspects.get('KeyBERT', {}):
                keybert_results = aspects['KeyBERT'][topic_id]
                enhanced_keywords = [word[0] if isinstance(word, tuple) else str(word) for word in keybert_results[:8]]
                representation_source = 'keybert'

            elif topic_id in aspects.get('KeyBERTInspired', {}):
                keybert_results = aspects['KeyBERTInspired'][topic_id]
                enhanced_keywords = [word[0] if isinstance(word, tuple) else str(word) for word in keybert_results[:8]]
                representation_source = 'keybert'

            elif topic_id in aspects.get('POS', {}):
                pos_results = aspects['POS'][topic_id]
                enhanced_keywords = [word[0] if isinstance(word, tuple) else str(word) for word in pos_results[:8]]
                representation_source = 'pos'

        # Use enhanced if available, otherwise default
        final_keywords = enhanced_keywords if enhanced_keywords else default_keywords

        # Generate simple label from keywords (no custom semantic logic)
        if final_keywords:
            label = f"Topic: {', '.join(final_keywords[:3])}"
        else:
            label = f"Topic {topic_id}"

        return {
            'label': label,
            'keywords': final_keywords,
            'representation_source': representation_source,
            'topic_id': topic_id
        }

    def get_hierarchical_structure(self) -> Dict[str, Any]:
        """Get hierarchical structure using ONLY BERTopic's hierarchical_topics"""

        if self.hierarchical_topics is None:
            return {'available': False}

        # Pure hierarchical analysis - no custom interpretation
        hierarchy_info = {
            'available': True,
            'hierarchy_df': self.hierarchical_topics,
            'merge_count': len(self.hierarchical_topics),
            'topic_tree': None
        }

        # Try to get BERTopic's topic tree if available
        try:
            if hasattr(self.model, 'get_topic_tree'):
                hierarchy_info['topic_tree'] = self.model.get_topic_tree(self.hierarchical_topics)
        except Exception as e:
            logger.warning(f"Topic tree generation failed: {e}")

        return hierarchy_info

    def create_subtopic_analysis_pure(self, documents: List[str], topic_id: int) -> Dict[str, Any]:
        """Create sub-topic analysis using pure BERTopic recursion"""

        if self.model is None:
            return {'error': 'No model available'}

        # Get documents for this topic
        topics = self.model.topics_
        topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]

        if len(topic_docs) < 8:
            return {'error': 'Insufficient documents for sub-analysis', 'doc_count': len(topic_docs)}

        try:
            # Create new pure BERTopic model for sub-topics
            sub_model = BERTopic(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                min_topic_size=max(3, len(topic_docs) // 8),
                representation_model=KeyBERTInspired(top_n_words=8),
                verbose=False
            )

            # Pure sub-topic discovery
            sub_topics, sub_probabilities = sub_model.fit_transform(topic_docs)
            sub_topic_info = sub_model.get_topic_info()

            return {
                'sub_topics': sub_topics,
                'sub_topic_info': sub_topic_info,
                'sub_model': sub_model,
                'original_docs': topic_docs,
                'parent_topic_id': topic_id,
                'success': True
            }

        except Exception as e:
            logger.error(f"Pure sub-topic analysis failed: {e}")
            return {'error': str(e), 'success': False}

class PureBERTopicExplorer:
    """Pure exploration interface using only BERTopic results"""

    def __init__(self, service: PureBERTopicService):
        self.service = service

    def render_pure_results(self, discovery_results: Dict[str, Any], documents: List[str]):
        """Render results using ONLY BERTopic's discoveries"""

        st.header("🔬 Pure BERTopic Discovery Results")
        st.markdown("**Zero custom logic - only BERTopic's native capabilities**")

        topics = discovery_results['topics']
        topic_info = discovery_results['topic_info']

        # Pure metrics
        valid_topics = len(topic_info[topic_info['Topic'] != -1])
        outlier_count = sum(1 for t in topics if t == -1)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BERTopic Discovered", valid_topics)
        with col2:
            st.metric("Documents Processed", len(documents))
        with col3:
            st.metric("Outliers (BERTopic)", outlier_count)

        if valid_topics == 0:
            st.warning("BERTopic found no topics. Try different parameters or more diverse data.")
            return

        # Topic explorer
        self.render_pure_topic_explorer(topics, topic_info, documents)

    def render_pure_topic_explorer(self, topics: List[int], topic_info: pd.DataFrame, documents: List[str]):
        """Pure topic exploration using only BERTopic features"""

        # Get pure topic labels
        topic_labels = {}
        valid_topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()

        for topic_id in valid_topic_ids:
            topic_labels[topic_id] = self.service.get_pure_topic_labels(topic_id)

        if not topic_labels:
            st.warning("No valid topics found")
            return

        # Topic selection
        col1, col2 = st.columns(2)

        with col1:
            selected_topic_id = st.selectbox(
                "Select BERTopic Discovery:",
                options=list(topic_labels.keys()),
                format_func=lambda x: f"Topic {x}: {topic_labels[x]['label']}"
            )

        with col2:
            view_mode = st.selectbox(
                "Exploration Mode:",
                ["📋 Overview", "🔍 Details", "🌳 Hierarchical", "🎯 Sub-Topics"]
            )

        # Display based on mode
        selected_info = topic_labels[selected_topic_id]

        if view_mode == "📋 Overview":
            self.render_topic_overview(selected_topic_id, selected_info, topics, documents)
        elif view_mode == "🔍 Details":
            self.render_topic_details(selected_topic_id, selected_info, topics, documents)
        elif view_mode == "🌳 Hierarchical":
            self.render_hierarchical_view()
        elif view_mode == "🎯 Sub-Topics":
            self.render_subtopic_view(selected_topic_id, topics, documents)

    def render_topic_overview(self, topic_id: int, topic_info: Dict, topics: List[int], documents: List[str]):
        """Pure topic overview - no custom analysis"""

        st.markdown(f"### 📋 {topic_info['label']}")

        # Get topic documents
        topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]

        # Display pure BERTopic information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔑 BERTopic Keywords:**")
            for i, keyword in enumerate(topic_info['keywords'][:8], 1):
                st.write(f"{i}. **{keyword}**")

            st.markdown(f"**📊 Representation Source:** {topic_info['representation_source']}")

        with col2:
            st.metric("Documents in Topic", len(topic_docs))
            st.metric("Topic ID", topic_id)

        # Show sample documents
        st.markdown(f"**📑 Sample Documents (showing {min(10, len(topic_docs))} of {len(topic_docs)})**")

        for i, doc in enumerate(topic_docs[:10], 1):
            st.write(f"**{i}.** {doc}")

    def render_topic_details(self, topic_id: int, topic_info: Dict, topics: List[int], documents: List[str]):
        """Detailed topic view using pure BERTopic data"""

        st.markdown(f"### 🔍 Detailed Analysis: {topic_info['label']}")

        # Get topic documents
        topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]

        # BERTopic topic representation
        if self.service.model:
            topic_words = self.service.model.get_topic(topic_id)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🎯 BERTopic Word Scores:**")
                for word, score in topic_words[:10]:
                    st.write(f"• **{word}**: {score:.4f}")

            with col2:
                st.markdown("**📈 Topic Statistics:**")
                st.metric("Total Documents", len(topic_docs))
                st.metric("Keyword Count", len(topic_info['keywords']))
                st.metric("Representation Method", topic_info['representation_source'])

        # All documents for this topic
        st.markdown(f"**📚 All Documents in Topic {topic_id}**")
        for i, doc in enumerate(topic_docs, 1):
            with st.expander(f"Document {i}"):
                st.write(doc)

    def render_hierarchical_view(self):
        """Show BERTopic's hierarchical analysis"""

        hierarchy = self.service.get_hierarchical_structure()

        if not hierarchy['available']:
            st.warning("Hierarchical analysis not available. Try with more topics.")
            return

        st.markdown("### 🌳 BERTopic Hierarchical Structure")
        st.markdown("**Pure BERTopic hierarchical_topics() output**")

        # Show hierarchical dataframe
        if 'hierarchy_df' in hierarchy:
            st.markdown("**📊 Hierarchy Data:**")
            st.dataframe(hierarchy['hierarchy_df'], use_container_width=True)

        # Show topic tree if available
        if hierarchy.get('topic_tree'):
            st.markdown("**🌲 Topic Tree:**")
            st.text(hierarchy['topic_tree'])

        # BERTopic visualization if available
        if self.service.model and hasattr(self.service.model, 'visualize_hierarchy'):
            try:
                st.markdown("**📈 BERTopic Hierarchy Visualization:**")
                fig = self.service.model.visualize_hierarchy(hierarchical_topics=self.service.hierarchical_topics)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Visualization not available: {e}")

    def render_subtopic_view(self, topic_id: int, topics: List[int], documents: List[str]):
        """Pure sub-topic analysis using BERTopic recursion"""

        st.markdown(f"### 🎯 Sub-Topic Analysis for Topic {topic_id}")
        st.markdown("**Using pure BERTopic recursion - no custom logic**")

        # Get topic documents count
        topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]

        if len(topic_docs) < 8:
            st.warning(f"Need at least 8 documents for sub-analysis. Topic {topic_id} has {len(topic_docs)} documents.")
            return

        with st.spinner("🔬 Running pure BERTopic sub-analysis..."):
            sub_analysis = self.service.create_subtopic_analysis_pure(documents, topic_id)

        if not sub_analysis.get('success', False):
            st.error(f"Sub-analysis failed: {sub_analysis.get('error', 'Unknown error')}")
            return

        # Display sub-topic results
        sub_topic_info = sub_analysis['sub_topic_info']
        sub_topics = sub_analysis['sub_topics']
        sub_docs = sub_analysis['original_docs']
        sub_model = sub_analysis['sub_model']

        valid_sub_topics = len(sub_topic_info[sub_topic_info['Topic'] != -1])
        st.success(f"🎉 BERTopic discovered {valid_sub_topics} sub-topics!")

        # Show sub-topics
        for _, row in sub_topic_info.iterrows():
            sub_topic_id = row['Topic']
            if sub_topic_id == -1:
                continue

            sub_topic_docs = [doc for doc, t in zip(sub_docs, sub_topics) if t == sub_topic_id]

            # Get sub-topic words
            try:
                sub_topic_words = sub_model.get_topic(sub_topic_id)
                sub_keywords = [word for word, score in sub_topic_words[:5]] if sub_topic_words else []
            except:
                sub_keywords = [f"subtopic_{sub_topic_id}"]

            with st.expander(f"Sub-Topic {sub_topic_id}: {', '.join(sub_keywords)} ({len(sub_topic_docs)} docs)"):
                st.markdown("**🔑 Keywords:**")
                for keyword in sub_keywords:
                    st.write(f"• {keyword}")

                st.markdown("**📑 Sample Documents:**")
                for i, doc in enumerate(sub_topic_docs[:5], 1):
                    st.write(f"{i}. {doc}")

def main():
    """Main application for pure BERTopic discovery"""

    st.title("🔬 Pure BERTopic 2025")
    st.markdown("**Zero custom logic - only BERTopic's native capabilities**")

    # Initialize service
    service = PureBERTopicService()
    explorer = PureBERTopicExplorer(service)

    # Initialize session state
    if 'pure_analysis_complete' not in st.session_state:
        st.session_state.pure_analysis_complete = False

    # Sidebar configuration
    render_pure_sidebar()

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        process_pure_pipeline(uploaded_file, service, explorer)
    else:
        show_pure_welcome()

    # Results
    if st.session_state.pure_analysis_complete and 'pure_results' in st.session_state:
        results = st.session_state.pure_results
        explorer.render_pure_results(results['discovery_results'], results['documents'])
        render_pure_export(results)

def render_pure_sidebar():
    """Pure configuration sidebar"""

    st.sidebar.header("🔬 Pure BERTopic Settings")

    # Representation strategy
    st.session_state.representation_strategy = st.sidebar.selectbox(
        "Representation Model:",
        [
            "keybert_mmr",      # KeyBERT + MMR chain
            "keybert_only",     # KeyBERT only
            "mmr_only",         # MMR only
            "pos_only",         # Part of Speech only
            "multi_aspect",     # Multiple aspects
            "zero_shot"         # Zero-shot classification
        ],
        help="Choose BERTopic's built-in representation models"
    )

    # Zero-shot topics (if selected)
    if st.session_state.representation_strategy == "zero_shot":
        st.session_state.zero_shot_topics = st.sidebar.text_area(
            "Zero-shot Topics (one per line):",
            value="Technology\nBusiness\nEducation\nHealth\nPolitics",
            help="Predefined topics for zero-shot classification"
        ).strip().split('\n')
    else:
        st.session_state.zero_shot_topics = None

    # Core parameters
    with st.sidebar.expander("📊 Core Parameters"):
        st.session_state.min_topic_size = st.sidebar.slider("Min Topic Size", 3, 20, 8)
        st.session_state.nr_topics = st.sidebar.selectbox("Target Topics", [None, 5, 10, 15, 20, 25])

    # Embedding model
    st.session_state.embedding_model = st.sidebar.selectbox(
        "Embedding Model:",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
    )

def process_pure_pipeline(uploaded_file, service: PureBERTopicService, explorer: PureBERTopicExplorer):
    """Process file through pure BERTopic pipeline"""

    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows")
        st.dataframe(df.head(3), use_container_width=True)

        # Text column selection
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        if not text_cols:
            st.error("No text columns found")
            return

        text_col = st.selectbox("Select text column:", text_cols)

        if st.button("🔬 Pure BERTopic Discovery", type="primary"):
            run_pure_analysis(df, text_col, service)

    except Exception as e:
        st.error(f"Processing failed: {e}")

def run_pure_analysis(df: pd.DataFrame, text_col: str, service: PureBERTopicService):
    """Run pure BERTopic analysis"""

    try:
        # Prepare documents
        documents = df[text_col].dropna().astype(str).tolist()
        documents = [doc.strip() for doc in documents if len(doc.strip()) > 10]

        if len(documents) < 10:
            st.error(f"Need at least 10 documents, found {len(documents)}")
            return

        # Progress tracking
        progress = st.progress(0)
        status = st.empty()

        # Get parameters
        params = {
            'min_topic_size': st.session_state.get('min_topic_size', 8),
            'nr_topics': st.session_state.get('nr_topics', None),
            'representation_strategy': st.session_state.get('representation_strategy', 'keybert_mmr'),
            'zero_shot_topics': st.session_state.get('zero_shot_topics', None),
            'embedding_model': st.session_state.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        }

        # Create model
        status.text("🏗️ Creating pure BERTopic model...")
        progress.progress(25)

        model = service.create_pure_bertopic_model(documents, **params)

        # Discover topics
        status.text("🔬 Pure topic discovery...")
        progress.progress(60)

        discovery_results = service.discover_topics_pure(documents, model)

        status.text("✅ Pure discovery complete!")
        progress.progress(100)

        # Store results
        st.session_state.pure_results = {
            'discovery_results': discovery_results,
            'documents': documents
        }
        st.session_state.pure_analysis_complete = True

        # Show summary
        valid_topics = len(discovery_results['topic_info'][discovery_results['topic_info']['Topic'] != -1])
        st.success(f"🎉 Pure BERTopic discovered {valid_topics} topics!")

    except Exception as e:
        st.error(f"Pure discovery failed: {e}")

        with st.expander("🐛 Error Details"):
            st.write(f"**Error:** {str(e)}")
            st.write(f"**Document Count:** {len(documents) if 'documents' in locals() else 'Unknown'}")
            if 'params' in locals():
                st.json(params)

def show_pure_welcome():
    """Show welcome screen for pure BERTopic"""

    st.info("👋 Upload CSV for pure BERTopic discovery")

    st.markdown("""
    ### 🔬 Pure BERTopic Features:
    - **Zero custom logic** - only BERTopic's native capabilities
    - **Multiple representation models** - KeyBERT, MMR, POS, Zero-shot
    - **Hierarchical topics** - using BERTopic's hierarchical_topics()
    - **Sub-topic discovery** - pure BERTopic recursion
    - **No hardcoded patterns** - completely data-driven
    """)

def render_pure_export(results):
    """Export pure results"""

    st.header("💾 Export Pure Discovery")

    documents = results['documents']
    discovery_results = results['discovery_results']

    # Create export dataframe
    export_df = pd.DataFrame({
        'document': documents,
        'bertopic_topic_id': discovery_results['topics'],
        'bertopic_probability': discovery_results.get('probabilities', [0] * len(documents))
    })

    csv_data = export_df.to_csv(index=False)
    st.download_button(
        "📥 Download Pure Results",
        data=csv_data,
        file_name="pure_bertopic_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()