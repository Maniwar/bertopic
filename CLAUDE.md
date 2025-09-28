# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based demonstration application for BERTopic, a topic modeling technique that uses embeddings and clustering algorithms. The application provides an interactive interface for users to upload CSV files, configure topic modeling parameters, and visualize results.

## Architecture

### Core Components

**Main Application (`bertopic_demo.py`):**
- Single-file Streamlit application (214 lines)
- Uses caching with `@st.cache_resource` for model creation to avoid recomputation
- Integrates multiple ML libraries: BERTopic, UMAP, HDBSCAN, scikit-learn, TensorFlow

**Topic Modeling Pipeline:**
1. **Dimensionality Reduction**: UMAP for reducing document embeddings to lower dimensions
2. **Clustering**: HDBSCAN for density-based clustering of documents
3. **Vectorization**: CountVectorizer for creating document-term matrices
4. **Topic Representation**: ClassTfidfTransformer (optionally with seed words) for topic-word scoring

### Key Features

- **Seed Word Support**: Boost specific words in topic modeling using `ClassTfidfTransformer`
- **Interactive Parameter Tuning**: Sidebar controls for all major hyperparameters
- **Document Grouping**: View all documents assigned to each topic in scrollable tables
- **CSV Export**: Download results with topic assignments

## Development Commands

### Running the Application
```bash
streamlit run bertopic_demo.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Management
The project uses `requirements.txt` with these key packages:
- `streamlit` - Web application framework
- `bertopic` - Topic modeling library
- `tensorflow` - Deep learning framework
- `scikit-learn` - Machine learning utilities
- `umap-learn` - Dimensionality reduction
- `hdbscan` - Density-based clustering
- `plotly` - Interactive visualizations

## Code Architecture Patterns

### Memory Management
- Uses `tf.compat.v1.reset_default_graph()` and `tf.keras.backend.clear_session()` to prevent TensorFlow memory issues
- Streamlit caching prevents unnecessary model recreation

### Modular Design
- `create_bertopic_model()`: Configurable model factory function
- `convert_df_to_csv()`: Utility for data export
- `main()`: Application orchestration and UI logic

### Parameter Configuration
All hyperparameters are exposed through Streamlit widgets:
- **Topic Settings**: `min_topic_size`, `nr_topics`
- **UMAP Parameters**: `n_neighbors`, `n_components`, `min_dist`
- **HDBSCAN Parameters**: `min_cluster_size`, `min_samples`
- **Seed Words**: Optional topic guidance with boost multiplier

### Data Flow
1. CSV upload and column selection
2. Parameter configuration via sidebar
3. Model creation with configured parameters
4. Document processing and topic assignment
5. Results visualization and export

## Development Notes

- The application is designed as a single-file demo rather than a production system
- TensorFlow session management is explicitly handled to prevent memory leaks
- All text preprocessing uses single-word tokenization (`ngram_range=(1,1)`)
- Topic results are merged back into the original DataFrame for export
- HTML is used for custom scrollable document displays

## Critical Pattern Recognition - Embedding-based Systems

**NEVER REVERT TO KEYWORD LISTS** when working with embedding-based systems (BERTopic, sentence-transformers, etc.)

### Root Cause Analysis (5 Whys):
1. **Why keyword lists?** → Trying to solve mixed topic problems
2. **Why not trust embeddings?** → Assumed embeddings failed instead of tuning parameters
3. **Why assume failure?** → Defaulted to "quick fix" pattern matching
4. **Why quick fixes?** → Didn't trust that embeddings CAN distinguish concerns with proper parameters
5. **Why no trust?** → Forgot the core principle: embeddings discover patterns dynamically

### Correct Approach:
- **ALWAYS** use cosine similarity between embeddings
- **ALWAYS** use dynamic clustering (AgglomerativeClustering, HDBSCAN)
- **NEVER** define static categories like `{'customer_service': [...], 'shipping': [...]}`
- **NEVER** use predefined theme detection
- Let data define its own categories through embeddings
- Adjust similarity thresholds (0.65-0.7 range typically works)
- Use hierarchical clustering for sub-topic refinement