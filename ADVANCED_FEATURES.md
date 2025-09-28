# Advanced BERTopic Features Documentation

## Overview
The `advanced_bertopic.py` application is a professional-grade topic modeling system with sentiment awareness and enterprise features.

## Core Features

### 1. Sentiment-Aware Topic Modeling
- **Automatic Sentiment Analysis**: Uses `nlptown/bert-base-multilingual-uncased-sentiment` for 5-star rating analysis
- **Smart Document Augmentation**: Adds semantic context without polluting topic names
- **Sentiment Indicators**: Topics marked with 😊 (positive), 😔 (negative), or 😐 (mixed)
- **Proper Separation**: Ensures "excellent customer service" and "poor customer service" end up in different topics

### 2. Advanced Visualizations

#### Topic Map (2D Visualization)
- Interactive scatter plot showing topic relationships
- Topics positioned based on semantic similarity
- Click to explore specific topics

#### Hierarchical Topics
- Discover parent-child topic relationships
- Understand topic taxonomy
- Identify subtopics and meta-topics

#### Similarity Heatmap
- Visual matrix of topic-to-topic similarities
- Identify related topic clusters
- Understand topic overlap

#### Topics Over Time (requires timestamps)
- Track topic evolution
- Identify trending topics
- Understand temporal patterns

### 3. Export Options

#### CSV Export
- Complete results with topic assignments
- Sentiment labels included
- Ready for further analysis

#### Interactive HTML Report
- Standalone HTML file with all visualizations
- No dependencies required
- Share with stakeholders easily

#### Model Persistence
- Save trained models for later use
- Load existing models
- Version control your topic models

### 4. LLM Integration (Optional)

#### Bring Your Own API Key
- **Privacy First**: API keys never stored
- **Optional Feature**: Only for organizations that allow it
- **Multiple Models**: Support for GPT-3.5, GPT-4, GPT-4-turbo

#### Enhanced Topic Labels
- Generate human-readable topic descriptions
- Improve topic interpretability
- Customizable prompts

## Technical Architecture

### Class-Based Design
```python
class SentimentAwareTopicModeler:
    - analyze_sentiment()
    - create_sentiment_augmented_docs()
    - compute_embeddings()
    - create_advanced_topic_model()
    - fit_transform()
    - create_enhanced_labels()
    - create_hierarchical_topics()
    - visualize_hierarchy()
    - visualize_topics_2d()
    - visualize_heatmap()
    - track_topics_over_time()
    - generate_llm_topic_labels()
    - export_to_html()
    - save_model()
    - load_model()
```

### Best Practices Implemented
1. **Pre-computed Embeddings**: Performance optimization
2. **Fixed Random State**: Reproducible results
3. **Multi-Aspect Representation**: KeyBERT + MMR + PartOfSpeech
4. **Adaptive Parameters**: Automatic adjustment based on dataset size
5. **Intelligent Outlier Reduction**: Only when necessary
6. **Caching**: @st.cache_resource for expensive operations

## Configuration Options

### Sidebar Controls

#### Data Input
- CSV file upload
- Text column selection

#### Model Configuration
- Sentiment-aware clustering (on/off)
- Min topic size
- Number of topics (auto or manual)
- Embedding model selection

#### Advanced Features
- Hierarchical analysis
- Topic mapping
- Similarity analysis
- Auto-save model
- Cache embeddings

#### Model Management
- Load existing models
- Save trained models

## Usage Examples

### Basic Usage
1. Upload CSV with text data
2. Select text column
3. Enable "Sentiment-Aware Clustering"
4. Click "Analyze Topics & Sentiment"
5. Explore results in tabs

### Advanced Workflow
1. Load existing model (if available)
2. Configure advanced parameters
3. Run analysis
4. Generate visualizations
5. Export HTML report
6. Optionally: Generate LLM labels with API key

## Performance Considerations

### Small Datasets (<100 docs)
- Uses K-means clustering
- Reduced dimensions (3 components)
- Smaller vocabulary (1000 features)

### Medium Datasets (100-1000 docs)
- HDBSCAN clustering
- Standard dimensions (5 components)
- Medium vocabulary (2000 features)

### Large Datasets (1000+ docs)
- Optimized HDBSCAN
- Higher dimensions (10 components)
- Large vocabulary (5000 features)
- Low-memory mode for UMAP

## Security & Privacy

### API Key Handling
- Never stored permanently
- Session-only usage
- Password field masking
- Optional feature

### Data Privacy
- All processing local
- No data sent to external services (except optional LLM)
- Model files stored locally

## Troubleshooting

### Common Issues

1. **Too Many Outliers**
   - Reduce min_topic_size
   - Enable auto-reduce outliers
   - Increase number of topics

2. **Poor Topic Separation**
   - Enable sentiment-aware clustering
   - Adjust embedding model
   - Use seed topics

3. **Memory Issues**
   - Enable embedding caching
   - Use smaller embedding model
   - Process in batches

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Custom embedding models
- [ ] Real-time topic modeling
- [ ] Topic merging interface
- [ ] A/B testing for parameters
- [ ] Integration with vector databases
- [ ] Custom visualization themes

## Credits

Built using:
- BERTopic by Maarten Grootendorst
- Sentence Transformers
- Streamlit
- Plotly
- scikit-learn
- HDBSCAN
- UMAP