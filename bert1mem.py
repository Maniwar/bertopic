import streamlit as st
import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import re
from collections import Counter
from scipy.spatial.distance import cosine
import plotly.express as px
import plotly.graph_objects as go
# ✅ NEW IMPORTS FOR LLM OPTIMIZATION
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psutil
import os


# =====================================================
# 🚀 MEMORY OPTIMIZATION PROFILES
# =====================================================
class MemoryProfileConfig:
    """
    Configure memory usage vs performance tradeoff
    Higher memory usage = faster processing
    """
    
    PROFILES = {
        'conservative': {
            'name': '💾 Conservative',
            'description': 'Low memory usage, slower (safe for 8GB RAM)',
            'llm_batch_size_multiplier': 0.5,
            'max_workers_multiplier': 0.5,
            'cache_embeddings': True,
            'cache_topic_docs': False,
            'precompute_metadata': False,
            'aggressive_caching': False,
            'tokenizer_cache_size': 1000,
            'max_docs_in_memory': 5000,
        },
        'balanced': {
            'name': '⚖️ Balanced',
            'description': 'Moderate memory, good speed (16GB RAM)',
            'llm_batch_size_multiplier': 1.0,
            'max_workers_multiplier': 1.0,
            'cache_embeddings': True,
            'cache_topic_docs': True,
            'precompute_metadata': True,
            'aggressive_caching': False,
            'tokenizer_cache_size': 5000,
            'max_docs_in_memory': 20000,
        },
        'aggressive': {
            'name': '🚀 Aggressive',
            'description': 'High memory, maximum speed (32GB+ RAM)',
            'llm_batch_size_multiplier': 2.0,
            'max_workers_multiplier': 1.5,
            'cache_embeddings': True,
            'cache_topic_docs': True,
            'precompute_metadata': True,
            'aggressive_caching': True,
            'tokenizer_cache_size': 20000,
            'max_docs_in_memory': 100000,
        },
        'extreme': {
            'name': '⚡ Extreme',
            'description': 'Maximum memory, extreme speed (64GB+ RAM)',
            'llm_batch_size_multiplier': 3.0,
            'max_workers_multiplier': 2.0,
            'cache_embeddings': True,
            'cache_topic_docs': True,
            'precompute_metadata': True,
            'aggressive_caching': True,
            'tokenizer_cache_size': 50000,
            'max_docs_in_memory': 500000,
        }
    }
    
    @staticmethod
    def get_profile(profile_name='balanced'):
        """Get a memory profile configuration"""
        return MemoryProfileConfig.PROFILES.get(profile_name, MemoryProfileConfig.PROFILES['balanced'])
    
    @staticmethod
    def estimate_memory_usage(profile_name, num_docs, embedding_dim=384):
        """Estimate memory usage in GB for a given profile"""
        profile = MemoryProfileConfig.get_profile(profile_name)
        
        # Base memory estimates
        embeddings_gb = (num_docs * embedding_dim * 4) / (1024**3)  # float32
        docs_gb = (num_docs * 500) / (1024**3)  # ~500 chars per doc
        
        # Additional memory based on profile
        cache_multiplier = 2.0 if profile['aggressive_caching'] else 1.2
        
        if profile['cache_topic_docs']:
            total_gb = (embeddings_gb + docs_gb) * cache_multiplier
        else:
            total_gb = embeddings_gb * cache_multiplier
        
        return total_gb
    
    @staticmethod
    def get_recommended_profile(available_ram_gb, num_docs):
        """Recommend a profile based on available RAM and document count"""
        estimates = {}
        for profile_name in MemoryProfileConfig.PROFILES.keys():
            est_usage = MemoryProfileConfig.estimate_memory_usage(profile_name, num_docs)
            estimates[profile_name] = est_usage
        
        # Leave 20% RAM free for system
        usable_ram = available_ram_gb * 0.8
        
        # Find best profile that fits
        for profile_name in ['extreme', 'aggressive', 'balanced', 'conservative']:
            if estimates[profile_name] <= usable_ram:
                return profile_name
        
        return 'conservative'


# =====================================================
# 🚀 LLM MODEL CONTEXT WINDOW CONFIGURATION
# =====================================================
LLM_MODEL_CONFIG = {
    "microsoft/Phi-3-mini-4k-instruct": {
        "context_window": 4096,
        "description": "3.8GB - Fast, good quality",
        "recommended_docs": 8  # 4k tokens allows ~8 docs with 500 chars each
    },
    "microsoft/Phi-3-mini-128k-instruct": {
        "context_window": 131072,  # 128k tokens
        "description": "3.8GB - Fast, massive context",
        "recommended_docs": 50  # 128k tokens allows ~50+ docs with full context
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "context_window": 8192,  # 8k tokens
        "description": "14GB - Better quality",
        "recommended_docs": 15  # 8k tokens allows ~15 docs
    },
    "HuggingFaceH4/zephyr-7b-beta": {
        "context_window": 8192,  # 8k tokens
        "description": "14GB - Good quality",
        "recommended_docs": 15  # 8k tokens allows ~15 docs
    },
}

def get_max_docs_for_model(model_name):
    """Calculate optimal number of documents to use based on model's context window"""
    if model_name not in LLM_MODEL_CONFIG:
        return 8  # Default fallback
    
    config = LLM_MODEL_CONFIG[model_name]
    return config["recommended_docs"]

def get_max_tokens_for_model(model_name):
    """Get the maximum token limit for a model"""
    if model_name not in LLM_MODEL_CONFIG:
        return 4096  # Default fallback
    
    config = LLM_MODEL_CONFIG[model_name]
    return config["context_window"]

# Set Streamlit to wide mode for better layout
st.set_page_config(layout="wide", page_title="Complete BERTopic with All Features", page_icon="🚀")

# Make accelerate optional - not strictly required
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

# Try to import GPU-accelerated libraries
try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False

try:
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    cuml_available = True
except ImportError:
    from umap import UMAP
    from hdbscan import HDBSCAN
    cuml_available = False

# Try to import FAISS for GPU-accelerated similarity search
try:
    import faiss
    faiss_available = True
    # Check if GPU version is available
    if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
        faiss_gpu_available = True
    else:
        faiss_gpu_available = False
except ImportError:
    faiss_available = False
    faiss_gpu_available = False

# Force CUDA initialization if available
if torch.cuda.is_available():
    try:
        torch.cuda.init()
        torch.cuda.empty_cache()
        # Enable TF32 for better performance on newer GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
    except Exception as e:
        st.warning(f"CUDA initialization warning: {e}")

# =====================================================
# ⚡ OPTIMIZED LLM LABELING - AUTO-ADAPTIVE & PARALLEL
# =====================================================
class SystemPerformanceDetector:
    """Auto-detect system capabilities and recommend optimal parameters"""
    
    @staticmethod
    def detect_optimal_parameters(memory_profile='balanced'):
        """Detect system capabilities and return optimal parameters for LLM labeling"""
        # Get memory profile config
        profile = MemoryProfileConfig.get_profile(memory_profile)
        
        params = {
            'has_gpu': False,
            'gpu_memory_gb': 0,
            'cpu_cores': os.cpu_count() or 4,
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'batch_size': 10,
            'max_workers': 1,
            'recommended_docs_per_topic': 5,
            'memory_profile': memory_profile,
            'profile_config': profile
        }
        
        if torch.cuda.is_available():
            params['has_gpu'] = True
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                params['gpu_memory_gb'] = gpu_props.total_memory / (1024**3)
                params['gpu_name'] = gpu_props.name
                
                # Apply memory profile multipliers
                base_batch_size = 0
                if params['gpu_memory_gb'] >= 20:
                    base_batch_size = 40
                    base_workers = 4
                    params['tier'] = 'High-end GPU'
                elif params['gpu_memory_gb'] >= 14:
                    base_batch_size = 25
                    base_workers = 3
                    params['tier'] = 'Mid-range GPU'
                elif params['gpu_memory_gb'] >= 10:
                    base_batch_size = 20
                    base_workers = 3
                    params['tier'] = 'Standard GPU'
                elif params['gpu_memory_gb'] >= 6:
                    base_batch_size = 12
                    base_workers = 2
                    params['tier'] = 'Entry-level GPU'
                else:
                    base_batch_size = 8
                    base_workers = 1
                    params['tier'] = 'Low-memory GPU'
                
                # ✅ Apply memory profile multipliers
                params['batch_size'] = int(base_batch_size * profile['llm_batch_size_multiplier'])
                params['max_workers'] = max(1, int(base_workers * profile['max_workers_multiplier']))
                
            except Exception:
                params['batch_size'] = int(15 * profile['llm_batch_size_multiplier'])
                params['max_workers'] = max(1, int(2 * profile['max_workers_multiplier']))
                params['tier'] = 'Unknown GPU'
        else:
            params['tier'] = 'CPU'
            params['batch_size'] = int(10 * profile['llm_batch_size_multiplier'])
            params['max_workers'] = 1
            if params['cpu_cores'] >= 16:
                params['max_workers'] = max(1, int(2 * profile['max_workers_multiplier']))
        
        return params


# =====================================================
# 🚀 AGGRESSIVE MEMORY CACHING SYSTEM
# =====================================================
class AggressiveDocumentCache:
    """Cache documents and metadata in memory for ultra-fast access"""
    
    def __init__(self, memory_profile='balanced'):
        self.profile = MemoryProfileConfig.get_profile(memory_profile)
        self.enabled = self.profile['aggressive_caching']
        
        # Caches
        self.topic_docs_cache = {}
        self.topic_metadata_cache = {}
        self.tokenized_docs_cache = {}
        self.embedding_cache = None
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
    
    def cache_topic_documents(self, topics_dict):
        """Pre-load all topic documents into memory"""
        if not self.profile['cache_topic_docs']:
            return
        
        self.topic_docs_cache = {}
        for topic_id, docs in topics_dict.items():
            # Limit based on profile
            max_docs = self.profile['max_docs_in_memory']
            self.topic_docs_cache[topic_id] = docs[:max_docs]
    
    def cache_topic_metadata(self, topic_info_df):
        """Pre-compute and cache topic metadata"""
        if not self.profile['precompute_metadata']:
            return
        
        self.topic_metadata_cache = {}
        for _, row in topic_info_df.iterrows():
            topic_id = row['Topic']
            self.topic_metadata_cache[topic_id] = {
                'count': row['Count'],
                'keywords': row.get('Keywords', ''),
                'human_label': row.get('Human_Label', ''),
                'representative_docs': row.get('Representative_Docs', [])
            }
    
    def cache_embeddings(self, embeddings):
        """Cache embeddings array"""
        if not self.profile['cache_embeddings']:
            return
        
        self.embedding_cache = embeddings.copy() if hasattr(embeddings, 'copy') else embeddings
    
    def get_topic_docs(self, topic_id):
        """Get documents for a topic from cache"""
        if topic_id in self.topic_docs_cache:
            self.cache_hits += 1
            return self.topic_docs_cache[topic_id]
        self.cache_misses += 1
        return None
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        memory_used = 0
        if self.topic_docs_cache:
            memory_used += sum(len(str(docs)) for docs in self.topic_docs_cache.values()) / (1024**2)
        if self.embedding_cache is not None:
            memory_used += self.embedding_cache.nbytes / (1024**2) if hasattr(self.embedding_cache, 'nbytes') else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_mb': memory_used
        }


class AdaptiveProgressTracker:
    """Track and display detailed progress with adaptive updates"""
    
    def __init__(self, total_items, operation_name="Processing"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.start_time = time.time()
        self.completed = 0
        self.progress_bar = st.progress(0.0)
        self.status_container = st.empty()
        self.metrics_container = st.empty()
        
    def update(self, completed_count, extra_info=None):
        """Update progress with detailed stats"""
        self.completed = completed_count
        progress = min(1.0, completed_count / self.total_items)
        self.progress_bar.progress(progress)
        
        elapsed = time.time() - self.start_time
        rate = completed_count / elapsed if elapsed > 0 else 0
        remaining = self.total_items - completed_count
        eta = remaining / rate if rate > 0 else 0
        
        elapsed_str = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed/60)}m {int(elapsed%60)}s"
        eta_str = f"{int(eta)}s" if eta < 60 else f"{int(eta/60)}m {int(eta%60)}s"
        
        status_msg = f"🚀 {self.operation_name}: {completed_count}/{self.total_items}"
        if extra_info:
            status_msg += f" | {extra_info}"
        self.status_container.info(status_msg)
        
        col1, col2, col3, col4 = self.metrics_container.columns(4)
        with col1:
            st.metric("Progress", f"{progress*100:.1f}%")
        with col2:
            st.metric("Rate", f"{rate:.1f}/sec")
        with col3:
            st.metric("Elapsed", elapsed_str)
        with col4:
            st.metric("ETA", eta_str if eta > 0 else "Finishing...")
    
    def complete(self, final_message=None):
        """Mark as complete and clean up"""
        elapsed = time.time() - self.start_time
        self.progress_bar.empty()
        self.metrics_container.empty()
        
        if final_message:
            self.status_container.success(f"✅ {final_message} (took {elapsed:.1f}s)")
        else:
            self.status_container.success(f"✅ {self.operation_name} complete! (took {elapsed:.1f}s, {self.total_items/elapsed:.1f} items/sec)")




def clean_documents_for_llm(docs, max_docs=50):
    """
    Clean documents for LLM by removing sanitized tokens and noise
    
    This handles:
    - <entity_type> and similar sanitization markers
    - XML-style tags
    - Excessive repetition
    - Special characters that confuse LLMs
    """
    cleaned = []
    for doc in docs[:max_docs]:
        # Remove common sanitization patterns
        text = str(doc)
        
        # Remove XML-style sanitization tags but keep the context
        text = re.sub(r'<entity_type>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)  # Remove any other XML tags
        
        # Remove excessive repetition (same word 3+ times in a row)
        text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Only keep if meaningful length
        if len(text) > 20:
            cleaned.append(text)
    
    return cleaned


def generate_batch_labels_with_llm(batch_data, llm_model, max_docs_per_topic=8, max_topics_per_batch=20, model_name=None, labeler=None):
    """
    ✅ REIMAGINED: Trust the LLM with rich context and chain-of-thought reasoning

    Let the LLM actually READ and THINK instead of forcing it into a box.

    Now with FAISS-powered document selection for optimal representativeness!
    """
    if not batch_data or llm_model is None:
        return {}

    try:
        model, tokenizer = llm_model

        # Calculate max prompt length based on model's context window
        if model_name:
            max_prompt_length = int(get_max_tokens_for_model(model_name) * 0.70)  # Use 70% for prompt
        else:
            max_prompt_length = 3000

        # Build rich topic descriptions
        topics_text = []
        for i, item in enumerate(batch_data[:max_topics_per_batch], 1):
            topic_id = item['topic_id']
            keywords = item['keywords']
            all_docs = clean_documents_for_llm(item['docs'])

            if not all_docs:
                continue

            # ✅ FAISS-POWERED DOCUMENT SELECTION (if available)
            if labeler and labeler.use_faiss_selection:
                # Get document indices for this topic
                topic_doc_indices = item.get('doc_indices', list(range(len(all_docs))))
                sample_docs = labeler.select_representative_documents_with_faiss(
                    all_docs,
                    topic_doc_indices,
                    max_docs_per_topic
                )
            else:
                # Fallback: Strategic sampling (beginning, middle, end, plus random)
                if len(all_docs) <= max_docs_per_topic:
                    sample_docs = all_docs
                else:
                    indices = [
                        0,
                        len(all_docs) // 4,
                        len(all_docs) // 2,
                        3 * len(all_docs) // 4,
                        len(all_docs) - 1,
                    ]
                    import random
                    remaining = max_docs_per_topic - len(indices)
                    if remaining > 0:
                        available = [idx for idx in range(len(all_docs)) if idx not in indices]
                        if available:
                            indices.extend(random.sample(available, min(remaining, len(available))))

                    sample_docs = [all_docs[idx] for idx in sorted(set(indices[:max_docs_per_topic]))]
            
            # ✅ SMART SCALING: More docs = shorter excerpts to fit in context
            if max_docs_per_topic >= 40:
                doc_length = 400  # Phi-128k with 50 docs: 400 chars each
            elif max_docs_per_topic >= 20:
                doc_length = 500  # Mid-size: 500 chars each
            elif max_docs_per_topic >= 10:
                doc_length = 600  # Standard: 600 chars each
            else:
                doc_length = 800  # Few docs: show more per doc
            
            # Show documents with CRYSTAL CLEAR boundaries
            docs_preview = "\n".join([
                f"━━━ DOCUMENT {j+1} of {len(sample_docs)} ━━━\n{doc[:doc_length]}\n━━━ END DOC {j+1} ━━━" 
                for j, doc in enumerate(sample_docs[:max_docs_per_topic])
            ])
            
            topics_text.append(
                f"\n{'='*70}\n"
                f"📋 TOPIC {i}\n"
                f"Keywords: {keywords}\n"
                f"Analyzing {len(sample_docs)} representative documents (from {len(all_docs)} total)\n"
                f"\n{docs_preview}\n"
            )
        
        # ✅ HIERARCHICAL PROMPT: Request deeply structured labels with multiple layers
        batch_prompt = f"""Create DETAILED hierarchical category names for these customer support topics.

REQUIREMENTS for each category name:
• Use format: "Main Category - Specific Details - Additional Context"
• Add as many layers with "-" as needed to make each label UNIQUE and DESCRIPTIVE
• Main Category: Broad theme (2-3 words)
• Specific Details: Concrete specifics (2-4 words)
• Additional Context: Extra distinguishing details (1-3 words) if needed
• Mention specific products, services, issues, or contexts
• Each label must be distinct from all others
• No generic phrases like "Help" or "Question"

EXAMPLES OF GOOD HIERARCHICAL LABELS:
• "Product Orders - Samsung Washer - Delivery Scheduling"
• "Product Orders - LG Refrigerator - Installation Issues"
• "Customer Service - Response Time - Phone Support"
• "Customer Service - Response Time - Email Support"
• "Technical Support - Installation - Dishwasher Setup"
• "Account Management - Student Discount - Verification Process"
• "Billing Issues - Payment Failed - Credit Card"
• "Billing Issues - Payment Failed - Bank Transfer"

{chr(10).join(topics_text)}

Create a UNIQUE, DETAILED hierarchical category name for each topic above.
Add as many layers with "-" as needed to distinguish it from other topics.

FORMAT (respond EXACTLY like this):
1: [Main Category - Specific Details - Additional Context]
2: [Main Category - Specific Details - Additional Context]
3: [Main Category - Specific Details - Additional Context]

Your response:"""
        
        # Generate with much more space for thoughtful responses
        inputs = tokenizer(batch_prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,  # ✅ Shorter - just need category names
                temperature=0.3,     # ✅ Lower for consistency
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,  # ✅ Prevent repeating same label
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        if "Your response:" in response:
            response = response.split("Your response:")[-1]
        elif "response:" in response.lower():
            response = response.split("response:")[-1]
        
        # ✅ IMPROVED PARSING: More forgiving, looks for "NUMBER: label" patterns
        labels_dict = {}
        seen_labels = set()  # Track to prevent duplicates
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Match patterns like: "1: label", "1. label", "Topic 1: label", etc.
            import re
            match = re.match(r'(?:topic\s*)?(\d+)[\s:.\)]+(.+)', line, re.IGNORECASE)
            if match:
                try:
                    topic_num = int(match.group(1))
                    label = match.group(2).strip()
                    
                    # Clean up
                    label = label.strip(' "\'.,;[](){}')
                    label = re.sub(r'\s+', ' ', label)  # Normalize whitespace
                    label = re.sub(r'^(category name|label):\s*', '', label, flags=re.IGNORECASE)  # Remove prefix if present
                    
                    # Validate and check for duplicates
                    label_lower = label.lower()
                    if (5 <= len(label) <= 200 and  # ✅ Increased to 200 for multi-level labels
                        2 <= len(label.split()) <= 20 and  # ✅ Increased to 20 words for detailed labels
                        label_lower not in seen_labels and  # ✅ No duplicates in same batch
                        not any(bad in label_lower for bad in ['help buy', 'question buy', '[', ']'])):  # ✅ Filter obvious bad ones

                        # ✅ Ensure hierarchical format: Add "-" if not present and has enough words
                        if ' - ' not in label and len(label.split()) >= 3:
                            # Split label into two parts for hierarchy
                            words = label.split()
                            mid = len(words) // 2
                            label = f"{' '.join(words[:mid])} - {' '.join(words[mid:])}"

                        if 1 <= topic_num <= len(batch_data):
                            actual_topic_id = batch_data[topic_num - 1]['topic_id']
                            labels_dict[actual_topic_id] = label
                            seen_labels.add(label_lower)
                except:
                    continue
        
        return labels_dict
        
    except Exception as e:
        # Silent fail - fallback will handle it
        return {}



# =====================================================
# ✅ GLOBAL DEDUPLICATION
# =====================================================
def generate_batch_llm_analysis(topic_batch, llm_model):
    """
    Generate LLM analysis for multiple topics in a single batch call.

    This is MUCH faster than processing topics one-by-one because:
    1. Single LLM inference call for multiple topics
    2. Better GPU utilization
    3. No thread serialization issues

    Args:
        topic_batch: List of dicts with keys: topic_id, label, docs
        llm_model: Tuple of (model, tokenizer)

    Returns:
        Dict mapping topic_id to analysis string
    """
    if not topic_batch or llm_model is None:
        return {}

    try:
        model, tokenizer = llm_model

        # Build batched prompt
        prompt_parts = ["Analyze these topics and provide a one-sentence summary for each:\n"]

        for item in topic_batch:
            topic_id = item['topic_id']
            label = item['label']
            docs = item['docs'][:3]  # Use fewer docs per topic for batching

            # Clean docs
            cleaned = [str(d).strip()[:150] for d in docs if d and str(d).strip()]
            docs_text = " | ".join(cleaned[:3])

            prompt_parts.append(f"\nTopic {topic_id} ({label}):")
            prompt_parts.append(f"Documents: {docs_text}")
            prompt_parts.append(f"Analysis:")

        prompt = "\n".join(prompt_parts)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # More tokens for multiple topics
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse responses for each topic
        results = {}
        lines = response.split('\n')

        current_topic_id = None
        for line in lines:
            line = line.strip()

            # Look for "Analysis:" followed by text
            if 'Analysis:' in line:
                parts = line.split('Analysis:', 1)
                if len(parts) == 2:
                    analysis_text = parts[1].strip()
                    if analysis_text and current_topic_id is not None:
                        # Clean up
                        analysis_text = analysis_text.split('\n')[0].strip()
                        analysis_text = analysis_text.strip('"\'.,;[](){}')
                        if len(analysis_text) > 150:
                            analysis_text = analysis_text[:150] + "..."
                        results[current_topic_id] = analysis_text

            # Track which topic we're parsing
            if line.startswith('Topic ') and '(' in line:
                try:
                    topic_num = int(line.split()[1].split('(')[0])
                    current_topic_id = topic_num
                except:
                    pass

        return results

    except Exception as e:
        st.caption(f"⚠️ Batch LLM analysis failed: {str(e)[:100]}")
        return {}


def generate_simple_llm_analysis(topic_id, sample_docs, topic_label, llm_model, max_length=150):
    """
    Generate a simple one-sentence analysis of what users are saying in this topic.

    Prompt: "Look at topic X and tell me what users are saying in one sentence."
    """
    if not sample_docs or llm_model is None:
        return None

    try:
        model, tokenizer = llm_model

        # Clean and prepare documents
        cleaned_docs = [str(doc).strip() for doc in sample_docs if doc and str(doc).strip()]
        if not cleaned_docs:
            return None

        # Build simple prompt
        docs_text = "\n".join([f"- {doc[:200]}" for doc in cleaned_docs[:5]])

        prompt = f"""Topic: {topic_label}

Sample documents:
{docs_text}

In one clear sentence, what are users saying in this topic? Focus on their main concerns, questions, or feedback.

Answer:"""

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,  # Short response
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        # Clean up
        response = response.split('\n')[0].strip()  # Take first line
        response = response.strip('"\'.,;[](){}')

        # Truncate if needed
        if len(response) > max_length:
            response = response[:max_length].strip() + "..."

        return response if response else None

    except Exception as e:
        st.caption(f"⚠️ Topic {topic_id} LLM analysis failed: {str(e)[:100]}")
        return None


def deduplicate_labels_globally(labels_dict, keywords_dict, topics_dict=None):
    """
    Deduplicate labels across ALL topics by progressively adding layers with '-'.
    Keeps adding detail levels until every label is unique.

    Examples:
    - "Customer Service - Response Times" becomes "Customer Service - Response Times - Phone Support"
    - "Product Orders - Delivery" becomes "Product Orders - Delivery - Samsung Appliances"
    """
    MAX_ITERATIONS = 5  # Prevent infinite loops
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # Count label occurrences
        label_to_topics = {}
        for topic_id, label in labels_dict.items():
            label_lower = label.lower().strip()
            if label_lower not in label_to_topics:
                label_to_topics[label_lower] = []
            label_to_topics[label_lower].append(topic_id)

        # Check if we have any duplicates
        duplicates = {label: topics for label, topics in label_to_topics.items() if len(topics) > 1}

        if not duplicates:
            # All labels are unique!
            break

        # Fix duplicates by adding another layer of detail
        for label_lower, topic_ids in duplicates.items():
            for topic_id in topic_ids:
                original_label = labels_dict[topic_id]
                keywords = keywords_dict.get(topic_id, '')

                # Extract distinctive keywords not already in the label
                kw_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                distinctive = []

                # Filter out keywords already in label and common words
                common_words = {'help', 'buy', 'question', 'new', 'order', 'phone', 'customer', 'service', 'product', 'support'}
                label_words = set(original_label.lower().split())

                for kw in kw_list[:10]:  # Look at more keywords
                    kw_clean = kw.lower().strip()
                    # Skip if already in label or is common word
                    if kw_clean not in common_words and kw_clean not in label_words:
                        # Also check if not substring of any word in label
                        if not any(kw_clean in word for word in label_words):
                            distinctive.append(kw.title())
                            if len(distinctive) >= 2:
                                break

                # If we have distinctive keywords, add them as a new layer
                if distinctive:
                    new_detail = ' '.join(distinctive[:2])
                    labels_dict[topic_id] = f"{original_label} - {new_detail}"
                else:
                    # Try to use document content if available
                    if topics_dict and topic_id in topics_dict:
                        docs = topics_dict[topic_id]
                        # Extract unique phrases from documents
                        extra_phrases = _top_phrases(docs[:50], (1, 2), top_k=10)
                        extra_clean = []
                        for phrase in extra_phrases:
                            phrase_clean = _clean_phrase(phrase)
                            phrase_words = set(phrase_clean.lower().split())
                            if not phrase_words.issubset(label_words) and phrase_clean.lower() not in common_words:
                                extra_clean.append(_to_title_case(phrase_clean))
                                if len(extra_clean) >= 2:
                                    break

                        if extra_clean:
                            new_detail = ' '.join(extra_clean[:2])
                            labels_dict[topic_id] = f"{original_label} - {new_detail}"
                        else:
                            # Last resort: add topic ID
                            labels_dict[topic_id] = f"{original_label} - Topic {topic_id}"
                    else:
                        # No documents available, use index
                        variant_num = topic_ids.index(topic_id) + 1
                        labels_dict[topic_id] = f"{original_label} - Variant {variant_num}"

    return labels_dict


class AdaptiveParallelLLMLabeler:
    """Production-ready parallel LLM labeler with adaptive batch sizing and detailed progress"""

    def __init__(self, llm_model, model_name=None, system_params=None, max_docs_per_topic=None, faiss_index=None, embeddings=None, documents=None):
        self.llm_model = llm_model
        self.model_name = model_name

        # ✅ FAISS-based document selection for better quality
        self.faiss_index = faiss_index
        self.embeddings = embeddings
        self.documents = documents
        self.use_faiss_selection = faiss_index is not None and embeddings is not None and documents is not None

        # ✅ Calculate max_docs dynamically based on model's context window
        if max_docs_per_topic is None and model_name:
            self.max_docs_per_topic = get_max_docs_for_model(model_name)
        elif max_docs_per_topic is None:
            self.max_docs_per_topic = 8  # Default fallback
        else:
            # Use at least 8 documents for better label quality
            self.max_docs_per_topic = max(max_docs_per_topic, 8)

        if system_params is None:
            system_params = SystemPerformanceDetector.detect_optimal_parameters()

        self.system_params = system_params
        self.batch_size = min(system_params['batch_size'], 10)  # ✅ Cap at 10 for better quality
        self.max_workers = system_params['max_workers'] if llm_model else 0
        self.original_batch_size = self.batch_size
        self.oom_count = 0
        self.successful_batches = 0

    def select_representative_documents_with_faiss(self, topic_docs, topic_doc_indices, max_docs):
        """
        Use FAISS to intelligently select the most representative documents for a topic.

        Strategy:
        1. Calculate topic centroid from document embeddings
        2. Find documents closest to centroid (most representative)
        3. Also include some diverse documents (farthest from each other)
        4. Blend for both coverage and representativeness
        """
        if not self.use_faiss_selection or len(topic_doc_indices) == 0:
            return topic_docs[:max_docs]

        try:
            # Get embeddings for this topic's documents
            topic_embeddings = self.embeddings[topic_doc_indices]

            # Calculate centroid
            centroid = np.mean(topic_embeddings, axis=0).reshape(1, -1).astype('float32')

            # Find closest documents to centroid (most representative)
            n_representative = max(1, int(max_docs * 0.7))  # 70% representative
            distances, indices = self.faiss_index.search(centroid, min(len(topic_doc_indices), n_representative * 2))

            # Map back to topic document indices
            representative_indices = []
            for idx in indices[0]:
                if idx in topic_doc_indices:
                    representative_indices.append(topic_doc_indices.index(idx))
                    if len(representative_indices) >= n_representative:
                        break

            # Add some diverse documents (for coverage)
            n_diverse = max_docs - len(representative_indices)
            if n_diverse > 0:
                # Use documents farthest from centroid
                remaining_indices = [i for i in range(len(topic_docs)) if i not in representative_indices]
                if remaining_indices:
                    # Simple diversity: take evenly spaced documents
                    step = max(1, len(remaining_indices) // n_diverse)
                    diverse_indices = remaining_indices[::step][:n_diverse]
                    representative_indices.extend(diverse_indices)

            # Select documents
            selected_docs = [topic_docs[i] for i in representative_indices[:max_docs]]

            return selected_docs if selected_docs else topic_docs[:max_docs]

        except Exception as e:
            # Fallback to original approach
            st.warning(f"⚠️ FAISS selection failed, using fallback sampling: {str(e)}")
            return topic_docs[:max_docs]
        
    def _adaptive_reduce_batch_size(self):
        """Reduce batch size if OOM errors occur"""
        if self.batch_size > 5:
            self.batch_size = max(5, self.batch_size // 2)
            self.oom_count += 1
            st.warning(f"⚠️ Reducing batch size to {self.batch_size} due to memory constraints (attempt {self.oom_count})")
            
            if self.oom_count >= 2 and self.max_workers > 1:
                self.max_workers = max(1, self.max_workers - 1)
                st.warning(f"⚠️ Reducing parallel workers to {self.max_workers}")
            
            return True
        return False
    
    def label_all_topics(self, topics_dict, keywords_dict, topic_doc_indices=None, fallback_func=None, show_progress=True):
        """Label all topics using adaptive parallel batching with detailed progress"""
        topic_items = []
        for topic_id, docs in topics_dict.items():
            if topic_id == -1:
                continue
            item = {
                'topic_id': topic_id,
                'docs': docs,
                'keywords': keywords_dict.get(topic_id, '')
            }
            # Add document indices for FAISS-based selection
            if topic_doc_indices and topic_id in topic_doc_indices:
                item['doc_indices'] = topic_doc_indices[topic_id]
            topic_items.append(item)

        total_topics = len(topic_items)
        if total_topics == 0:
            return {}, {'llm_count': 0, 'fallback_count': 0, 'total': 0}

        if show_progress:
            sys_info = self.system_params
            st.info(
                f"🖥️ **System**: {sys_info['tier']} | "
                f"**Batch Size**: {self.batch_size} topics | "
                f"**Parallel Workers**: {self.max_workers} | "
                f"**Total Topics**: {total_topics}"
            )

            if sys_info['has_gpu']:
                st.info(f"🎮 **GPU**: {sys_info.get('gpu_name', 'Unknown')} ({sys_info['gpu_memory_gb']:.1f}GB)")

        # Track statistics
        self.llm_labeled_count = 0
        self.fallback_labeled_count = 0

        if self.llm_model is None or self.max_workers == 0:
            if self.llm_model is None:
                st.warning("⚠️ LLM model is None - falling back to TF-IDF labeling")
            elif self.max_workers == 0:
                st.warning(f"⚠️ max_workers is 0 (llm_model={'None' if self.llm_model is None else 'loaded'}) - falling back to TF-IDF labeling")
            result = self._sequential_fallback(topic_items, fallback_func, show_progress)
            result = deduplicate_labels_globally(result, keywords_dict, topics_dict)
            stats = {'llm_count': 0, 'fallback_count': len(result), 'total': len(result)}
            return result, stats

        max_retries = 3
        for retry in range(max_retries):
            try:
                result = self._process_batches_parallel(topic_items, fallback_func, show_progress)
                # ✅ FIX 6: Deduplicate labels globally with progressive detail
                result = deduplicate_labels_globally(result, keywords_dict, topics_dict)
                stats = {
                    'llm_count': self.llm_labeled_count,
                    'fallback_count': self.fallback_labeled_count,
                    'total': len(result)
                }
                return result, stats
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    torch.cuda.empty_cache()
                    if self._adaptive_reduce_batch_size():
                        st.warning(f"🔄 Retrying with smaller batch size... (attempt {retry + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        st.error("❌ Cannot reduce batch size further. Falling back to sequential processing.")
                        result = self._sequential_fallback(topic_items, fallback_func, show_progress)
                        result = deduplicate_labels_globally(result, keywords_dict, topics_dict)
                        stats = {'llm_count': 0, 'fallback_count': len(result), 'total': len(result)}
                        return result, stats
                else:
                    raise e

        st.warning("⚠️ All parallel attempts failed. Using sequential fallback.")
        result = self._sequential_fallback(topic_items, fallback_func, show_progress)
        result = deduplicate_labels_globally(result, keywords_dict, topics_dict)
        stats = {'llm_count': 0, 'fallback_count': len(result), 'total': len(result)}
        return result, stats
    
    def _process_batches_parallel(self, topic_items, fallback_func, show_progress):
        """Process batches in parallel with detailed progress tracking"""
        batches = [topic_items[i:i + self.batch_size] for i in range(0, len(topic_items), self.batch_size)]
        total_topics = len(topic_items)
        num_batches = len(batches)
        
        if show_progress:
            tracker = AdaptiveProgressTracker(total_topics, f"Labeling with LLM ({num_batches} batches)")
        
        all_labels = {}
        completed_topics = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(generate_batch_labels_with_llm, batch, self.llm_model, self.max_docs_per_topic, self.batch_size, self.model_name, self): (batch_idx, batch)
                for batch_idx, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]
                
                try:
                    batch_labels = future.result(timeout=120)
                    self.successful_batches += 1
                    all_labels.update(batch_labels)

                    # Track LLM vs fallback
                    llm_count_in_batch = len(batch_labels)
                    self.llm_labeled_count += llm_count_in_batch

                    fallback_count = 0
                    for item in batch:
                        topic_id = item['topic_id']
                        if topic_id not in batch_labels:
                            fallback_label = fallback_func(item['docs'], item['keywords'])
                            all_labels[topic_id] = fallback_label
                            fallback_count += 1

                    self.fallback_labeled_count += fallback_count

                    completed_topics += len(batch)
                    if show_progress:
                        extra_info = f"Batch {batch_idx+1}/{num_batches}"
                        if fallback_count > 0:
                            extra_info += f" ({fallback_count} fallbacks)"
                        tracker.update(completed_topics, extra_info)

                except Exception:
                    # Entire batch failed, use fallback for all
                    for item in batch:
                        topic_id = item['topic_id']
                        fallback_label = fallback_func(item['docs'], item['keywords'])
                        all_labels[topic_id] = fallback_label

                    self.fallback_labeled_count += len(batch)

                    completed_topics += len(batch)
                    if show_progress:
                        tracker.update(completed_topics, f"Batch {batch_idx+1}/{num_batches} (failed, used fallback)")
        
        elapsed = time.time() - start_time
        
        if show_progress:
            success_rate = (self.successful_batches / num_batches * 100) if num_batches > 0 else 0
            tracker.complete(
                f"Labeled {total_topics} topics | Success rate: {success_rate:.0f}% | "
                f"Rate: {total_topics/elapsed:.1f} topics/sec"
            )
        
        return all_labels
    
    def _sequential_fallback(self, topic_items, fallback_func, show_progress):
        """Sequential processing without LLM"""
        total = len(topic_items)
        
        if show_progress:
            st.warning("⚠️ Using TF-IDF fallback method (no LLM)")
            tracker = AdaptiveProgressTracker(total, "Generating labels with TF-IDF")
        
        all_labels = {}
        
        for i, item in enumerate(topic_items):
            topic_id = item['topic_id']
            label = fallback_func(item['docs'], item['keywords'])
            all_labels[topic_id] = label
            
            if show_progress and (i % 10 == 0 or i == total - 1):
                tracker.update(i + 1)
        
        if show_progress:
            tracker.complete()
        
        return all_labels


# -----------------------------------------------------
# ROBUST TEXT PREPROCESSOR
# -----------------------------------------------------
class RobustTextPreprocessor:
    """Handles all text preprocessing edge cases to prevent tokenization errors"""

    @staticmethod
    def clean_text(text, max_length=5000, min_length=10):
        """Clean and validate text to prevent tokenization errors"""
        # Handle None or NaN
        if pd.isna(text) or text is None:
            return None

        # If someone passed a list/tuple, join as text early (prevents nested tokenization)
        if isinstance(text, (list, tuple)):
            try:
                text = ' '.join(map(str, text))
            except Exception:
                return None

        # Convert to string if not already
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return None

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Handle empty or too short text
        if len(text.strip()) < min_length:
            return None

        # Handle nested structures (lists, dicts converted to string)
        if text.startswith('[') or text.startswith('{'):
            text = re.sub(r'[\[\]{}"\']', ' ', text)
            text = ' '.join(text.split())

        # Remove problematic characters
        text = text.replace('\x00', '')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')

        # Fix encoding issues
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            pass

        # Truncate if too long
        if len(text) > max_length:
            truncated = text[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.8:
                text = truncated[:last_period + 1]
            else:
                text = truncated + '...'

        text = text.strip()
        if len(text) < min_length:
            return None

        return text

    @staticmethod
    def preprocess_documents(documents, max_length=5000, min_length=10, show_warnings=True):
        """Preprocess a list of documents robustly"""
        cleaned_docs = []
        valid_indices = []

        stats = {
            'total': len(documents),
            'valid': 0,
            'empty_removed': 0,
            'too_short': 0,
            'too_long_truncated': 0,
            'type_converted': 0
        }

        for idx, doc in enumerate(documents):
            original_type = type(doc)
            original_length = len(str(doc)) if doc is not None else 0

            cleaned = RobustTextPreprocessor.clean_text(doc, max_length, min_length)

            if cleaned is not None:
                cleaned_docs.append(cleaned)
                valid_indices.append(idx)
                stats['valid'] += 1

                if original_type != str and not isinstance(doc, (list, tuple)):
                    stats['type_converted'] += 1
                if original_length > max_length:
                    stats['too_long_truncated'] += 1
            else:
                if pd.isna(doc) or doc is None or str(doc).strip() == '':
                    stats['empty_removed'] += 1
                elif len(str(doc).strip()) < min_length:
                    stats['too_short'] += 1

        if show_warnings and stats['total'] != stats['valid']:
            removed = stats['total'] - stats['valid']
            st.warning(f"⚠️ Preprocessing: {removed} documents removed ({removed/stats['total']*100:.1f}%)")

        return cleaned_docs, valid_indices, stats

# -----------------------------------------------------
# TOPIC MERGER FOR MINIMUM SIZE ENFORCEMENT
# -----------------------------------------------------
class TopicMerger:
    """Merge small topics to enforce minimum topic size"""

    @staticmethod
    def merge_small_topics(topics, embeddings, min_size=10):
        """Merge topics that are smaller than min_size with their nearest neighbors"""
        topics = np.array(topics)
        topic_counts = Counter(topics)

        small_topics = [t for t, count in topic_counts.items()
                        if t != -1 and count < min_size]

        if not small_topics:
            return topics

        # Calculate topic centroids
        topic_centroids = {}
        for topic in set(topics):
            if topic != -1:
                topic_mask = topics == topic
                if np.any(topic_mask):
                    topic_embeddings = embeddings[topic_mask]
                    if len(topic_embeddings) > 0 and not np.all(topic_embeddings == 0):
                        topic_centroids[topic] = np.mean(topic_embeddings, axis=0)

        # Merge small topics
        merged_topics = topics.copy()

        for small_topic in small_topics:
            if small_topic not in topic_centroids:
                continue

            small_centroid = topic_centroids[small_topic]
            min_distance = float('inf')
            best_merge_topic = None

            for topic, centroid in topic_centroids.items():
                if topic != small_topic and topic_counts[topic] >= min_size:
                    try:
                        distance = cosine(small_centroid, centroid)
                        if distance < min_distance:
                            min_distance = distance
                            best_merge_topic = topic
                    except:
                        continue

            if best_merge_topic is not None:
                merged_topics[merged_topics == small_topic] = best_merge_topic
                topic_counts[best_merge_topic] += topic_counts[small_topic]
                del topic_counts[small_topic]

        # Renumber topics
        unique_topics = sorted(set(merged_topics[merged_topics != -1]))
        topic_mapping = {old: new for new, old in enumerate(unique_topics)}
        topic_mapping[-1] = -1

        final_topics = np.array([topic_mapping.get(t, t) for t in merged_topics])

        return final_topics

# -----------------------------------------------------
# CATEGORY BALANCE ANALYZER
# -----------------------------------------------------
class CategoryBalanceAnalyzer:
    """Analyzes topic distribution and suggests splits for oversized categories"""

    def __init__(self, min_topic_ratio=0.30, ideal_max_ratio=0.20):
        self.min_topic_ratio = min_topic_ratio
        self.ideal_max_ratio = ideal_max_ratio

    def analyze_distribution(self, labels, include_outliers=False):
        """Analyze the distribution of documents across topics"""
        total_docs = len(labels)
        unique_topics = set(labels)

        if not include_outliers:
            labels_filtered = [l for l in labels if l != -1]
            total_docs = len(labels_filtered)
            unique_topics = set(labels_filtered)
        else:
            labels_filtered = labels

        if total_docs == 0:
            return {
                'balanced': False,
                'oversized_topics': [],
                'distribution': {},
                'warnings': ['No documents to analyze']
            }

        # Calculate distribution
        topic_counts = {}
        for topic in unique_topics:
            count = labels_filtered.count(topic)
            topic_counts[topic] = {
                'count': count,
                'ratio': count / total_docs
            }

        # Find oversized topics
        oversized_topics = []
        warnings_list = []

        for topic, stats in topic_counts.items():
            if stats['ratio'] > self.min_topic_ratio:
                oversized_topics.append({
                    'topic': topic,
                    'count': stats['count'],
                    'ratio': stats['ratio'],
                    'suggested_splits': max(2, int(stats['ratio'] / self.ideal_max_ratio))
                })
                warnings_list.append(
                    f"Topic {topic}: {stats['count']} docs ({stats['ratio']*100:.1f}%) - Consider splitting into {max(2, int(stats['ratio'] / self.ideal_max_ratio))} subtopics"
                )

        ratios = [stats['ratio'] for stats in topic_counts.values()]
        balance_score = 1 - (np.std(ratios) if len(ratios) > 1 else 0)

        return {
            'balanced': len(oversized_topics) == 0,
            'balance_score': balance_score,
            'oversized_topics': oversized_topics,
            'distribution': topic_counts,
            'warnings': warnings_list,
            'total_topics': len(unique_topics),
            'outlier_ratio': (len(labels) - len(labels_filtered)) / len(labels) if len(labels) > 0 else 0
        }

# -----------------------------------------------------
# OUTLIER REDUCTION STRATEGIES
# -----------------------------------------------------
class OutlierReducer:
    """Strategies to reduce outliers in topic modeling"""

    @staticmethod
    def suggest_parameters(num_docs, current_outlier_ratio, min_topic_size=10):
        """Suggest better parameters to reduce outliers"""
        suggestions = {
            'method': '',
            'parameters': {},
            'explanation': ''
        }

        if current_outlier_ratio > 0.3:
            # High outliers - use K-means
            suggestions['method'] = 'kmeans'
            max_topics = max(5, num_docs // min_topic_size)
            suggested_topics = min(max_topics, max(5, max(1, num_docs // 50)))

            suggestions['parameters'] = {
                'use_kmeans': True,
                'use_gpu_kmeans': True,
                'nr_topics': suggested_topics,
                'min_topic_size': min_topic_size,
                'merge_small_topics': True
            }
            suggestions['explanation'] = f"High outliers detected. K-means will assign all documents to topics. Topics smaller than {min_topic_size} will be merged."

        elif current_outlier_ratio > 0.15:
            # Moderate outliers - adjust HDBSCAN
            suggestions['method'] = 'hdbscan'
            suggestions['parameters'] = {
                'use_kmeans': False,
                'min_cluster_size': max(3, min_topic_size // 3),
                'min_samples': max(1, min_topic_size // 5),
                'prediction_data': True,
                'min_topic_size': min_topic_size
            }
            suggestions['explanation'] = f"Moderate outliers. Using lenient HDBSCAN parameters with min topic size of {min_topic_size}."

        else:
            # Low outliers - standard parameters
            suggestions['method'] = 'hdbscan'
            suggestions['parameters'] = {
                'use_kmeans': False,
                'min_cluster_size': min_topic_size,
                'min_samples': max(1, min_topic_size // 2),
                'prediction_data': False,
                'min_topic_size': min_topic_size
            }
            suggestions['explanation'] = f"Low outliers. Using standard HDBSCAN with min topic size of {min_topic_size}."

        return suggestions

# -----------------------------------------------------
# GPU-ACCELERATED K-MEANS WRAPPER
# -----------------------------------------------------
class GPUKMeans:
    """K-means clustering with GPU acceleration if available"""

    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

        if cuml_available and torch.cuda.is_available():
            try:
                from cuml.cluster import KMeans as cuKMeans
                self.model = cuKMeans(n_clusters=n_clusters, random_state=random_state)
                self.use_gpu = True
            except Exception:
                from sklearn.cluster import KMeans
                self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
                self.use_gpu = False
        else:
            from sklearn.cluster import KMeans
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
            self.use_gpu = False

    def fit(self, X):
        return self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X):
        return self.model.fit_predict(X)

# -----------------------------------------------------
# SAFE EMBEDDING MODEL
# -----------------------------------------------------
class SafeEmbeddingModel:
    """Wrapper for sentence transformer that handles edge cases"""

    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_name = model_name  # Store the model name
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.model.max_seq_length = min(self.model.max_seq_length, 512)

    def encode_safe(self, documents, batch_size=32, show_progress=True):
        """Safely encode documents with error handling"""
        if not documents:
            raise ValueError("No documents to encode")

        documents = [str(doc) if doc is not None else "" for doc in documents]
        non_empty_docs = [(i, doc) for i, doc in enumerate(documents) if doc.strip()]

        if not non_empty_docs:
            raise ValueError("All documents are empty after preprocessing")

        indices, clean_docs = zip(*non_empty_docs)

        try:
            embeddings = self.model.encode(
                list(clean_docs),
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            full_embeddings = np.zeros((len(documents), embeddings.shape[1]))
            for i, idx in enumerate(indices):
                full_embeddings[idx] = embeddings[i]

            return full_embeddings

        except RuntimeError as e:
            if "too large" in str(e).lower() or "out of memory" in str(e).lower():
                st.warning("Reducing batch size due to memory constraints...")
                return self.encode_safe(documents, batch_size=max(1, batch_size//2), show_progress=show_progress)
            elif "padding" in str(e).lower() or "truncation" in str(e).lower():
                st.warning("Processing documents individually due to length variance...")
                return self._encode_one_by_one(clean_docs, indices, len(documents))
            else:
                raise e

    def _encode_one_by_one(self, documents, indices, total_length):
        """Emergency fallback: encode documents one at a time"""
        embeddings_list = []

        progress_bar = st.progress(0)
        for i, doc in enumerate(documents):
            try:
                emb = self.model.encode(
                    [doc],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings_list.append(emb[0])
            except:
                embeddings_list.append(np.zeros(self.model.get_sentence_embedding_dimension()))

            progress_bar.progress((i + 1) / len(documents))

        progress_bar.empty()

        full_embeddings = np.zeros((total_length, len(embeddings_list[0])))
        for i, idx in enumerate(indices):
            full_embeddings[idx] = embeddings_list[i]

        return full_embeddings


# -----------------------------------------------------
# IMPROVED HUMAN-READABLE TOPIC LABELS
# -----------------------------------------------------
def _top_phrases(texts, ngram_range=(2,3), top_k=5, max_features=5000):
    """Return top_k high-signal phrases from texts using TF-IDF (prefers bigrams/trigrams)."""
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english",
            min_df=2
        )
        X = vec.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(vec.get_feature_names_out())
        order = np.argsort(scores)[::-1]
        phrases = [vocab[i] for i in order[:top_k]]
        return phrases
    except Exception:
        return []

def _clean_phrase(phrase):
    """Clean and normalize a phrase for use in labels."""
    # Remove punctuation but keep spaces
    phrase = re.sub(r'[^\w\s-]', ' ', phrase)
    # Normalize whitespace
    phrase = ' '.join(phrase.split())
    return phrase.strip()

def _to_title_case(text):
    """Smart title case that preserves acronyms and common patterns."""
    # Common acronyms and abbreviations to keep uppercase
    keep_upper = {
        "AI", "ML", "NLP", "API", "SQL", "GPU", "CPU", "FAQ", "KPI", "OKR", 
        "CRM", "IT", "HR", "PR", "SEO", "SEM", "ROI", "B2B", "B2C", "SaaS",
        "UI", "UX", "iOS", "AWS", "API", "REST", "HTTP", "JSON", "XML", "CSV",
        "PDF", "URL", "HTML", "CSS", "JS", "PHP", "SQL", "NoSQL", "CI", "CD"
    }
    
    # Words to keep lowercase
    keep_lower = {
        "a", "an", "and", "or", "but", "for", "nor", "on", "at", "to", "by",
        "in", "of", "the", "with", "from", "into", "onto", "upon", "as", "vs"
    }
    
    words = text.split()
    result = []
    
    for i, word in enumerate(words):
        word_clean = re.sub(r'[^\w-]', '', word)
        
        # First word or after certain punctuation - capitalize
        if i == 0:
            if word_clean.upper() in keep_upper:
                result.append(word_clean.upper())
            else:
                result.append(word_clean.capitalize())
        # Check if it's an acronym
        elif word_clean.upper() in keep_upper:
            result.append(word_clean.upper())
        # Check if it should be lowercase
        elif word_clean.lower() in keep_lower:
            result.append(word_clean.lower())
        # Default: capitalize
        else:
            result.append(word_clean.capitalize())
    
    return ' '.join(result)

def _create_descriptive_label(phrases_23, phrases_1, keywords, max_len=70):
    """
    Create a hierarchical, descriptive label with main category and specific details.
    Format: "Main Category - Specific Details"
    Examples:
    - "Customer Service - Response Times & Issues"
    - "Product Orders - Samsung Washer Delivery"
    - "Technical Support - Installation Problems"
    """
    # Clean all phrases
    clean_phrases_23 = [_clean_phrase(p) for p in phrases_23 if p and len(_clean_phrase(p)) > 3]
    clean_phrases_1 = [_clean_phrase(p) for p in phrases_1 if p and len(_clean_phrase(p)) > 2]

    # Parse keywords
    if isinstance(keywords, str):
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    else:
        keyword_list = list(keywords) if keywords else []

    # Build main category and details
    main_category = None
    details = None

    # PRIORITY 1: Use longest trigram as main category, and second as details
    if len(clean_phrases_23) >= 2:
        # First phrase is main category
        main_category = _to_title_case(clean_phrases_23[0])

        # Find complementary second phrase for details
        first_words = set(clean_phrases_23[0].lower().split())
        for second in clean_phrases_23[1:4]:
            second_words = set(second.lower().split())

            # Skip if too similar
            if second_words.issubset(first_words) or first_words.issubset(second_words):
                continue

            overlap = len(first_words & second_words)
            total = len(first_words | second_words)
            if overlap / total < 0.6:  # Less than 60% overlap
                details = _to_title_case(second)
                break

        # If no good second phrase, use keywords for details
        if not details and len(keyword_list) >= 2:
            kw_detail = ' '.join([_to_title_case(kw) for kw in keyword_list[:2] if kw.lower() not in main_category.lower()])
            if kw_detail:
                details = kw_detail

    # PRIORITY 2: Use first trigram as main, combine unigrams for details
    elif len(clean_phrases_23) >= 1:
        main_category = _to_title_case(clean_phrases_23[0])

        # Use unigrams that aren't in main category for details
        main_words = set(main_category.lower().split())
        detail_words = [w for w in clean_phrases_1[:4] if w.lower() not in main_words]
        if len(detail_words) >= 2:
            details = _to_title_case(' '.join(detail_words[:2]))
        elif len(keyword_list) >= 2:
            kw_detail = ' '.join([_to_title_case(kw) for kw in keyword_list[:2] if kw.lower() not in main_category.lower()])
            if kw_detail:
                details = kw_detail

    # PRIORITY 3: Build from unigrams
    elif len(clean_phrases_1) >= 3:
        # First 1-2 words as main category
        main_category = _to_title_case(' '.join(clean_phrases_1[:2]))
        # Next 1-2 words as details
        detail_words = clean_phrases_1[2:4]
        if detail_words:
            details = _to_title_case(' '.join(detail_words))

    # PRIORITY 4: Use keywords
    elif len(keyword_list) >= 3:
        main_category = _to_title_case(' '.join(keyword_list[:2]))
        details = _to_title_case(' '.join(keyword_list[2:4]))

    # Fallback to simple labels if hierarchical structure isn't possible
    if not main_category:
        if clean_phrases_23:
            main_category = _to_title_case(clean_phrases_23[0])
        elif clean_phrases_1:
            main_category = _to_title_case(' '.join(clean_phrases_1[:2]))
        elif keyword_list:
            main_category = _to_title_case(' '.join(keyword_list[:2]))
        else:
            main_category = "Miscellaneous Topic"

    # Ensure we have details - use remaining keywords if needed
    if not details:
        if len(keyword_list) >= 1:
            main_words = set(main_category.lower().split())
            detail_kws = [_to_title_case(kw) for kw in keyword_list[:3] if kw.lower() not in main_words]
            if detail_kws:
                details = ' '.join(detail_kws[:2])

        # Last resort: use "Related Topics" or part of main category
        if not details:
            if len(clean_phrases_1) > 0:
                main_words = set(main_category.lower().split())
                extra_words = [w for w in clean_phrases_1 if w.lower() not in main_words]
                if extra_words:
                    details = _to_title_case(extra_words[0])
                else:
                    details = "General Topics"
            else:
                details = "General Topics"

    # Construct final hierarchical label
    label = f"{main_category} - {details}"

    # Truncate if too long
    if len(label) > max_len:
        # Try to shorten details first
        max_details_len = max_len - len(main_category) - 3  # 3 for " - "
        if max_details_len > 10:
            details = details[:max_details_len].rstrip() + "…"
            label = f"{main_category} - {details}"
        else:
            label = label[:max_len].rstrip() + "…"

    return label

# -----------------------------------------------------
# LOCAL LLM FOR BETTER TOPIC LABELS
# -----------------------------------------------------
def generate_topic_label_with_llm(topic_docs, keywords_str, llm_model=None, max_docs=10):
    """
    Generate a descriptive topic label using a local LLM.
    
    Args:
        topic_docs: List of documents in the topic
        keywords_str: Comma-separated keywords
        llm_model: Loaded HuggingFace model and tokenizer (tuple)
        max_docs: Number of sample documents to show the LLM
    
    Returns:
        str: Generated topic label
    """
    if llm_model is None:
        # Fallback to non-LLM method
        return make_human_label(topic_docs, keywords_str)
    
    try:
        model, tokenizer = llm_model
        
        # Sample documents for context
        sample_docs = topic_docs[:max_docs]
        docs_text = "\n".join([f"- {doc[:200]}" for doc in sample_docs])
        
        # Create prompt for LLM
        prompt = f"""Based on these sample documents and keywords, generate a short, descriptive topic label (2-5 words).

Keywords: {keywords_str}

Sample documents:
{docs_text}

Generate only the topic label, nothing else. Make it specific and descriptive.
Topic label:"""

        # Generate label
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        label = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the label part (after the prompt)
        label = label.split("Topic label:")[-1].strip()
        
        # Clean up the label
        label = label.split("\n")[0]  # Take first line only
        label = label.strip('"\'')  # Remove quotes
        label = label[:70]  # Limit length
        
        # If label is too short or looks bad, fallback
        if len(label) < 3 or len(label.split()) > 8:
            return make_human_label(topic_docs, keywords_str)
        
        return label
        
    except Exception as e:
        # Fallback to non-LLM method on any error
        return make_human_label(topic_docs, keywords_str)


def clear_gpu_memory():
    """Clear GPU memory cache to free up space"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()


@st.cache_resource
def load_local_llm(model_name, force_cpu=False):
    """
    Load a local LLM for topic labeling with intelligent GPU/CPU handling.
    
    Smart fallback strategy:
    1. Try GPU if available and not forced to CPU
    2. If GPU OOM, clear cache and try CPU with system RAM
    3. Use appropriate precision for each device
    
    Recommended models:
    - "microsoft/Phi-3-mini-4k-instruct" (3.8GB, fast, good for 16GB GPU, 8 docs)
    - "microsoft/Phi-3-mini-128k-instruct" (3.8GB, fast, massive context, 50+ docs)
    - "mistralai/Mistral-7B-Instruct-v0.2" (14GB, better quality, needs 24GB+ GPU, 15 docs)
    - "HuggingFaceH4/zephyr-7b-beta" (14GB, good quality, needs 20GB+ GPU, 15 docs)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import psutil
        
        # Check available system RAM
        system_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        st.info(f"🔄 Loading {model_name}... This may take a few minutes the first time.")
        
        # Load tokenizer (small, always works)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = None
        device_used = None
        
        # Strategy 1: Try GPU first (if available and not forced to CPU)
        if torch.cuda.is_available() and not force_cpu:
            try:
                # Check available GPU memory
                gpu_free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                st.info(f"📊 Available GPU memory: {gpu_free_gb:.1f} GB")
                
                # Only try GPU if we have at least 4GB free (minimum for small models)
                if gpu_free_gb >= 4.0:
                    st.info("🎮 Attempting to load LLM on GPU...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,  # Half precision for GPU
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        max_memory={0: f"{int(gpu_free_gb * 0.85)}GB"}  # Use 85% of available
                    )
                    device_used = "GPU"
                    st.success(f"✅ LLM loaded on GPU ({gpu_free_gb:.1f}GB available)")
                else:
                    st.warning(f"⚠️ Only {gpu_free_gb:.1f}GB GPU memory available - switching to CPU")
                    raise RuntimeError("Insufficient GPU memory, falling back to CPU")
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    st.warning("⚠️ GPU out of memory - clearing cache and trying CPU...")
                    torch.cuda.empty_cache()
                    model = None  # Will fall through to CPU loading
                else:
                    raise e
        
        # Strategy 2: Load on CPU with system RAM
        if model is None:
            st.info("💻 Loading LLM on CPU using system RAM...")
            
            # Check if we have enough system RAM (need ~8GB minimum for 7B model)
            if system_ram_gb < 12:
                st.error(f"❌ Insufficient system RAM ({system_ram_gb:.1f}GB). Need at least 12GB for CPU inference.")
                st.info("💡 Consider using Phi-3-mini-4k-instruct or Phi-3-mini-128k-instruct (needs ~6GB)")
                return None
            
            try:
                # Load in 8-bit on CPU for efficiency (if bitsandbytes available)
                try:
                    import bitsandbytes
                    st.info("Using 8-bit quantization for CPU efficiency...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        load_in_8bit=True,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                except ImportError:
                    # Fall back to float32 on CPU
                    st.info("Loading in float32 (install bitsandbytes for faster CPU inference)")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    model = model.to('cpu')
                
                device_used = "CPU"
                st.success(f"✅ LLM loaded on CPU (using system RAM: {system_ram_gb:.1f}GB available)")
                st.info("⚡ Note: CPU inference is slower than GPU but will work reliably")
                
            except Exception as cpu_error:
                st.error(f"❌ Failed to load on CPU: {str(cpu_error)}")
                st.info("💡 Try a smaller model or ensure you have enough system RAM")
                return None
        
        # Verify model loaded successfully
        if model is None:
            st.error("❌ Failed to load LLM on both GPU and CPU")
            return None
        
        # Display final configuration
        st.success(f"🎯 LLM ready on {device_used}")
        if device_used == "CPU":
            st.info("💡 Pro tip: Close other applications to free up system RAM for faster inference")
        
        return (model, tokenizer)
        
    except Exception as e:
        st.error(f"❌ Failed to load LLM: {str(e)}")
        st.info("📦 Make sure you have transformers installed: pip install transformers accelerate")
        if "out of memory" in str(e).lower():
            st.error("💥 Out of memory error. Try one of these solutions:")
            st.markdown("""
            1. **Use a smaller model or Phi-128k**: Try 'microsoft/Phi-3-mini-4k-instruct' or 'microsoft/Phi-3-mini-128k-instruct' (only 3.8GB)
            2. **Force CPU mode**: Check 'Force CPU for LLM' in advanced settings
            3. **Free up memory**: Close other applications
            4. **Use Conservative profile**: Reduces memory usage
            5. **Skip LLM**: Uncheck 'Use LLM for labels' (uses TF-IDF fallback)
            """)
        return None


def make_human_label(topic_docs, fallback_keywords, max_len=70):
    """
    Build a unique, descriptive topic label based on the actual content.
    
    Examples of output:
    - "Product Launch Strategy & Marketing"
    - "Customer Service Response Times"
    - "Technical Documentation Updates"
    - "Sales Pipeline Management"
    - "Employee Onboarding Process"
    
    NOT generic categories like "Customer Support" or "Marketing"
    """
    # Extract phrases using TF-IDF with different n-gram ranges
    phrases_23 = _top_phrases(topic_docs, (2, 3), top_k=5)  # Bigrams and trigrams
    phrases_1 = _top_phrases(topic_docs, (1, 1), top_k=10)   # Unigrams
    
    # Create the descriptive label
    label = _create_descriptive_label(phrases_23, phrases_1, fallback_keywords, max_len)
    
    # Final cleanup
    label = re.sub(r'\s+', ' ', label).strip()
    if len(label) > max_len:
        label = label[:max_len].rstrip() + "…"
    
    return label

# -----------------------------------------------------
# FAST RECLUSTERING ENGINE
# -----------------------------------------------------
class FastReclusterer:
    """Fast reclustering using pre-computed embeddings"""

    def __init__(self, documents, embeddings, umap_embeddings=None, llm_model_name=None):
        self.documents = documents
        self.embeddings = embeddings
        self.umap_embeddings = umap_embeddings
        self.use_gpu = torch.cuda.is_available() and cuml_available
        self.llm_model = None  # Will be set if LLM labeling is enabled
        self.llm_model_name = llm_model_name  # Store model name for dynamic doc count

    def recluster(self, n_topics, min_topic_size=10, use_reduced=True, method='kmeans'):
        """Quickly recluster documents into new topics"""
        clustering_embeddings = self.umap_embeddings if (use_reduced and self.umap_embeddings is not None) else self.embeddings

        valid_mask = np.any(clustering_embeddings != 0, axis=1)
        valid_embeddings = clustering_embeddings[valid_mask]

        if len(valid_embeddings) == 0:
            st.error("No valid embeddings for clustering!")
            return None, None

        try:
            if method == 'kmeans':
                if self.use_gpu:
                    try:
                        from cuml.cluster import KMeans as cuKMeans
                        clusterer = cuKMeans(n_clusters=min(n_topics, len(valid_embeddings)), random_state=42)
                    except:
                        clusterer = KMeans(n_clusters=min(n_topics, len(valid_embeddings)), random_state=42)
                else:
                    clusterer = KMeans(n_clusters=min(n_topics, len(valid_embeddings)), random_state=42)

                valid_topics = clusterer.fit_predict(valid_embeddings)
            else:
                min_cluster_size = max(min_topic_size, len(self.documents) // (max(1, n_topics) * 2))
                min_cluster_size = min(min_cluster_size, max(2, len(valid_embeddings) // 2))

                if self.use_gpu:
                    clusterer = cumlHDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=max(1, min_cluster_size // 5)
                    )
                else:
                    clusterer = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=max(1, min_cluster_size // 5)
                    )

                valid_topics = clusterer.fit_predict(valid_embeddings)

            topics = np.full(len(clustering_embeddings), -1)
            topics[valid_mask] = valid_topics

        except Exception as e:
            st.error(f"Clustering error: {str(e)}")
            return None, None

        if min_topic_size > 2:
            topics = TopicMerger.merge_small_topics(topics, self.embeddings, min_topic_size)

        topic_info = self._extract_topic_keywords(topics)

        return topics, topic_info

    def _extract_topic_keywords(self, topics, top_n_words=10):
        """⚡ OPTIMIZED: Extract keywords and generate human-readable labels with adaptive parallel batching"""
        # Group documents by topic (with document indices for FAISS)
        topics_dict = {}
        topic_doc_indices = {}  # Track which embedding indices belong to each topic
        for idx, topic in enumerate(topics):
            topics_dict.setdefault(topic, [])
            topic_doc_indices.setdefault(topic, [])
            if idx < len(self.documents):
                topics_dict[topic].append(self.documents[idx])
                topic_doc_indices[topic].append(idx)
        
        # Handle outliers
        topic_info_list = []
        if -1 in topics_dict:
            topic_info_list.append({
                'Topic': -1,
                'Count': len(topics_dict[-1]),
                'Keywords': 'Outliers',
                'Human_Label': 'Outliers',
                'Name': 'Outliers'
            })
        
        # PHASE 1: Extract keywords for all topics (fast, no LLM)
        keywords_dict = {}
        for topic_id in sorted(topics_dict.keys()):
            if topic_id == -1:
                continue
            
            topic_docs = topics_dict[topic_id]
            
            try:
                sample_size = min(200, len(topic_docs))
                topic_text = ' '.join(topic_docs[:sample_size])
                words = topic_text.lower().split()
                word_counts = Counter(words)
                
                common_words = {
                    'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','as',
                    'is','was','are','were','be','have','has','had','do','does','did','will','would','could',
                    'should','may','might','must','shall','can','need','it','this','that','these','those',
                    'i','you','he','she','we','they'
                }
                
                filtered = {w: c for w, c in word_counts.items() if w not in common_words and len(w) > 2}
                top_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
                keywords = [w for w, _ in top_words] or ['topic', str(topic_id)]
                keywords_str = ', '.join(keywords[:5])
            except Exception:
                keywords_str = f'topic_{topic_id}'
            
            keywords_dict[topic_id] = keywords_str

        # PHASE 2: Generate TF-IDF hierarchical labels (ALWAYS - fast and reliable)
        st.info("📝 Generating hierarchical labels using TF-IDF...")
        labels_dict = {}

        if len(topics_dict) > 100:
            progress_bar = st.progress(0.0)

        for i, (topic_id, docs) in enumerate(topics_dict.items()):
            if topic_id != -1:
                labels_dict[topic_id] = make_human_label(docs, keywords_dict[topic_id])

                if len(topics_dict) > 100:
                    progress_bar.progress((i + 1) / len(topics_dict))

        if len(topics_dict) > 100:
            progress_bar.empty()

        # Deduplicate TF-IDF labels
        labels_dict = deduplicate_labels_globally(labels_dict, keywords_dict, topics_dict)

        st.success(f"✅ Generated {len(labels_dict)} unique hierarchical labels using TF-IDF")

        # PHASE 3: Optional LLM Analysis (BATCHED for actual speedup)
        llm_analysis_dict = {}
        llm_success_count = 0
        llm_fallback_count = 0

        if self.llm_model is not None:
            num_topics = len(topics_dict) - (1 if -1 in topics_dict else 0)

            # Batch size: process 5 topics at once (optimal for most LLMs)
            batch_size = 5
            st.info(f"🤖 Generating LLM analysis for {num_topics} topics using batch processing (5 topics/batch)...")

            # Get FAISS index for document selection
            faiss_index = st.session_state.get('faiss_index')
            embeddings_for_analysis = self.embeddings if hasattr(self, 'embeddings') else None

            # Create progress tracking
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            # Prepare all topics with FAISS-selected documents
            all_topics_prepared = []
            for topic_id, docs in topics_dict.items():
                if topic_id == -1:
                    continue

                doc_indices = topic_doc_indices.get(topic_id, [])
                label = labels_dict.get(topic_id, f"Topic {topic_id}")

                # Select representative documents using FAISS
                if faiss_index is not None and embeddings_for_analysis is not None and len(doc_indices) > 0:
                    try:
                        topic_embeddings = embeddings_for_analysis[doc_indices]
                        centroid = np.mean(topic_embeddings, axis=0).reshape(1, -1).astype('float32')
                        _, indices = faiss_index.search(centroid, min(5, len(doc_indices)))

                        idx_to_pos = {embedding_idx: pos for pos, embedding_idx in enumerate(doc_indices)}
                        selected_docs = []
                        for idx in indices[0]:
                            if idx in idx_to_pos:
                                selected_docs.append(docs[idx_to_pos[idx]])
                                if len(selected_docs) >= 5:
                                    break
                    except Exception:
                        selected_docs = docs[:5]
                else:
                    selected_docs = docs[:5]

                all_topics_prepared.append({
                    'topic_id': topic_id,
                    'label': label,
                    'docs': selected_docs
                })

            # Process in batches
            num_batches = (len(all_topics_prepared) + batch_size - 1) // batch_size
            processed_count = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_topics_prepared))
                batch = all_topics_prepared[start_idx:end_idx]

                # Update progress
                status_text.info(f"🔄 Analyzing batch {batch_idx+1}/{num_batches} ({len(batch)} topics)...")

                # Process batch with LLM
                batch_results = generate_batch_llm_analysis(batch, self.llm_model)

                # Collect results
                for topic_item in batch:
                    topic_id = topic_item['topic_id']
                    if topic_id in batch_results and batch_results[topic_id]:
                        llm_analysis_dict[topic_id] = batch_results[topic_id]
                        llm_success_count += 1
                    else:
                        llm_analysis_dict[topic_id] = "No analysis available"
                        llm_fallback_count += 1

                    processed_count += 1

                # Update progress bar
                progress = processed_count / num_topics
                progress_bar.progress(progress)
                status_text.info(f"🔄 Analyzed {processed_count}/{num_topics} topics")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show LLM analysis statistics
            if llm_success_count > 0:
                success_pct = (llm_success_count / num_topics) * 100
                st.success(
                    f"✅ **LLM Analysis Complete:** {llm_success_count}/{num_topics} topics ({success_pct:.1f}% success)"
                )
                if llm_fallback_count > 0:
                    st.warning(f"⚠️ {llm_fallback_count} topics failed LLM analysis")
            else:
                st.warning("⚠️ LLM analysis failed for all topics")

        # PHASE 4: Build final topic info dataframe
        for topic_id in sorted(topics_dict.keys()):
            if topic_id == -1:
                continue

            human_label = labels_dict.get(topic_id, f"Topic {topic_id}")
            llm_analysis = llm_analysis_dict.get(topic_id, "")  # Empty if no LLM analysis

            topic_info_list.append({
                'Topic': topic_id,
                'Count': len(topics_dict[topic_id]),
                'Keywords': keywords_dict[topic_id],
                'Human_Label': human_label,
                'Name': human_label,
                'LLM_Analysis': llm_analysis
            })

        return pd.DataFrame(topic_info_list)


# -----------------------------------------------------
# GUARANTEE COLUMNS EXIST ON topic_info
# -----------------------------------------------------
def normalize_topic_info(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure topic_info has Human_Label, Keywords, Count, and LLM_Analysis columns."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Topic", "Human_Label", "Keywords", "Count", "LLM_Analysis"])
    df = df.copy()

    # LLM_Analysis
    if 'LLM_Analysis' not in df.columns:
        df['LLM_Analysis'] = ""

    # Keywords
    if "Keywords" not in df.columns:
        if "Representation" in df.columns:
            df["Keywords"] = df["Representation"].apply(
                lambda x: ", ".join(x[:5]) if isinstance(x, list) else (str(x) if pd.notna(x) else "")
            )
        else:
            df["Keywords"] = ""

    # Human_Label
    if "Human_Label" not in df.columns or df["Human_Label"].isna().any():
        if "Human_Label" not in df.columns:
            df["Human_Label"] = ""
        def _build_label(row):
            if isinstance(row.get("Human_Label"), str) and row["Human_Label"].strip():
                return row["Human_Label"]
            if isinstance(row.get("Name"), str) and row["Name"].strip():
                return row["Name"]
            kw = row.get("Keywords", "")
            # fall back to keywords-only label (documents unavailable here)
            return make_human_label([], kw)
        df["Human_Label"] = df.apply(_build_label, axis=1)

    # Count
    if "Count" not in df.columns:
        if "Document Count" in df.columns:
            df["Count"] = df["Document Count"]
        else:
            # compute from current topics if available
            if st.session_state.get("current_topics") is not None:
                counts = Counter(st.session_state.current_topics.tolist())
                df["Count"] = df["Topic"].map(counts).fillna(0).astype(int)
            else:
                df["Count"] = 0

    # Ensure Topic exists
    if "Topic" not in df.columns and "topic" in df.columns:
        df["Topic"] = df["topic"]

    return df


# -----------------------------------------------------
# STANDALONE LLM ANALYSIS FOR EXISTING TOPICS
# -----------------------------------------------------
def run_llm_analysis_on_topics(topics, topic_info, documents, embeddings, llm_model):
    """
    Run LLM analysis on already-clustered topics.
    This allows adding AI insights after clustering without reclustering.

    Args:
        topics: Array of topic assignments
        topic_info: DataFrame with topic information
        documents: List of document texts
        embeddings: Document embeddings (for FAISS selection)
        llm_model: Loaded LLM model tuple (model, tokenizer)

    Returns:
        Updated topic_info DataFrame with LLM_Analysis column
    """
    # Group documents by topic
    topics_dict = {}
    topic_doc_indices = {}
    for idx, topic in enumerate(topics):
        topics_dict.setdefault(topic, [])
        topic_doc_indices.setdefault(topic, [])
        if idx < len(documents):
            topics_dict[topic].append(documents[idx])
            topic_doc_indices[topic].append(idx)

    # Get labels from topic_info
    labels_dict = dict(zip(topic_info['Topic'], topic_info['Human_Label']))

    # Run BATCHED LLM analysis for actual speedup
    llm_analysis_dict = {}
    llm_success_count = 0
    llm_fallback_count = 0

    num_topics = len(topics_dict) - (1 if -1 in topics_dict else 0)

    # Batch size: process 5 topics at once
    batch_size = 5
    st.info(f"🤖 Generating LLM analysis for {num_topics} topics using batch processing (5 topics/batch)...")

    # Get FAISS index
    faiss_index = st.session_state.get('faiss_index')

    # Progress tracking
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # Prepare all topics with FAISS-selected documents
    all_topics_prepared = []
    for topic_id, docs in topics_dict.items():
        if topic_id == -1:
            continue

        doc_indices = topic_doc_indices.get(topic_id, [])
        label = labels_dict.get(topic_id, f"Topic {topic_id}")

        # Select docs with FAISS
        if faiss_index is not None and embeddings is not None and len(doc_indices) > 0:
            try:
                topic_embeddings = embeddings[doc_indices]
                centroid = np.mean(topic_embeddings, axis=0).reshape(1, -1).astype('float32')
                _, indices = faiss_index.search(centroid, min(5, len(doc_indices)))

                idx_to_pos = {embedding_idx: pos for pos, embedding_idx in enumerate(doc_indices)}
                selected_docs = []
                for idx in indices[0]:
                    if idx in idx_to_pos:
                        selected_docs.append(docs[idx_to_pos[idx]])
                        if len(selected_docs) >= 5:
                            break
            except Exception:
                selected_docs = docs[:5]
        else:
            selected_docs = docs[:5]

        all_topics_prepared.append({
            'topic_id': topic_id,
            'label': label,
            'docs': selected_docs
        })

    # Process in batches
    num_batches = (len(all_topics_prepared) + batch_size - 1) // batch_size
    processed_count = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_topics_prepared))
        batch = all_topics_prepared[start_idx:end_idx]

        # Update progress
        status_text.info(f"🔄 Analyzing batch {batch_idx+1}/{num_batches} ({len(batch)} topics)...")

        # Process batch with LLM
        batch_results = generate_batch_llm_analysis(batch, llm_model)

        # Collect results
        for topic_item in batch:
            topic_id = topic_item['topic_id']
            if topic_id in batch_results and batch_results[topic_id]:
                llm_analysis_dict[topic_id] = batch_results[topic_id]
                llm_success_count += 1
            else:
                llm_analysis_dict[topic_id] = "No analysis available"
                llm_fallback_count += 1

            processed_count += 1

        # Update progress bar
        progress = processed_count / num_topics
        progress_bar.progress(progress)
        status_text.info(f"🔄 Analyzed {processed_count}/{num_topics} topics")

    # Clear progress
    progress_bar.empty()
    status_text.empty()

    # Show stats
    if llm_success_count > 0:
        success_pct = (llm_success_count / num_topics) * 100
        st.success(f"✅ **LLM Analysis Complete:** {llm_success_count}/{num_topics} topics ({success_pct:.1f}% success)")
        if llm_fallback_count > 0:
            st.warning(f"⚠️ {llm_fallback_count} topics failed LLM analysis")
    else:
        st.warning("⚠️ LLM analysis failed for all topics")

    # Update topic_info with LLM_Analysis
    topic_info = topic_info.copy()
    topic_info['LLM_Analysis'] = topic_info['Topic'].map(llm_analysis_dict).fillna("")

    return topic_info


# -----------------------------------------------------
# MAIN APPLICATION FUNCTIONS
# -----------------------------------------------------
def check_gpu_capabilities():
    """Check and return GPU/CUDA capabilities"""
    capabilities = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'device_name': None,
        'gpu_memory_total': None,
        'gpu_memory_free': None,
        'cupy_available': cupy_available,
        'cuml_available': cuml_available,
        'faiss_available': faiss_available,
        'faiss_gpu_available': faiss_gpu_available,
        'accelerate_available': accelerate_available
    }

    if torch.cuda.is_available():
        capabilities['device_count'] = torch.cuda.device_count()
        capabilities['device_name'] = torch.cuda.get_device_name(0)

        try:
            mem_info = torch.cuda.mem_get_info(0)
            capabilities['gpu_memory_free'] = f"{mem_info[0] / 1024**3:.1f} GB"
            capabilities['gpu_memory_total'] = f"{mem_info[1] / 1024**3:.1f} GB"
        except Exception:
            pass

    return capabilities

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def load_embedding_model(model_name):
    """Load and cache the embedding model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    # keep SentenceTransformers' smart batching; just set a safe cap
    model.max_seq_length = 512
    # DO NOT compile model.encode; if you want compile, compile the inner HF module only (optional)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            first = model._first_module()
            if hasattr(first, "auto_model"):
                first.auto_model = torch.compile(first.auto_model)
        except Exception:
            pass
    return model

@st.cache_data
def compute_umap_embeddings(embeddings, n_neighbors=15, n_components=5):
    """Compute and cache UMAP reduced embeddings"""
    valid_mask = np.any(embeddings != 0, axis=1)
    if np.sum(valid_mask) < 10:
        return None

    if cuml_available and torch.cuda.is_available():
        reducer = cumlUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            random_state=42
        )
    else:
        reducer = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            random_state=42
        )

    umap_embeddings = np.zeros((len(embeddings), n_components))
    umap_embeddings[valid_mask] = reducer.fit_transform(embeddings[valid_mask])

    return umap_embeddings


# -----------------------------------------------------
# FAISS INDEXING FOR RAG CHAT
# -----------------------------------------------------
def build_faiss_index(embeddings):
    """Build FAISS index from embeddings for fast similarity search"""
    if not faiss_available:
        st.warning("⚠️ FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
        return None

    try:
        dimension = embeddings.shape[1]

        # Use GPU if available
        if faiss_gpu_available:
            st.info("🎮 Building FAISS index on GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dimension)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(embeddings.astype('float32'))
            return gpu_index
        else:
            st.info("💻 Building FAISS index on CPU...")
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            return index
    except Exception as e:
        st.error(f"❌ Failed to build FAISS index: {str(e)}")
        return None


def retrieve_relevant_documents(query, faiss_index, embeddings, documents, safe_model, top_k=5):
    """Retrieve top-k most relevant documents using FAISS"""
    if faiss_index is None:
        return []

    try:
        # Encode query
        query_embedding = safe_model.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].reshape(1, -1).astype('float32')

        # Search FAISS index
        distances, indices = faiss_index.search(query_embedding, top_k)

        # Get documents
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(documents):
                results.append({
                    'document': documents[idx],
                    'distance': float(distances[0][i]),
                    'index': int(idx)
                })

        return results
    except Exception as e:
        st.error(f"❌ Document retrieval error: {str(e)}")
        return []


def generate_rag_response(user_query, retrieved_docs, topic_info_df, topics, llm_model, current_topic_id=None):
    """Generate response using LLM with retrieved documents as context (RAG)"""
    if llm_model is None:
        return "❌ LLM not loaded. Please enable LLM in the sidebar and reload."

    try:
        model, tokenizer = llm_model

        # Build context from retrieved documents
        context_parts = []
        context_parts.append(f"You are analyzing topic modeling results with {len(topics)} topics.")

        if current_topic_id is not None and current_topic_id != -1:
            topic_row = topic_info_df[topic_info_df['Topic'] == current_topic_id]
            if len(topic_row) > 0:
                row = topic_row.iloc[0]
                context_parts.append(f"\nCurrently viewing Topic {current_topic_id}: {row['Human_Label']}")
                context_parts.append(f"Keywords: {row['Keywords']}")

        context_parts.append("\nRelevant Documents:")
        for i, doc_info in enumerate(retrieved_docs[:5], 1):
            doc_preview = doc_info['document'][:300] + "..." if len(doc_info['document']) > 300 else doc_info['document']
            context_parts.append(f"\n[Doc {i}]: {doc_preview}")

        context = "\n".join(context_parts)

        # Create prompt
        prompt = f"""{context}

User Question: {user_query}

Based on the relevant documents and topic information above, provide a clear, helpful answer. Be concise and specific.

Answer:"""

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        return response

    except Exception as e:
        st.error(f"❌ LLM generation error: {str(e)}")
        return f"Error generating response: {str(e)}"


def generate_chat_response(user_query, context, topic_info_df, topics, processed_df, current_topic_id=None, use_rag=False, llm_model=None, faiss_index=None, embeddings=None, documents=None, safe_model=None):
    """
    Generate a chat response based on user query and topic data.
    Supports both rule-based and RAG (Retrieval-Augmented Generation) modes.

    Args:
        current_topic_id: The currently selected topic in the browser (or None if viewing all)
        use_rag: Whether to use RAG with FAISS + LLM
        llm_model: Tuple of (model, tokenizer) for RAG
        faiss_index: FAISS index for document retrieval
        embeddings: Document embeddings
        documents: List of all documents
        safe_model: Embedding model for query encoding
    """
    query_lower = user_query.lower()

    # RAG MODE: Use FAISS + LLM for intelligent responses
    if use_rag and llm_model and faiss_index and documents and safe_model:
        st.info("🤖 Using RAG mode (FAISS + LLM)...")

        # Retrieve relevant documents
        retrieved_docs = retrieve_relevant_documents(
            user_query,
            faiss_index,
            embeddings,
            documents,
            safe_model,
            top_k=5
        )

        if retrieved_docs:
            st.caption(f"📚 Retrieved {len(retrieved_docs)} relevant documents")

            # Generate response using LLM with retrieved context
            response = generate_rag_response(
                user_query,
                retrieved_docs,
                topic_info_df,
                topics,
                llm_model,
                current_topic_id
            )

            # Add source references
            response += "\n\n**Sources:**\n"
            for i, doc in enumerate(retrieved_docs[:3], 1):
                doc_preview = doc['document'][:100] + "..." if len(doc['document']) > 100 else doc['document']
                response += f"{i}. {doc_preview}\n"

            return response
        else:
            st.warning("⚠️ No relevant documents found, using rule-based fallback")

    # RULE-BASED MODE (fallback or default)


    # Handle queries about the current topic
    if current_topic_id is not None and any(phrase in query_lower for phrase in ['current topic', 'this topic', 'about this']):
        topic_row = topic_info_df[topic_info_df['Topic'] == current_topic_id]
        if len(topic_row) > 0:
            row = topic_row.iloc[0]
            pct = (row['Count'] / len(processed_df)) * 100
            response = f"🔍 **Currently Viewing: Topic {current_topic_id}**\n\n"
            response += f"**Label:** {row['Human_Label']}\n\n"
            response += f"**Statistics:**\n"
            response += f"- Documents: {row['Count']:,} ({pct:.1f}% of total)\n"
            response += f"- Keywords: {row['Keywords']}\n\n"
            response += "💡 This hierarchical label shows both the main category and specific details separated by ' - '."
            return response

    # Handle different types of queries
    if any(word in query_lower for word in ['hello', 'hi', 'hey']):
        greeting = "👋 Hello! I'm here to help you explore your topic modeling results."
        if current_topic_id is not None:
            human_label = topic_info_df[topic_info_df['Topic'] == current_topic_id].iloc[0]['Human_Label'] if len(topic_info_df[topic_info_df['Topic'] == current_topic_id]) > 0 else f"Topic {current_topic_id}"
            greeting += f"\n\nYou're currently viewing **Topic {current_topic_id}: {human_label}**.\n\nWhat would you like to know?"
        else:
            greeting += " What would you like to know?"
        return greeting

    elif any(word in query_lower for word in ['how many', 'number of', 'count']):
        if 'topic' in query_lower:
            num_topics = len([t for t in topics if t != -1])
            return f"📊 I found **{num_topics} unique topics** in your data. The topics are identified with hierarchical labels showing both the main category and specific details."

        elif 'document' in query_lower or 'doc' in query_lower:
            if current_topic_id is not None:
                topic_row = topic_info_df[topic_info_df['Topic'] == current_topic_id]
                if len(topic_row) > 0:
                    count = topic_row.iloc[0]['Count']
                    return f"📄 The current topic has **{count:,} documents**. Your entire dataset contains **{len(processed_df):,} documents** in total."
            return f"📄 Your dataset contains **{len(processed_df):,} documents** in total."

    elif 'largest' in query_lower or 'biggest' in query_lower or 'most common' in query_lower:
        top_5 = topic_info_df.nlargest(5, 'Count')[['Topic', 'Human_Label', 'Count']]
        response = "📈 **Top 5 Largest Topics:**\n\n"
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            pct = (row['Count'] / len(processed_df)) * 100
            response += f"{i}. **{row['Human_Label']}** (Topic {row['Topic']})\n"
            response += f"   - {row['Count']:,} documents ({pct:.1f}% of total)\n\n"
        return response

    elif 'smallest' in query_lower or 'least common' in query_lower:
        bottom_5 = topic_info_df.nsmallest(5, 'Count')[['Topic', 'Human_Label', 'Count']]
        response = "📉 **5 Smallest Topics:**\n\n"
        for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
            pct = (row['Count'] / len(processed_df)) * 100
            response += f"{i}. **{row['Human_Label']}** (Topic {row['Topic']})\n"
            response += f"   - {row['Count']:,} documents ({pct:.1f}% of total)\n\n"
        return response

    elif re.search(r'topic\s+(\d+)', query_lower):
        # Extract topic number
        topic_match = re.search(r'topic\s+(\d+)', query_lower)
        topic_num = int(topic_match.group(1))

        topic_row = topic_info_df[topic_info_df['Topic'] == topic_num]
        if len(topic_row) == 0:
            return f"❌ Sorry, I couldn't find Topic {topic_num}. Please check if this topic number exists in your data."

        row = topic_row.iloc[0]
        pct = (row['Count'] / len(processed_df)) * 100
        response = f"🔍 **Topic {topic_num}: {row['Human_Label']}**\n\n"
        response += f"- **Documents:** {row['Count']:,} ({pct:.1f}% of total)\n"
        response += f"- **Keywords:** {row['Keywords']}\n\n"
        response += "💡 This hierarchical label shows the main category and specific details separated by ' - '."
        return response

    elif any(word in query_lower for word in ['main theme', 'overview', 'summary', 'what are']):
        num_topics = len([t for t in topics if t != -1])
        top_3 = topic_info_df.nlargest(3, 'Count')[['Human_Label', 'Count']]

        response = f"📊 **Topic Modeling Overview:**\n\n"
        response += f"Your data contains **{len(processed_df):,} documents** organized into **{num_topics} topics**.\n\n"
        response += "**Top 3 Most Common Themes:**\n\n"
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            pct = (row['Count'] / len(processed_df)) * 100
            response += f"{i}. {row['Human_Label']} - {row['Count']:,} docs ({pct:.1f}%)\n"

        response += f"\n💡 All labels now use a hierarchical format: 'Main Category - Specific Details' for better clarity!"
        return response

    elif 'hierarchical' in query_lower or 'hierarchy' in query_lower or 'label format' in query_lower:
        return ("📋 **Hierarchical Label Format:**\n\n"
                "All topic labels now follow the format: **'Main Category - Specific Details'**\n\n"
                "**Examples:**\n"
                "- 'Customer Service - Response Time Issues'\n"
                "- 'Product Orders - Samsung Washer Delivery'\n"
                "- 'Technical Support - Installation Problems'\n\n"
                "This two-tier structure helps you quickly understand both the general theme and the specific focus of each topic!")

    elif 'help' in query_lower or 'what can you' in query_lower:
        return ("💡 **I can help you with:**\n\n"
                "- Get overview: 'What are the main themes?'\n"
                "- Topic details: 'Tell me about topic 5'\n"
                "- Statistics: 'How many topics are there?'\n"
                "- Comparisons: 'Which topics are most common?'\n"
                "- Label format: 'Explain the hierarchical labels'\n\n"
                "Just ask your question in natural language!")

    else:
        # Generic response with helpful suggestions
        return ("🤔 I'm not sure how to answer that specific question. Here are some things you can ask:\n\n"
                "- 'What are the main themes in my data?'\n"
                "- 'Tell me about topic 5'\n"
                "- 'Which topics are most common?'\n"
                "- 'How many topics are there?'\n"
                "- 'Explain the hierarchical label format'\n\n"
                f"**Quick Stats:** {len(processed_df):,} documents across {len([t for t in topics if t != -1])} topics")

def main():
    st.title("🚀 Complete BERTopic with All Features")

    # Initialize session state
    if 'embeddings_computed' not in st.session_state:
        st.session_state.embeddings_computed = False
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'umap_embeddings' not in st.session_state:
        st.session_state.umap_embeddings = None
    if 'documents' not in st.session_state:
        st.session_state.documents = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'reclusterer' not in st.session_state:
        st.session_state.reclusterer = None
    if 'current_topics' not in st.session_state:
        st.session_state.current_topics = None
    if 'current_topic_info' not in st.session_state:
        st.session_state.current_topic_info = None
    if 'valid_indices' not in st.session_state:
        st.session_state.valid_indices = None
    if 'min_topic_size_used' not in st.session_state:
        st.session_state.min_topic_size_used = 10
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = 'data'
    if 'topic_info' not in st.session_state:
        st.session_state.topic_info = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'min_topic_size' not in st.session_state:
        st.session_state.min_topic_size = 10
    if 'clustering_method' not in st.session_state:
        st.session_state.clustering_method = 'Unknown'
    if 'gpu_used' not in st.session_state:
        st.session_state.gpu_used = False
    if 'text_col' not in st.session_state:
        st.session_state.text_col = None
    if 'browser_df' not in st.session_state:
        st.session_state.browser_df = None
    if 'topic_human' not in st.session_state:
        st.session_state.topic_human = {}
    if 'last_topics_hash' not in st.session_state:
        st.session_state.last_topics_hash = None

    # Check GPU capabilities (cached in session state to avoid repeated checks)
    if 'gpu_capabilities' not in st.session_state:
        st.session_state.gpu_capabilities = check_gpu_capabilities()
    gpu_capabilities = st.session_state.gpu_capabilities

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # GPU Status Display
        if gpu_capabilities['cuda_available']:
            st.success(f"✅ GPU: {gpu_capabilities['device_name']}")
            if gpu_capabilities['gpu_memory_free']:
                st.info(f"Memory: {gpu_capabilities['gpu_memory_free']} / {gpu_capabilities['gpu_memory_total']}")
        else:
            st.warning("⚠️ No GPU detected. Using CPU (slower)")

        # Acceleration packages status
        st.subheader("📦 Acceleration Status")
        accel_cols = st.columns(2)
        with accel_cols[0]:
            st.write(f"{'✅' if gpu_capabilities['cuml_available'] else '❌'} cuML")
            st.write(f"{'✅' if gpu_capabilities['cupy_available'] else '❌'} CuPy")
        with accel_cols[1]:
            st.write(f"{'✅' if gpu_capabilities['faiss_gpu_available'] else '❌'} FAISS GPU")
            st.write(f"{'✅' if gpu_capabilities['accelerate_available'] else '❌'} Accelerate")

        # File upload
        st.header("📄 Data Input")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            st.session_state.uploaded_file_name = uploaded_file.name.replace('.csv', '')

            try:
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df):,} rows")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.stop()

            # Column selection
            text_col = st.selectbox(
                "Select text column",
                df.columns.tolist(),
                help="Column containing the text to analyze"
            )

            # Analysis Settings
            st.header("🎯 Analysis Settings")
            
            # ✅ MEMORY OPTIMIZATION PROFILE
            st.subheader("⚡ Performance & Memory")
            
            # Get system RAM
            system_ram = psutil.virtual_memory().total / (1024**3)
            num_docs = len(df)
            recommended_profile = MemoryProfileConfig.get_recommended_profile(system_ram, num_docs)
            
            # Profile selector
            profile_options = {
                'conservative': MemoryProfileConfig.PROFILES['conservative']['name'] + ' - ' + 
                               MemoryProfileConfig.PROFILES['conservative']['description'],
                'balanced': MemoryProfileConfig.PROFILES['balanced']['name'] + ' - ' + 
                           MemoryProfileConfig.PROFILES['balanced']['description'],
                'aggressive': MemoryProfileConfig.PROFILES['aggressive']['name'] + ' - ' + 
                             MemoryProfileConfig.PROFILES['aggressive']['description'],
                'extreme': MemoryProfileConfig.PROFILES['extreme']['name'] + ' - ' + 
                          MemoryProfileConfig.PROFILES['extreme']['description'],
            }
            
            selected_profile = st.selectbox(
                "Memory Profile",
                options=list(profile_options.keys()),
                format_func=lambda x: profile_options[x],
                index=list(profile_options.keys()).index(recommended_profile),
                help=f"Higher profiles use more RAM for faster processing. System RAM: {system_ram:.1f}GB | Recommended: {recommended_profile.title()}"
            )
            
            # Show estimated memory usage
            estimated_memory = MemoryProfileConfig.estimate_memory_usage(selected_profile, num_docs)
            profile_config = MemoryProfileConfig.get_profile(selected_profile)
            
            mem_col1, mem_col2 = st.columns(2)
            with mem_col1:
                st.metric("System RAM", f"{system_ram:.1f} GB")
            with mem_col2:
                color = "🟢" if estimated_memory < system_ram * 0.6 else "🟡" if estimated_memory < system_ram * 0.8 else "🔴"
                st.metric("Est. Usage", f"{color} {estimated_memory:.1f} GB")
            
            # Show what's enabled
            if profile_config['aggressive_caching']:
                st.info("✅ Aggressive caching enabled - Maximum speed!")
            
            # Store in session state
            if 'memory_profile' not in st.session_state or st.session_state.get('memory_profile') != selected_profile:
                st.session_state.memory_profile = selected_profile
                st.session_state.profile_config = profile_config
            
            st.divider()

            # Embedding model selection
            embedding_model = st.selectbox(
                "Embedding Model",
                [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "all-MiniLM-L12-v2",
                    "paraphrase-multilingual-MiniLM-L12-v2"
                ],
                help="Smaller models are faster but may be less accurate"
            )

            # Preprocessing Settings
            with st.expander("🔧 Preprocessing Settings"):
                max_doc_length = st.slider(
                    "Max document length (chars)",
                    1000, 10000, 5000,
                    help="Documents longer than this will be truncated"
                )
                min_doc_length = st.slider(
                    "Min document length (chars)",
                    1, 100, 10,
                    help="Documents shorter than this will be removed"
                )

            # Topic Size Control
            st.subheader("📏 Topic Size Control")
            default_min_topic_size = max(2, min(10, len(df) // 50))
            min_topic_size = st.slider(
                "Minimum Topic Size",
                min_value=2,
                max_value=max(2, min(100, len(df) // 10)),
                value=default_min_topic_size,
                help="Minimum number of documents per topic. Topics smaller than this will be merged."
            )

            # Outlier Reduction Strategy
            outlier_strategy = st.selectbox(
                "Outlier Reduction Strategy",
                ["Aggressive (K-means) - 0% outliers",
                 "Moderate (Lenient HDBSCAN) - <15% outliers",
                 "Conservative (Standard HDBSCAN) - Natural clustering"],
                help="Choose how aggressively to assign outliers to topics"
            )

            # Advanced Settings
            with st.expander("🔧 Advanced Settings"):
                # Number of topics (for K-means)
                if "Aggressive" in outlier_strategy:
                    safe_den = max(2, min_topic_size)
                    max_topics = max(5, len(df) // safe_den)
                    default_topics = min(max_topics, max(5, max(1, len(df) // 50)))

                    nr_topics = st.number_input(
                        "Initial Number of Topics",
                        min_value=2,
                        max_value=max(2, max_topics),
                        value=max(2, default_topics),
                        help=f"Number of topics to create (limited by min topic size of {min_topic_size})"
                    )
                else:
                    nr_topics = max(2, min(10, len(df) // 50))

                # UMAP parameters
                n_neighbors = st.slider("UMAP n_neighbors", 2, 50, 15)
                n_components = st.slider("UMAP n_components", 2, 10, 5)

                # Batch size
                batch_size = st.select_slider(
                    "Batch size",
                    options=[8, 16, 32, 64],
                    value=32,
                    help="Reduce if you get memory errors"
                )

                # Representation model (placeholder toggle kept for parity)
                use_mmr = st.checkbox("Use MMR for diverse keywords", value=True)
                
                # LLM-based topic labeling
                st.subheader("🤖 AI-Enhanced Topic Labels")
                use_llm_labeling = st.checkbox(
                    "Use Local LLM for Better Labels",
                    value=False,
                    help="Generate topic labels using a local language model (requires transformers library)"
                )
                
                llm_model_name = None
                force_llm_cpu = False
                if use_llm_labeling:
                    # Build model selection with context window info
                    model_options = list(LLM_MODEL_CONFIG.keys())
                    model_display_names = {
                        name: f"{name.split('/')[-1]} - {config['description']} ({config['recommended_docs']} docs)"
                        for name, config in LLM_MODEL_CONFIG.items()
                    }
                    
                    llm_model_name = st.selectbox(
                        "LLM Model",
                        options=model_options,
                        format_func=lambda x: model_display_names[x],
                        help="Phi-128 can analyze 50+ documents per topic for maximum accuracy. First load takes time to download."
                    )
                    
                    # ✅ NEW: Force CPU option
                    force_llm_cpu = st.checkbox(
                        "Force CPU for LLM (use system RAM)",
                        value=False,
                        help="Load LLM on CPU using system RAM instead of GPU. Use this if GPU is full or you get OOM errors."
                    )
                    
                    if force_llm_cpu:
                        st.info("💻 LLM will use CPU + system RAM (slower but reliable)")
                    elif torch.cuda.is_available():
                        gpu_free = torch.cuda.mem_get_info()[0] / (1024**3)
                        if gpu_free < 4.0:
                            st.warning(f"⚠️ Only {gpu_free:.1f}GB GPU memory free. Consider forcing CPU mode.")
                    
                    st.caption("⚠️ First-time download may be large (3-14GB). Model is cached after first use.")
                    st.caption("💡 Phi-3-mini-4k: Recommended for speed (8 docs). Phi-3-mini-128k: Maximum accuracy (50+ docs).")
                    if force_llm_cpu:
                        st.caption("🔧 CPU mode: Needs ~12GB system RAM. Slower but doesn't use GPU memory.")

                # Seed words
                st.subheader("🎯 Seed Words (Optional)")
                seed_words_input = st.text_area(
                    "Enter seed words (one set per line)",
                    placeholder="Example:\nfinance, money, budget, cost\nmarketing, campaign, advertising",
                    help="Guide topic discovery with predefined keyword sets"
                )

            # Compute embeddings button
            if st.button("🚀 Compute Embeddings & Enable Interactive Mode", type="primary"):
                # Step 1: Preprocess documents
                with st.spinner("Preprocessing documents..."):
                    raw_documents = df[text_col].tolist()

                    cleaned_docs, valid_indices, stats = RobustTextPreprocessor.preprocess_documents(
                        raw_documents,
                        max_length=max_doc_length,
                        min_length=min_doc_length,
                        show_warnings=True
                    )

                    if not cleaned_docs:
                        st.error("No valid documents after preprocessing!")
                        st.stop()

                    st.session_state.documents = cleaned_docs
                    st.session_state.valid_indices = valid_indices
                    st.session_state.text_col = text_col

                    st.info(f"✅ Preprocessed {len(cleaned_docs):,} valid documents")

                # Step 2: Load model and compute embeddings (SAFE)
                with st.spinner("Computing embeddings (this is the slow part, done only once)..."):
                    try:
                        _ = load_embedding_model(embedding_model)

                        safe_model = SafeEmbeddingModel(
                            model_name=embedding_model,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        embeddings = safe_model.encode_safe(
                            cleaned_docs,
                            batch_size=batch_size,
                            show_progress=True
                        )
                        st.session_state.embeddings = embeddings
                        
                        # Store model for semantic search
                        st.session_state.embedding_model_name = embedding_model
                        st.session_state.safe_model = safe_model
                    except RuntimeError as e:
                        st.error(f"Embedding error: {str(e)}")
                        st.info("If this persists, lower the batch size and/or max document length.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Embedding error: {str(e)}")
                        st.info("Try reducing batch size or document length")
                        st.stop()

                # Step 3: Compute UMAP reduction
                with st.spinner("Computing UMAP reduction for fast reclustering..."):
                    umap_embeddings = compute_umap_embeddings(embeddings, n_neighbors, n_components)
                    st.session_state.umap_embeddings = umap_embeddings

                # Step 4: Build FAISS index for RAG and LLM Analysis
                if use_llm_labeling:
                    with st.spinner("🔍 Step 4: Building FAISS index for LLM analysis and RAG chat..."):
                        faiss_index = build_faiss_index(embeddings)
                        st.session_state.faiss_index = faiss_index
                        if faiss_index:
                            st.success("✅ FAISS index ready! Will use for intelligent document selection in LLM analysis.")
                else:
                    # Still build for RAG chat even if LLM labeling is disabled
                    with st.spinner("🔍 Step 4: Building FAISS index for RAG chat..."):
                        faiss_index = build_faiss_index(embeddings)
                        st.session_state.faiss_index = faiss_index
                        if faiss_index:
                            st.success("✅ FAISS index ready for RAG chat!")

                # ✅ Clear GPU memory before loading LLM
                if torch.cuda.is_available() and use_llm_labeling:
                    st.info("🧹 Clearing GPU cache before loading LLM...")
                    clear_gpu_memory()
                    gpu_free_after = torch.cuda.mem_get_info()[0] / (1024**3)
                    st.info(f"📊 GPU memory available for LLM: {gpu_free_after:.1f} GB")

                # Step 5: Load LLM if enabled
                llm_model = None
                if use_llm_labeling and llm_model_name:
                    with st.spinner(f"Loading {llm_model_name} for enhanced labeling..."):
                        llm_model = load_local_llm(llm_model_name, force_cpu=force_llm_cpu)
                        if llm_model:
                            st.success("✅ LLM loaded successfully!")
                            # Debug: Show model info
                            model, tokenizer = llm_model
                            st.info(f"📦 Model loaded: {type(model).__name__} | Tokenizer: {type(tokenizer).__name__}")
                        else:
                            st.error("❌ LLM FAILED TO LOAD!")
                            st.warning("⚠️ Will use TF-IDF fallback for all labels")
                            st.info("💡 Common causes:\n"
                                   "- Insufficient GPU memory (try Force CPU mode)\n"
                                   "- Insufficient system RAM (need 12GB+ for CPU mode)\n"
                                   "- Model download failed (check internet connection)\n"
                                   "- Missing dependencies (install transformers, accelerate)")
                elif use_llm_labeling and not llm_model_name:
                    st.error("❌ LLM checkbox is checked but no model selected!")
                    st.info("Please select an LLM model from the dropdown")

                # Step 6: Create reclusterer
                st.session_state.reclusterer = FastReclusterer(
                    cleaned_docs, embeddings, umap_embeddings,
                    llm_model_name=llm_model_name if use_llm_labeling else None
                )

                # Set LLM model if loaded
                if llm_model:
                    st.session_state.reclusterer.llm_model = llm_model
                    st.session_state.reclusterer.llm_model_name = llm_model_name
                    st.success(f"✅ LLM attached to reclusterer: {llm_model_name}")
                else:
                    st.warning("⚠️ No LLM model to attach - will use TF-IDF fallback")

                # Step 7: (optional) Seed words parsed
                seed_topic_list = []
                if seed_words_input:
                    for line in seed_words_input.strip().split('\n'):
                        if line.strip():
                            words = [w.strip() for w in line.split(',')]
                            if words:
                                seed_topic_list.append(words)

                # Step 8: Perform initial clustering
                with st.spinner("Performing initial clustering..."):
                    if "Aggressive" in outlier_strategy:
                        method = 'kmeans'
                        initial_n_topics = nr_topics
                    else:
                        method = 'hdbscan'
                        initial_n_topics = 10

                    topics, topic_info = st.session_state.reclusterer.recluster(
                        n_topics=initial_n_topics,
                        min_topic_size=min_topic_size,
                        use_reduced=umap_embeddings is not None,
                        method=method
                    )

                    if topics is None:
                        st.error("Initial clustering failed!")
                        st.stop()

                    # ALWAYS normalize topic_info to guarantee Human_Label etc.
                    topic_info = normalize_topic_info(topic_info)

                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.topic_info = topic_info
                    st.session_state.min_topic_size = min_topic_size
                    st.session_state.min_topic_size_used = min_topic_size
                    st.session_state.embeddings_computed = True
                    st.session_state.clustering_method = "K-means" if method == 'kmeans' else "HDBSCAN"
                    st.session_state.gpu_used = gpu_capabilities['cuda_available']
                    st.session_state.model = None

                    st.success("✅ Embeddings computed! You can now adjust topics dynamically with the slider below.")
                    st.balloons()

    # Main content area
    if st.session_state.embeddings_computed:
        st.success("✅ Embeddings ready! Use the controls below for instant topic adjustment.")

        # Interactive controls section
        st.header("🎚️ Dynamic Topic Adjustment")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            denom = max(2, st.session_state.min_topic_size)
            max_topics = max(2, min(2000, len(st.session_state.documents) // denom))

            n_topics_slider = st.slider(
                "🎯 **Number of Topics** (Adjust in real-time!)",
                min_value=2,
                max_value=max_topics,
                value=min(10, max_topics),
                help="Move the slider to instantly re-cluster with different number of topics"
            )

        with col2:
            clustering_method = st.selectbox(
                "Method",
                ["K-means (Fast)", "HDBSCAN"],
                help="K-means is faster for interactive adjustment"
            )

        with col3:
            use_reduced = st.checkbox(
                "Use UMAP reduction",
                value=st.session_state.umap_embeddings is not None,
                disabled=st.session_state.umap_embeddings is None,
                help="Faster reclustering using reduced dimensions"
            )

        # Recluster button
        if st.button("🔄 Recluster with New Settings", type="secondary"):
            with st.spinner(f"Reclustering into {n_topics_slider} topics... (This is fast!)"):
                method = 'kmeans' if "K-means" in clustering_method else 'hdbscan'
                topics, topic_info = st.session_state.reclusterer.recluster(
                    n_topics=n_topics_slider,
                    min_topic_size=st.session_state.min_topic_size,
                    use_reduced=use_reduced and st.session_state.umap_embeddings is not None,
                    method=method
                )

                if topics is not None:
                    topic_info = normalize_topic_info(topic_info)
                    st.session_state.current_topics = topics
                    st.session_state.current_topic_info = topic_info
                    st.session_state.topic_info = topic_info
                    st.session_state.clustering_method = "K-means" if method == 'kmeans' else "HDBSCAN"
                    
                    # Clear cached data to force rebuild
                    if 'browser_df' in st.session_state:
                        del st.session_state.browser_df
                    if 'topic_human' in st.session_state:
                        del st.session_state.topic_human
                    if 'processed_df' in st.session_state:
                        del st.session_state.processed_df
                    
                    st.success(f"✅ Reclustered into {len(topic_info[topic_info['Topic'] != -1])} topics!")
                    st.rerun()  # CRITICAL: Force full page refresh
                else:
                    st.error("Reclustering failed. Try different parameters.")

        # Add LLM Analysis button (for running LLM analysis post-clustering)
        st.markdown("---")
        st.subheader("🤖 Add AI Insights")

        # Check if we already have LLM analysis
        has_llm_analysis = False
        if st.session_state.get('current_topic_info') is not None:
            current_info = st.session_state.current_topic_info
            if 'LLM_Analysis' in current_info.columns and current_info['LLM_Analysis'].any():
                has_llm_analysis = True
                st.info("💡 LLM analysis already exists. Running again will overwrite it.")

        # LLM selection for post-clustering analysis
        col_llm1, col_llm2 = st.columns([2, 1])
        with col_llm1:
            post_llm_model_name = st.selectbox(
                "Select LLM Model",
                options=list(LLM_MODEL_CONFIG.keys()),
                help="Choose an LLM to generate AI-powered insights for your topics",
                key="post_cluster_llm_selector"
            )

        with col_llm2:
            post_llm_force_cpu = st.checkbox(
                "Force CPU",
                value=False,
                help="Use CPU instead of GPU (slower but uses less GPU memory)",
                key="post_cluster_force_cpu"
            )

        # Run LLM Analysis button
        if st.button("🚀 Generate LLM Analysis for Current Topics", type="primary"):
            # Validate requirements
            if st.session_state.get('current_topics') is None:
                st.error("❌ No topics found. Please cluster your data first.")
            elif st.session_state.get('documents') is None:
                st.error("❌ No documents found. Please recompute embeddings.")
            elif st.session_state.get('embeddings') is None:
                st.error("❌ No embeddings found. Please recompute embeddings.")
            else:
                # Load LLM
                with st.spinner(f"Loading {post_llm_model_name}..."):
                    post_llm = load_local_llm(post_llm_model_name, force_cpu=post_llm_force_cpu)

                    if post_llm is None:
                        st.error("❌ Failed to load LLM. Check the error messages above.")
                    else:
                        # Run LLM analysis
                        try:
                            updated_topic_info = run_llm_analysis_on_topics(
                                topics=st.session_state.current_topics,
                                topic_info=st.session_state.current_topic_info,
                                documents=st.session_state.documents,
                                embeddings=st.session_state.embeddings,
                                llm_model=post_llm
                            )

                            # Update session state with new topic_info
                            st.session_state.current_topic_info = updated_topic_info
                            st.session_state.topic_info = updated_topic_info

                            # Clear cached browser_df to force rebuild with new LLM_Analysis
                            if 'browser_df' in st.session_state:
                                del st.session_state.browser_df

                            st.success("🎉 LLM analysis added successfully! Refreshing display...")
                            st.rerun()  # Trigger full refresh to show new column

                        except Exception as e:
                            st.error(f"❌ Error during LLM analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

        # Display results
        if st.session_state.current_topics is not None:
            topics = st.session_state.current_topics
            topic_info = normalize_topic_info(st.session_state.current_topic_info)

            # Calculate metrics
            total_docs = len(topics)
            unique_topics = len(set(topics)) - (1 if -1 in topics else 0)
            outlier_count = sum(1 for t in topics if t == -1)
            coverage = ((total_docs - outlier_count) / total_docs) * 100 if total_docs > 0 else 0

            # Create processed dataframe
            processed_df = pd.DataFrame()
            processed_df['document'] = st.session_state.documents
            processed_df['topic'] = topics
            processed_df['topic_label'] = [f"Topic {t}" if t != -1 else "Outlier" for t in topics]

            # Add keywords for each topic + human labels
            topic_keywords = {}
            topic_human = {}
            for _, row in topic_info.iterrows():
                topic_keywords[row['Topic']] = row['Keywords']
                topic_human[row['Topic']] = row['Human_Label'] if 'Human_Label' in row else (row.get('Name') or f"Topic {row['Topic']}")
            processed_df['keywords'] = processed_df['topic'].map(topic_keywords)

            # Store in session state for export and other tabs
            st.session_state.processed_df = processed_df

            # Balance Analysis
            balance_analyzer = CategoryBalanceAnalyzer()
            balance_analysis = balance_analyzer.analyze_distribution(
                topics,
                include_outliers=False
            )

            # Build browser-ready dataframe ONLY ONCE (not on every rerun!)
            # Check if we need to rebuild (topics changed or browser_df doesn't exist)
            rebuild_browser = False
            if 'browser_df' not in st.session_state or st.session_state.browser_df is None:
                rebuild_browser = True
            elif 'last_topics_hash' not in st.session_state:
                rebuild_browser = True
            else:
                # Check if topics changed by comparing hash
                current_hash = hash(tuple(topics))
                if st.session_state.last_topics_hash != current_hash:
                    rebuild_browser = True
            
            if rebuild_browser:
                # Only rebuild when necessary
                browser_df = st.session_state.df.copy()
                full_topics = np.full(len(browser_df), -1)
                for i, valid_idx in enumerate(st.session_state.valid_indices):
                    if i < len(topics):
                        full_topics[valid_idx] = topics[i]
                browser_df['Topic'] = full_topics
                browser_df['Topic_Label'] = np.where(browser_df['Topic'] == -1, "Outlier",
                                                     "Topic " + browser_df['Topic'].astype(str))
                # human labels / keywords / LLM analysis
                st.session_state.topic_human = topic_human
                topic_keywords_map = topic_keywords
                browser_df['Topic_Human_Label'] = browser_df['Topic'].map(st.session_state.topic_human).fillna(browser_df['Topic_Label'])
                browser_df['Topic_Keywords'] = browser_df['Topic'].map(topic_keywords_map).fillna('N/A')

                # Add LLM_Analysis column from topic_info
                if 'LLM_Analysis' in topic_info.columns:
                    topic_llm_analysis_map = dict(zip(topic_info['Topic'], topic_info['LLM_Analysis']))
                    browser_df['Topic_LLM_Analysis'] = browser_df['Topic'].map(topic_llm_analysis_map).fillna('')
                else:
                    browser_df['Topic_LLM_Analysis'] = ''

                browser_df['Valid_Document'] = [i in st.session_state.valid_indices for i in range(len(browser_df))]

                # Cache in session state
                st.session_state.browser_df = browser_df
                st.session_state.last_topics_hash = hash(tuple(topics))
            else:
                # Use cached version - no expensive operations!
                st.session_state.topic_human = topic_human  # Update labels (cheap)

            # Metrics display
            st.header("📊 Results Summary")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Documents", f"{total_docs:,}")
            with col2:
                st.metric("Topics", unique_topics)
            with col3:
                st.metric("Outliers", f"{outlier_count:,} ({100-coverage:.1f}%)")
            with col4:
                st.metric("Coverage", f"{coverage:.1f}%")
            with col5:
                st.metric("Min Topic Size", st.session_state.get('min_topic_size_used', 'N/A'))

            # Balance warnings
            if not balance_analysis['balanced']:
                st.error("⚠️ **Topic Distribution Imbalance Detected!**")
                for warning in balance_analysis['warnings']:
                    st.warning(warning)

            # Tabs for different views
            tabs = st.tabs([
                "📊 Topics Overview",
                "📈 Distribution Analysis",
                "🔍 Split Large Topics",
                "🗺️ Interactive Visualization",
                "📄 Topic Browser (OPTIMIZED)",
                "💾 Export"
            ])

            with tabs[0]:  # Topics Overview
                st.subheader("Topic Information")

                display_df = normalize_topic_info(topic_info)
                if -1 in display_df['Topic'].values:
                    display_df = display_df[display_df['Topic'] != -1]
                display_df['Percentage'] = (display_df['Count'] / total_docs * 100).round(2)

                # Include LLM_Analysis if available
                if 'LLM_Analysis' in display_df.columns and display_df['LLM_Analysis'].any():
                    display_df = display_df[['Topic', 'Human_Label', 'LLM_Analysis', 'Keywords', 'Count', 'Percentage']]
                    st.caption("🤖 **LLM_Analysis** column shows AI-generated insights about each topic")
                else:
                    display_df = display_df[['Topic', 'Human_Label', 'Keywords', 'Count', 'Percentage']]

                # Streamlit default dataframe (dark-mode friendly)
                st.dataframe(display_df, use_container_width=True)
                st.caption("✨ Notice: Human_Label now shows actual category names like 'Customer Support' instead of keyword lists!")

            with tabs[1]:  # Distribution Analysis
                st.subheader("📈 Topic Distribution Analysis")

                # Prepare data for visualization
                viz_df = pd.DataFrame(Counter([t for t in topics if t != -1]).items(), columns=['Topic', 'Count'])
                viz_df = viz_df.sort_values('Count', ascending=False)
                viz_df['Topic_Label'] = viz_df['Topic'].apply(
                    lambda x: st.session_state.topic_human.get(x, f"Topic {x}")
                )

                # Bar chart
                fig = px.bar(viz_df, x='Topic_Label', y='Count',
                             title='Document Distribution Across Topics',
                             color='Count',
                             color_continuous_scale='Viridis')

                # Add threshold line
                threshold_count = total_docs * 0.3
                fig.add_hline(y=threshold_count, line_dash="dash", line_color="red",
                              annotation_text="30% threshold")

                st.plotly_chart(fig, use_container_width=True)

                # Balance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Balance Score", f"{balance_analysis['balance_score']:.2f}",
                              help="1.0 = perfectly balanced, 0 = highly imbalanced")
                with col2:
                    st.metric("Oversized Topics", len(balance_analysis['oversized_topics']))
                with col3:
                    avg_size = total_docs / unique_topics if unique_topics > 0 else 0
                    st.metric("Avg Topic Size", f"{avg_size:.0f} docs")

            with tabs[2]:  # Split Large Topics
                st.subheader("🔍 Split Large Topics")

                if balance_analysis['oversized_topics']:
                    st.warning(f"Found {len(balance_analysis['oversized_topics'])} oversized topic(s)")

                    oversized_options = [
                        f"Topic {t['topic']}: {st.session_state.topic_human.get(t['topic'], 'Topic ' + str(t['topic']))} - {t['count']} docs ({t['ratio']*100:.1f}%)"
                        for t in balance_analysis['oversized_topics']
                    ]

                    selected_topic_to_split = st.selectbox(
                        "Select topic to split",
                        oversized_options
                    )

                    topic_to_split = int(selected_topic_to_split.split(':')[0].replace('Topic ', ''))

                    suggested_splits = next(
                        t['suggested_splits'] for t in balance_analysis['oversized_topics']
                        if t['topic'] == topic_to_split
                    )

                    sub_n_topics = st.slider(
                        "Number of subtopics",
                        min_value=2,
                        max_value=min(10, suggested_splits * 2),
                        value=suggested_splits,
                        help="How many subtopics to create from this large topic"
                    )

                    if st.button(f"🔍 Split Topic {topic_to_split} into {sub_n_topics} subtopics"):
                        docs_to_split = [doc for doc, t in zip(st.session_state.documents, topics) if t == topic_to_split]

                        if len(docs_to_split) >= max(10, sub_n_topics * 2):
                            with st.spinner(f"Analyzing {len(docs_to_split):,} documents..."):
                                # Use K-means for splitting via BERTopic's API (for convenience)
                                sub_model = BERTopic(
                                    hdbscan_model=GPUKMeans(n_clusters=sub_n_topics) if gpu_capabilities['cuda_available']
                                    else KMeans(n_clusters=sub_n_topics, random_state=42),
                                    min_topic_size=max(2, len(docs_to_split) // (sub_n_topics * 2)),
                                    calculate_probabilities=False,
                                    verbose=False
                                )

                                sub_topics, _ = sub_model.fit_transform(docs_to_split)

                                st.success(f"✅ Split into {len(set(sub_topics))} subtopics")

                                st.write("### 📊 Subtopics Found:")
                                sub_topic_info = sub_model.get_topic_info()

                                sub_topic_display = sub_topic_info[['Topic', 'Count', 'Name']].copy()
                                sub_topic_display['Keywords'] = sub_topic_info['Representation'].apply(
                                    lambda x: ', '.join(x[:3]) if isinstance(x, list) else str(x)[:50]
                                )
                                sub_topic_display['% of Parent'] = (
                                    sub_topic_display['Count'] / len(docs_to_split) * 100
                                ).round(1)

                                st.dataframe(sub_topic_display, use_container_width=True)

                                sub_outliers = sum(1 for t in sub_topics if t == -1)
                                sub_coverage = ((len(sub_topics) - sub_outliers) / len(sub_topics)) * 100

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Subtopics Created", len(set(sub_topics)) - (1 if -1 in sub_topics else 0))
                                with col2:
                                    st.metric("Documents Reassigned", len(docs_to_split) - sub_outliers)
                                with col3:
                                    st.metric("Subtopic Coverage", f"{sub_coverage:.1f}%")
                        else:
                            st.warning(f"Not enough documents for splitting")
                else:
                    st.success("✅ No oversized categories detected. All topics are well-balanced!")

            with tabs[3]:  # Interactive Visualization
                st.subheader("🗺️ Interactive Topic Visualization")

                if len(set(topics)) > 1:
                    viz_df = pd.DataFrame({
                        'Topic': topics,
                        'Document': st.session_state.documents
                    })

                    # Use UMAP embeddings for visualization if available
                    if st.session_state.umap_embeddings is not None and st.session_state.umap_embeddings.shape[1] >= 2:
                        viz_df['X'] = st.session_state.umap_embeddings[:, 0]
                        viz_df['Y'] = st.session_state.umap_embeddings[:, 1]
                    else:
                        # Use PCA for 2D projection
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        coords = pca.fit_transform(st.session_state.embeddings)
                        viz_df['X'] = coords[:, 0]
                        viz_df['Y'] = coords[:, 1]

                    viz_df['Topic_Label'] = [
                        st.session_state.topic_human.get(t, f"Topic {t}") if t != -1 else "Outliers"
                        for t in viz_df['Topic']
                    ]

                    # Sample for performance if too many points
                    if len(viz_df) > 5000:
                        viz_df_sample = viz_df.sample(5000, random_state=42)
                    else:
                        viz_df_sample = viz_df

                    fig = px.scatter(
                        viz_df_sample,
                        x='X', y='Y',
                        color='Topic_Label',
                        hover_data={'Document': True, 'Topic': True},
                        title='Document Clusters in 2D Space',
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 topics for visualization")

            with tabs[4]:  # Fast Topic Browser
                st.subheader("📄 Topic Browser")

                browser_df = st.session_state.browser_df
                text_col = st.session_state.text_col or (st.session_state.df.columns[0] if len(st.session_state.df.columns) else None)

                # Build topic options ONCE and cache in session state
                if 'topic_options_cache' not in st.session_state or st.session_state.get('topic_options_hash') != hash(tuple(topics)):
                    # Simple dropdown for topic selection
                    topic_counts = pd.Series([t for t in topics if t != -1]).value_counts().sort_index()
                    
                    # Build topic options with human labels
                    topic_options = {}
                    topic_options["All Topics"] = "all"
                    
                    for tid in sorted(topic_counts.index):
                        count = topic_counts.get(tid, 0)
                        human_label = st.session_state.topic_human.get(tid, f"Topic {tid}")
                        topic_options[f"Topic {tid}: {human_label} ({count} docs)"] = tid
                    
                    if -1 in topics:
                        outlier_count = sum(1 for t in topics if t == -1)
                        topic_options[f"Outliers ({outlier_count} docs)"] = -1
                    
                    # Cache it
                    st.session_state.topic_options_cache = topic_options
                    st.session_state.topic_options_hash = hash(tuple(topics))
                else:
                    # Use cached version - instant!
                    topic_options = st.session_state.topic_options_cache

                # Single dropdown selector
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_option = st.selectbox(
                        "Select a topic to view documents:",
                        options=list(topic_options.keys()),
                        index=0,
                        key="topic_browser_selector",
                        help="Choose one topic at a time for fast rendering"
                    )
                
                with col2:
                    max_rows = st.number_input(
                        "Max rows to display:",
                        min_value=10,
                        max_value=10000,
                        value=1000,
                        step=100,
                        key="topic_browser_max_rows",
                        help="Limit rows for faster display"
                    )

                # Semantic Search Feature
                st.markdown("---")
                with st.expander("🔍 Semantic Search (AI-powered)", expanded=False):
                    st.caption("Find documents similar to your query using AI embeddings (not just keywords)")
                    
                    search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
                    
                    with search_col1:
                        search_query = st.text_input(
                            "Search query:",
                            placeholder="e.g., 'customer complaints about shipping delays'",
                            key="semantic_search_query",
                            help="Enter a phrase or question - AI will find semantically similar documents"
                        )
                    
                    with search_col2:
                        search_top_n = st.number_input(
                            "Top results:",
                            min_value=5,
                            max_value=500,
                            value=50,
                            step=5,
                            key="semantic_search_top_n",
                            help="Number of most similar documents to show"
                        )
                    
                    with search_col3:
                        search_scope = st.selectbox(
                            "Search in:",
                            options=["Current Topic", "All Topics"],
                            key="semantic_search_scope",
                            help="Search within current topic or across all topics"
                        )
                    
                    search_button = st.button("🔍 Search", key="semantic_search_button", type="primary")
                    
                    if search_button and search_query.strip():
                        with st.spinner("🤖 Computing semantic similarity..."):
                            try:
                                # Get the embedding model
                                if 'safe_model' not in st.session_state or st.session_state.safe_model is None:
                                    st.error("❌ Model not loaded. Please run topic modeling first.")
                                else:
                                    # Encode the search query
                                    query_embedding = st.session_state.safe_model.model.encode(
                                        [search_query],
                                        convert_to_numpy=True,
                                        normalize_embeddings=True
                                    )[0]
                                    
                                    # Get embeddings and valid indices
                                    embeddings = st.session_state.embeddings
                                    valid_indices = st.session_state.valid_indices
                                    
                                    # Determine which documents to search
                                    if search_scope == "Current Topic" and selected_topic_id != "all":
                                        # Get indices for current topic
                                        topic_mask = browser_df['Topic'].values == selected_topic_id
                                        topic_indices = np.where(topic_mask)[0]
                                        
                                        # Map to valid_indices
                                        search_indices = []
                                        for idx in topic_indices:
                                            if idx in valid_indices:
                                                search_indices.append(valid_indices.index(idx))
                                        
                                        if len(search_indices) == 0:
                                            st.warning("No valid documents in current topic to search")
                                            search_indices = list(range(len(embeddings)))
                                    else:
                                        # Search all documents
                                        search_indices = list(range(len(embeddings)))
                                    
                                    # Calculate cosine similarity for selected documents
                                    similarities = []
                                    for idx in search_indices:
                                        if idx < len(embeddings):
                                            similarity = np.dot(query_embedding, embeddings[idx])
                                            similarities.append((idx, similarity, valid_indices[idx]))
                                    
                                    # Sort by similarity (highest first)
                                    similarities.sort(key=lambda x: x[1], reverse=True)
                                    
                                    # Get top N results
                                    top_results = similarities[:search_top_n]
                                    
                                    if len(top_results) > 0:
                                        # Get the browser indices for top results
                                        result_browser_indices = [x[2] for x in top_results]
                                        result_similarities = [x[1] for x in top_results]
                                        
                                        # Create results dataframe
                                        search_results_df = browser_df.iloc[result_browser_indices].copy()
                                        search_results_df.insert(0, 'Similarity', [f"{sim:.3f}" for sim in result_similarities])
                                        
                                        # Store in session state to display below
                                        st.session_state.search_active = True
                                        st.session_state.search_results_df = search_results_df
                                        st.session_state.search_query_display = search_query
                                        
                                        st.success(f"✅ Found {len(top_results)} similar documents!")
                                    else:
                                        st.warning("No results found")
                                        
                            except Exception as e:
                                st.error(f"❌ Search error: {str(e)}")
                    
                    elif search_button:
                        st.warning("Please enter a search query")
                
                st.markdown("---")

                # Check if we should display search results instead of topic view
                display_search_results = st.session_state.get('search_active', False)
                
                if display_search_results and 'search_results_df' in st.session_state:
                    # Display search results
                    search_results_df = st.session_state.search_results_df
                    search_query_display = st.session_state.get('search_query_display', '')
                    
                    st.info(f"🔍 **Semantic Search Results** for: \"{search_query_display}\" — Showing {len(search_results_df)} most similar documents")
                    
                    # Add a button to clear search and return to normal view
                    if st.button("❌ Clear Search Results", key="clear_search"):
                        st.session_state.search_active = False
                        if 'search_results_df' in st.session_state:
                            del st.session_state.search_results_df
                        st.rerun()
                    
                    # Reorder columns
                    meta_cols = ['Similarity', 'Topic', 'Topic_Human_Label', 'Topic_Keywords']
                    other_cols = [c for c in search_results_df.columns if c not in meta_cols and c not in ['Topic_Label', 'Valid_Document']]
                    ordered_cols = [c for c in meta_cols if c in search_results_df.columns] + other_cols
                    search_results_df = search_results_df[ordered_cols]
                    
                    # Display
                    st.dataframe(
                        search_results_df,
                        use_container_width=True,
                        height=600
                    )
                    
                    # Download button
                    st.download_button(
                        label=f"📥 Download Search Results ({len(search_results_df)} rows)",
                        data=search_results_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"semantic_search_{search_query_display[:30]}_{st.session_state.uploaded_file_name}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    # Normal topic browsing view
                    # Get the selected topic ID
                    selected_topic_id = topic_options[selected_option]

                    # Fast filtering - single topic only
                    if selected_topic_id == "all":
                        display_df = browser_df.head(max_rows)
                        st.info(f"📊 Showing first {len(display_df):,} of {len(browser_df):,} total documents")
                    else:
                        # Use numpy for fast filtering
                        mask = browser_df['Topic'].values == selected_topic_id
                        filtered_df = browser_df[mask]
                        display_df = filtered_df.head(max_rows)
                        
                        if selected_topic_id == -1:
                            st.info(f"📊 Showing {len(display_df):,} of {len(filtered_df):,} outlier documents")
                        else:
                            human_label = st.session_state.topic_human.get(selected_topic_id, f"Topic {selected_topic_id}")
                            st.info(f"📊 Topic {selected_topic_id}: **{human_label}** — Showing {len(display_df):,} of {len(filtered_df):,} documents")

                    # Reorder columns - put topic metadata first
                    meta_cols = ['Topic', 'Topic_Human_Label', 'Topic_LLM_Analysis', 'Topic_Keywords']
                    other_cols = [c for c in display_df.columns if c not in meta_cols and c not in ['Topic_Label', 'Valid_Document']]
                    ordered_cols = [c for c in meta_cols if c in display_df.columns] + other_cols
                    display_df = display_df[ordered_cols]

                    # Simple, fast dataframe display
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=600
                    )

                    # Download button
                    if selected_topic_id == "all":
                        download_df = browser_df
                        filename = f"all_topics_{st.session_state.uploaded_file_name}.csv"
                    else:
                        download_df = browser_df[browser_df['Topic'] == selected_topic_id]
                        filename = f"topic_{selected_topic_id}_{st.session_state.uploaded_file_name}.csv"

                    st.download_button(
                        label=f"📥 Download {selected_option} ({len(download_df):,} rows)",
                        data=download_df.to_csv(index=False).encode('utf-8'),
                        file_name=filename,
                        mime="text/csv"
                    )

                # Chat Interface integrated into Topic Browser
                st.markdown("---")
                with st.expander("💬 Ask Questions About Topics (RAG-Powered)", expanded=False):
                    st.caption("Get AI-powered insights using FAISS retrieval and LLM generation")

                    # RAG Settings
                    col_rag1, col_rag2 = st.columns(2)
                    with col_rag1:
                        use_rag_chat = st.checkbox(
                            "🤖 Enable RAG Mode (FAISS + LLM)",
                            value=False,
                            help="Use Retrieval-Augmented Generation for intelligent responses based on your actual documents"
                        )
                    with col_rag2:
                        if use_rag_chat:
                            chat_llm_model_name = st.selectbox(
                                "Chat LLM Model",
                                options=list(LLM_MODEL_CONFIG.keys()),
                                help="Select LLM for chat. Can be different from labeling LLM.",
                                key="chat_llm_selector"
                            )
                        else:
                            chat_llm_model_name = None

                    # Load chat LLM if RAG is enabled
                    chat_llm = None
                    if use_rag_chat and chat_llm_model_name:
                        if st.session_state.get('chat_llm_loaded_model') == chat_llm_model_name:
                            # Already loaded
                            chat_llm = st.session_state.get('chat_llm')
                            st.caption(f"✅ Using cached LLM: {chat_llm_model_name.split('/')[-1]}")
                        else:
                            # Need to load
                            with st.spinner(f"Loading {chat_llm_model_name} for chat..."):
                                chat_llm = load_local_llm(chat_llm_model_name, force_cpu=False)
                                if chat_llm:
                                    st.session_state.chat_llm = chat_llm
                                    st.session_state.chat_llm_loaded_model = chat_llm_model_name
                                    st.success(f"✅ Chat LLM loaded: {chat_llm_model_name.split('/')[-1]}")
                                else:
                                    st.error("❌ Failed to load chat LLM. Falling back to rule-based mode.")
                                    use_rag_chat = False

                    # Check if FAISS index is available
                    has_faiss = st.session_state.get('faiss_index') is not None
                    if use_rag_chat and not has_faiss:
                        st.warning("⚠️ FAISS index not available. Please recompute embeddings to enable RAG mode.")
                        use_rag_chat = False

                    # Initialize chat history in session state
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []

                    # Chat input at the top with clear button
                    col_input, col_clear = st.columns([4, 1])
                    with col_clear:
                        if st.button("🗑️ Clear", key="clear_chat_browser", help="Clear chat history"):
                            st.session_state.chat_history = []
                            st.rerun()

                    # Chat input
                    if prompt := st.chat_input("Ask a question about your topics...", key="topic_browser_chat"):
                        # Add user message to chat history
                        st.session_state.chat_history.append({"role": "user", "content": prompt})

                        # Generate response based on topic data and current context
                        with st.spinner("Thinking..."):
                            # Get topic information
                            topic_info_for_chat = normalize_topic_info(topic_info)

                            # Create context from topic data
                            context_parts = []
                            context_parts.append(f"Total documents: {len(processed_df):,}")
                            context_parts.append(f"Number of topics: {unique_topics}")
                            context_parts.append(f"Outliers: {outlier_count} ({100-coverage:.1f}%)")

                            # Add current topic context
                            if selected_topic_id != "all":
                                human_label = st.session_state.topic_human.get(selected_topic_id, f"Topic {selected_topic_id}")
                                context_parts.append(f"\nCurrently viewing: Topic {selected_topic_id} - {human_label}")
                                topic_row = topic_info_for_chat[topic_info_for_chat['Topic'] == selected_topic_id]
                                if len(topic_row) > 0:
                                    context_parts.append(f"Keywords: {topic_row.iloc[0]['Keywords']}")
                                    context_parts.append(f"Document count: {topic_row.iloc[0]['Count']}")

                            context_parts.append("\nTop Topics:")

                            # Add top 10 topics
                            top_topics = topic_info_for_chat.nlargest(10, 'Count')[['Topic', 'Human_Label', 'Keywords', 'Count']]
                            for _, row in top_topics.iterrows():
                                context_parts.append(f"- Topic {row['Topic']}: {row['Human_Label']} ({row['Count']} docs)")
                                context_parts.append(f"  Keywords: {row['Keywords']}")

                            context = "\n".join(context_parts)

                            # Generate response with RAG or rule-based
                            response = generate_chat_response(
                                prompt,
                                context,
                                topic_info_for_chat,
                                topics,
                                processed_df,
                                current_topic_id=selected_topic_id if selected_topic_id != "all" else None,
                                use_rag=use_rag_chat,
                                llm_model=chat_llm if use_rag_chat else None,
                                faiss_index=st.session_state.get('faiss_index'),
                                embeddings=st.session_state.get('embeddings'),
                                documents=st.session_state.get('documents'),
                                safe_model=st.session_state.get('safe_model')
                            )

                            # Add assistant response to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})

                        # Rerun to display the new messages
                        st.rerun()

                    # Display welcome message or chat history
                    st.markdown("---")

                    if len(st.session_state.chat_history) == 0:
                        if use_rag_chat:
                            st.info("🤖 **RAG Mode Active!** I'll retrieve relevant documents and use LLM to answer your questions.\n\n"
                                   "Try asking:\n"
                                   "- 'What do customers say about delivery?'\n"
                                   "- 'Find issues related to installation'\n"
                                   "- 'Summarize the main complaints'\n"
                                   "- 'What are people saying about Samsung products?'")
                        else:
                            st.info("👋 Ask me anything about your topics. For example:\n"
                                   "- 'What are the main themes in my data?'\n"
                                   "- 'Tell me about the current topic'\n"
                                   "- 'Which topics are most common?'\n"
                                   "- 'Show me insights about customer complaints'\n\n"
                                   "💡 **Tip:** Enable RAG Mode above for AI-powered responses!")
                    else:
                        # Display chat history (newest first)
                        for message in reversed(st.session_state.chat_history):
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

            with tabs[5]:  # Export
                st.subheader("💾 Export Results")

                export_df = st.session_state.browser_df.copy()
                safe_topic_info = normalize_topic_info(topic_info)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=convert_df_to_csv(export_df),
                        file_name=f"bertopic_results_{st.session_state.uploaded_file_name}_{n_topics_slider}topics.csv",
                        mime="text/csv",
                        help="Full dataset with topic assignments and human labels"
                    )

                with col2:
                    # Include LLM_Analysis if available
                    if 'LLM_Analysis' in safe_topic_info.columns and safe_topic_info['LLM_Analysis'].any():
                        export_cols = ['Topic', 'Human_Label', 'LLM_Analysis', 'Keywords', 'Count']
                    else:
                        export_cols = ['Topic', 'Human_Label', 'Keywords', 'Count']

                    st.download_button(
                        label="📥 Download Topic Info (CSV)",
                        data=convert_df_to_csv(safe_topic_info[export_cols]),
                        file_name=f"topic_info_{st.session_state.uploaded_file_name}_{n_topics_slider}topics.csv",
                        mime="text/csv",
                        help="Topic descriptions and statistics (with human labels and LLM analysis if enabled)"
                    )

                with col3:
                    # Create summary report
                    summary = f"""Topic Modeling Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
File: {st.session_state.uploaded_file_name}
Documents: {len(processed_df)}
Topics Found: {unique_topics}
Outliers: {outlier_count} ({100-coverage:.1f}%)
Coverage: {coverage:.1f}%
Min Topic Size: {st.session_state.get('min_topic_size_used', 'N/A')}
GPU Used: {gpu_capabilities['cuda_available']}
Clustering Method: {st.session_state.clustering_method}
Human Labels: Auto-generated using semantic category inference
Balance Status: {'Balanced' if balance_analysis['balanced'] else 'Needs Attention'}
Oversized Categories: {len(balance_analysis['oversized_topics'])}
"""
                    st.download_button(
                        label="📥 Download Report (TXT)",
                        data=summary.encode("utf-8"),
                        file_name=f"report_{st.session_state.uploaded_file_name}_{n_topics_slider}topics.txt",
                        mime="text/plain",
                        help="Summary report"
                    )

    elif 'df' not in st.session_state or st.session_state.df is None:
        # Welcome screen
        st.info("👆 Please upload a CSV file in the sidebar to begin.")

        # Feature highlights
        st.header("🚀 Complete Feature Set")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("⚡ Interactive Features")
            st.markdown("""
            - **Dynamic slider** for instant topic adjustment
            - **One-time embedding** computation
            - **Fast reclustering** (<1 second)
            - **Real-time visualization** updates
            - **Interactive scatter plots**
            """)

        with col2:
            st.subheader("🛡️ Robust Processing")
            st.markdown("""
            - **Handles all edge cases**
            - **Smart text preprocessing**
            - **Memory error recovery**
            - **Batch size auto-adjustment**
            - **Encoding error handling**
            """)

        with col3:
            st.subheader("📊 Analysis Tools")
            st.markdown("""
            - **Split large topics** tool
            - **Balance analysis**
            - **Seed words** support
            - **Outlier reduction** strategies
            - **Multiple export** formats
            """)

        st.header("✨ What's New")
        st.success("""
        **🧠 Chain-of-Thought LLM Labeling**: LLM now reads full documents and THINKS before labeling! 
        It analyzes what customers need, then creates descriptive category names. Much smarter than forcing it into a box.
        
        **📖 Richer Context**: Shows up to 800 characters per document (was 400) - LLM sees the full story!
        
        **🎯 Trust the AI**: Removed restrictive validation. Let the LLM use as many words as needed to be clear and specific.
        
        **🚀 Longer Responses**: 800 tokens for LLM output (was 200-300) - room for thoughtful analysis.
        
        **⚡ Better Generation**: Higher temperature (0.5), more diverse sampling (top_p 0.92), gentler repetition penalty.
        
        **🎯 Improved Labels**: Instead of "Help Order Placed", you'll get "Samsung Washer Delivery Scheduling and Installation"!
        
        **🚀 Phi-3-mini-128k Support**: Analyze 50+ full documents per topic with 128k context window!
        
        **🚀 Memory Optimization Profiles**: Choose Conservative (8GB), Balanced (16GB), Aggressive (32GB+), or Extreme (64GB+)!
        """)

        # System check
        with st.expander("🔍 Check Your System"):
            if st.button("Run System Check"):
                capabilities = check_gpu_capabilities()

                st.write("### System Capabilities:")

                gpu_col, pkg_col = st.columns(2)

                with gpu_col:
                    st.write("**GPU Status:**")
                    for key in ['cuda_available', 'device_name', 'gpu_memory_total']:
                        if key in capabilities and capabilities[key] is not None:
                            icon = "✅" if capabilities.get('cuda_available', False) else "❌"
                            value = capabilities[key]
                            if isinstance(value, bool):
                                st.write(f"{icon} {key.replace('_', ' ').title()}")
                            else:
                                st.write(f"{icon} {key.replace('_', ' ').title()}: {value}")

                with pkg_col:
                    st.write("**Packages:**")
                    for key in ['accelerate_available', 'cupy_available', 'faiss_available']:
                        if key in capabilities:
                            icon = "✅" if capabilities[key] else "❌"
                            st.write(f"{icon} {key.replace('_available', '').title()}")

if __name__ == "__main__":
    main()
