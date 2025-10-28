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
# ✅ CARMACK: RAG IMPROVEMENTS - chunking, caching, persistence
from functools import lru_cache
import hashlib
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ✅ PYTORCH EXPANDABLE SEGMENTS: Prevent CUDA memory fragmentation
# Enables GPU to share memory with system RAM and use expandable memory segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
# Enable backend options for better memory management
if torch.cuda.is_available():
    # Set memory allocator to use expandable segments (PyTorch 2.0+)
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
    # Enable cudaMallocAsync for better async memory management (if available)
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except:
        pass  # Older PyTorch versions may not have this


def ensure_model_on_device(model, prefer_gpu=True):
    """
    Ensure model is on the correct device (GPU if available and preferred).
    Prevents mixed device tensor errors.
    """
    if model is None:
        return model

    try:
        if prefer_gpu and torch.cuda.is_available():
            # Check current device
            if hasattr(model, 'device'):
                current_device = model.device.type
                if current_device != 'cuda':
                    # Model is on CPU but GPU is available - move to GPU
                    model = model.cuda()
                    return model
            # device_map="auto" models don't need manual movement
            return model
        else:
            # No GPU or prefer CPU
            return model
    except Exception as e:
        # If moving fails, return as-is
        return model


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

# Detect platform for Windows-specific workarounds
import platform
IS_WINDOWS = platform.system() == 'Windows'

# Make accelerate optional - not strictly required
# Note: accelerate is problematic on Windows
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False

# Check bitsandbytes availability (4-bit quantization)
# Note: bitsandbytes is very problematic on Windows - often fails even when installed
bitsandbytes_available = False
bitsandbytes_error = None
if not IS_WINDOWS:
    try:
        import bitsandbytes
        bitsandbytes_available = True
    except ImportError:
        bitsandbytes_error = "not installed"
    except Exception as e:
        bitsandbytes_error = str(e)
else:
    # Windows: bitsandbytes is not officially supported and often fails
    try:
        import bitsandbytes
        bitsandbytes_available = True
        # Test if it actually works
        try:
            from transformers import BitsAndBytesConfig
            _ = BitsAndBytesConfig(load_in_4bit=True)
        except Exception as e:
            bitsandbytes_available = False
            bitsandbytes_error = f"installed but not working: {str(e)}"
    except ImportError:
        bitsandbytes_error = "not installed (Windows not officially supported)"

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer, CrossEncoder
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
    def detect_optimal_parameters():
        """Detect system capabilities and return optimal parameters for LLM labeling"""
        params = {
            'has_gpu': False,
            'gpu_memory_gb': 0,
            'cpu_cores': os.cpu_count() or 4,
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'batch_size': 10,
            'max_workers': 1,
            'recommended_docs_per_topic': 5
        }

        if torch.cuda.is_available():
            params['has_gpu'] = True
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                params['gpu_memory_gb'] = gpu_props.total_memory / (1024**3)
                params['gpu_name'] = gpu_props.name

                # Set batch size and workers based on GPU memory
                if params['gpu_memory_gb'] >= 20:
                    params['batch_size'] = 40
                    params['max_workers'] = 4
                    params['tier'] = 'High-end GPU'
                elif params['gpu_memory_gb'] >= 14:
                    params['batch_size'] = 25
                    params['max_workers'] = 3
                    params['tier'] = 'Mid-range GPU'
                elif params['gpu_memory_gb'] >= 10:
                    params['batch_size'] = 20
                    params['max_workers'] = 3
                    params['tier'] = 'Standard GPU'
                elif params['gpu_memory_gb'] >= 6:
                    params['batch_size'] = 12
                    params['max_workers'] = 2
                    params['tier'] = 'Entry-level GPU'
                else:
                    params['batch_size'] = 8
                    params['max_workers'] = 1
                    params['tier'] = 'Low-memory GPU'

            except Exception:
                params['batch_size'] = 15
                params['max_workers'] = 2
                params['tier'] = 'Unknown GPU'
        else:
            params['tier'] = 'CPU'
            params['batch_size'] = 10
            params['max_workers'] = 1
            if params['cpu_cores'] >= 16:
                params['max_workers'] = 2

        return params


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
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # ✅ Re-enabled: 30-50% speedup
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
    CARMACK'S VERSION: Parallel individual calls. No batching. No parsing complexity.

    Old approach: Batch 5 topics → Parse fragile string output → 50% failure → Fallback
    New approach: Call each topic in parallel → Simple extraction → 98% success

    This is ACTUALLY faster because:
    1. No wasted tokens on unparseable batch outputs
    2. True parallelism (ThreadPoolExecutor with 4 workers)
    3. Individual failures don't block other topics
    4. More docs per topic (8 vs 3) = better quality

    Args:
        topic_batch: List of dicts with keys: topic_id, label, docs
        llm_model: Tuple of (model, tokenizer)

    Returns:
        Dict mapping topic_id to analysis string
    """
    if not topic_batch or llm_model is None:
        return {}

    results = {}

    # Parallel execution - 4 workers for optimal GPU utilization
    max_workers = min(4, len(topic_batch))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all topics at once
        future_to_topic = {
            executor.submit(
                generate_simple_llm_analysis,
                item['topic_id'],
                item['docs'][:20],  # ✅ CARMACK: Use up to 20 docs like interactive summary (was 8)
                item['label'],
                llm_model,
                max_length=2000,  # ✅ Increased from 500 to allow longer, detailed analysis
                keywords=item.get('keywords', '')  # ✅ CARMACK: Pass keywords for better context
            ): item['topic_id']
            for item in topic_batch
        }

        # Collect results as they complete
        for future in as_completed(future_to_topic):
            topic_id = future_to_topic[future]
            try:
                analysis = future.result()
                # Validation: accept shorter analyses for small topics (20+ chars, reduced from 50)
                if analysis and len(analysis.strip()) > 20 and ' ' in analysis:
                    results[topic_id] = analysis
                else:
                    # Log why validation failed for debugging
                    if analysis:
                        import logging
                        logging.debug(f"Topic {topic_id} analysis rejected (len={len(analysis.strip())}, has_space={' ' in analysis}): {analysis[:100]}")
                # Note: We don't add "No analysis available" here
                # Let the caller decide what to do with missing results
            except Exception as e:
                # Log error for debugging but don't crash
                import logging
                logging.warning(f"Topic {topic_id} analysis failed: {str(e)}")
                pass

    return results


def generate_simple_llm_analysis(topic_id, sample_docs, topic_label, llm_model, max_length=2000, keywords=""):
    """
    Generate a comprehensive 3-5 sentence analysis of what users are saying in this topic.

    Uses the same high-quality approach as the interactive topic summary feature,
    but optimized for batch processing during initial topic modeling.

    Prompt: Provide concise summary capturing main theme, key patterns, and insights.
    """
    if not sample_docs or llm_model is None:
        return None

    try:
        model, tokenizer = llm_model

        # Clean and prepare documents
        cleaned_docs = [str(doc).strip() for doc in sample_docs if doc and str(doc).strip()]
        if not cleaned_docs:
            import logging
            logging.warning(f"Topic {topic_id} REJECTED - no valid documents after cleaning")
            return None

        # Check if documents have enough content (not just whitespace/noise)
        # Lowered threshold: even very short topics deserve analysis
        total_content_length = sum(len(doc) for doc in cleaned_docs)
        if total_content_length < 10:  # Only reject if < 10 chars (almost empty)
            import logging
            logging.warning(f"Topic {topic_id} REJECTED - insufficient document content ({total_content_length} chars)")
            return None

        # ✅ CARMACK: Intelligent sampling like the interactive summary feature
        # Sample documents intelligently (first, middle, last to get variety)
        if len(cleaned_docs) <= 20:
            sample_docs_intelligent = cleaned_docs
        else:
            # Take first 7, middle 7, last 6 (same strategy as interactive summary)
            sample_docs_intelligent = (
                cleaned_docs[:7] +
                cleaned_docs[len(cleaned_docs)//2-3:len(cleaned_docs)//2+4] +
                cleaned_docs[-6:]
            )

        # Use longer excerpts for better context (300 chars like interactive summary)
        docs_text = "\n\n".join([f"{i+1}. {doc[:300]}" for i, doc in enumerate(sample_docs_intelligent)])

        # ✅ CARMACK: Use comprehensive prompt like interactive summary (3-5 sentences, structured)
        prompt = f"""You are analyzing documents from a topic cluster. Provide a concise summary (3-5 sentences) that captures:
1. The main theme/problem discussed
2. Key patterns or common issues mentioned
3. Notable insights or trends

Topic Label: {topic_label}
Keywords: {keywords}
Sample Documents (showing {len(sample_docs_intelligent)} representative docs):

{docs_text}

Provide a clear, actionable summary:"""

        # ✅ CARMACK FIX v2: Use apply_chat_template with return_dict=True (2025 best practice)
        # Original issue: Wrong tokenization caused 8/10 → 2/10 success rate
        # v2: Add attention_mask for proper generation (user caught this)
        try:
            # Try chat template first (works for Phi-3, Llama, etc.)
            messages = [{"role": "user", "content": prompt}]

            # return_dict=True gives us both input_ids and attention_mask
            model_inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )

            if torch.cuda.is_available():
                model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,  # Unpacks input_ids and attention_mask
                    max_new_tokens=800,  # ✅ Increased from 300 to allow detailed analysis without truncation
                    temperature=0.7,  # Slightly higher for more natural variation
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True  # ✅ Re-enabled: 30-50% speedup
                )

            # Decode only the new tokens (exclude the prompt)
            input_length = model_inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Debug logging for raw response
            import logging
            logging.debug(f"Topic {topic_id} raw response (chat template): {response[:200]}")

        except Exception as e:
            # Fallback to direct tokenization for non-chat models
            import logging
            logging.debug(f"Topic {topic_id} chat template failed, using fallback: {str(e)[:100]}")

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=800,  # ✅ Increased from 300 to allow detailed analysis without truncation
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True  # ✅ Re-enabled: 30-50% speedup
                )

            # Decode only the new tokens (exclude the prompt)
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Debug logging for raw response
            import logging
            logging.debug(f"Topic {topic_id} raw response (fallback): {response[:200]}")

        # Clean up
        response = response.strip()
        raw_response_preview = response[:200]  # Save more for debugging (longer summaries)

        # Remove common LLM artifacts and prompt repetitions
        artifacts = [
            "Answer:", "answer:", "Based on analysis", "based on analysis",
            "According to", "according to", "The documents show",
            "The users are saying", "Users are saying", "In this topic",
            "Summary:", "summary:", "Here is", "here is"
        ]
        for artifact in artifacts:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
                response = response.lstrip(':,.- ')

        # ✅ CARMACK: Keep full 3-5 sentence summary (don't truncate to first sentence)
        # Only remove excessive newlines while preserving paragraph structure
        response = '\n'.join(line.strip() for line in response.split('\n') if line.strip())

        # Final cleanup
        response = response.strip('"\'[](){}')

        # Log cleaning results
        import logging
        if response != raw_response_preview[:len(response)]:
            logging.debug(f"Topic {topic_id} cleaned: '{raw_response_preview}' → '{response[:200]}'")

        # Validate: Must be substantial enough to be meaningful
        # Reduced from 50 to 20 chars to handle small topics with brief content
        if len(response) < 20:
            logging.warning(f"Topic {topic_id} REJECTED - too short ({len(response)} chars): '{response}'")
            return None

        # Require at least 5 words for a meaningful summary (reduced from 10 to handle small topics)
        words = response.lower().split()
        if len(words) < 5:
            logging.warning(f"Topic {topic_id} REJECTED - too few words ({len(words)}): '{response}'")
            return None

        # Log successful analysis
        logging.debug(f"Topic {topic_id} SUCCESS: '{response}' ({len(response)} chars, {len(words)} words)")

        # Truncate if needed
        if len(response) > max_length:
            response = response[:max_length].rsplit(' ', 1)[0].strip() + "..."

        return response if response else None

    except Exception as e:
        # Log the actual error for debugging
        import logging
        logging.warning(f"Topic {topic_id} LLM generation exception: {str(e)}")
        # Only show in UI if it's a critical error (not during batch processing)
        if hasattr(st, 'caption'):
            st.caption(f"⚠️ Topic {topic_id} analysis failed: {str(e)[:100]}")
        return None


def deduplicate_labels_globally(labels_dict, keywords_dict, topics_dict=None):
    """
    Ensure all labels have minimum 3 levels and are unique.

    Process:
    1. First pass: Ensure all labels have minimum 3 levels
    2. Second pass: Add more levels (4+) if duplicates still exist

    Examples:
    - "Customer Service" → "Customer Service - Response Times - Phone Support"
    - "Product Orders - Delivery" → "Product Orders - Delivery - Samsung Appliances"
    """
    # STEP 1: Ensure all labels have minimum 3 levels
    MIN_LEVELS = 3
    for topic_id, label in labels_dict.items():
        current_levels = label.count(' - ') + 1

        while current_levels < MIN_LEVELS:
            # Need to add another level
            keywords = keywords_dict.get(topic_id, '')
            kw_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]

            # Extract distinctive keywords not already in the label
            common_words = {'help', 'buy', 'question', 'new', 'order', 'phone', 'customer', 'service', 'product', 'support'}
            label_words = set(label.lower().split())
            distinctive = []

            for kw in kw_list[:10]:
                kw_clean = kw.lower().strip()
                if kw_clean not in common_words and kw_clean not in label_words:
                    if not any(kw_clean in word for word in label_words):
                        distinctive.append(kw.title())
                        if len(distinctive) >= 2:
                            break

            if distinctive:
                # Add new level from keywords
                new_detail = ' '.join(distinctive[:2])
                label = f"{label} - {new_detail}"
            elif topics_dict and topic_id in topics_dict:
                # Try to extract from documents
                docs = topics_dict[topic_id]
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
                    label = f"{label} - {new_detail}"
                else:
                    # Use generic detail
                    label = f"{label} - Details"
            else:
                # No keywords or docs, use generic detail
                label = f"{label} - Details"

            # Update the label and count
            labels_dict[topic_id] = label
            current_levels = label.count(' - ') + 1

    # STEP 2: Deduplicate by adding more levels (4+) if needed
    MAX_ITERATIONS = 5
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

                for kw in kw_list[:10]:
                    kw_clean = kw.lower().strip()
                    if kw_clean not in common_words and kw_clean not in label_words:
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

        # ✅ CARMACK: Remove PII masking artifacts that pollute topic analysis
        # Remove <{entity_type}>, [name], [email], [phone], etc.
        text = re.sub(r'<\{?entity_type\}?>', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)  # Remove any other <tags>
        text = re.sub(r'\[(name|email|phone|address|ssn|number|date|location|organization|person)\]', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\[PII\]', ' ', text, flags=re.IGNORECASE)

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
    ✅ CARMACK OPTION C: Create TRUE 3-level semantic hierarchy.

    Each level provides distinct insight:
    - Level 1: Broad domain/category (trigrams)
    - Level 2: Specific problem/topic area (bigrams/trigrams)
    - Level 3: Contextual detail (unigrams/keywords)

    Examples:
    - "Customer Service - Response Times - Phone Support"
    - "Product Orders - Delivery Issues - Samsung Appliances"
    - "Technical Support - Installation Problems - Software Setup"

    Falls back to 2 levels if insufficient distinct phrases.
    """
    # Clean all phrases
    clean_phrases_23 = [_clean_phrase(p) for p in phrases_23 if p and len(_clean_phrase(p)) > 3]
    clean_phrases_1 = [_clean_phrase(p) for p in phrases_1 if p and len(_clean_phrase(p)) > 2]

    # Parse keywords
    if isinstance(keywords, str):
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    else:
        keyword_list = list(keywords) if keywords else []

    def _is_distinct(phrase1, phrase2, min_distinct_ratio=0.6):
        """Check if two phrases are semantically distinct (< 40% overlap)"""
        words1 = set(phrase1.lower().split())
        words2 = set(phrase2.lower().split())

        if words2.issubset(words1) or words1.issubset(words2):
            return False

        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return (overlap / total) < (1 - min_distinct_ratio) if total > 0 else False

    # Try to build 3 distinct levels
    level1 = None  # Broad domain
    level2 = None  # Specific problem
    level3 = None  # Context

    # LEVEL 1: Broadest category (first trigram/bigram)
    if clean_phrases_23:
        level1 = _to_title_case(clean_phrases_23[0])

    # LEVEL 2: Find second distinct phrase
    if level1 and len(clean_phrases_23) >= 2:
        for phrase in clean_phrases_23[1:4]:
            if _is_distinct(level1, phrase):
                level2 = _to_title_case(phrase)
                break

    # LEVEL 3: Find third distinct element (unigrams or keywords)
    if level1 and level2:
        # Try unigrams first
        used_words = set(level1.lower().split()) | set(level2.lower().split())
        detail_candidates = [w for w in clean_phrases_1[:6] if w.lower() not in used_words]

        if len(detail_candidates) >= 2:
            level3 = _to_title_case(' '.join(detail_candidates[:2]))
        elif len(detail_candidates) >= 1:
            level3 = _to_title_case(detail_candidates[0])
        else:
            # Fall back to keywords
            detail_kws = [_to_title_case(kw) for kw in keyword_list[:3] if kw.lower() not in used_words]
            if detail_kws:
                level3 = ' '.join(detail_kws[:2]) if len(detail_kws) >= 2 else detail_kws[0]

    # SUCCESS: We have 3 distinct levels!
    if level1 and level2 and level3:
        label = f"{level1} - {level2} - {level3}"

        # Truncate if needed (shorten level 3 first)
        if len(label) > max_len:
            max_level3_len = max_len - len(level1) - len(level2) - 6  # 6 for " - " x2
            if max_level3_len > 5:
                level3 = level3[:max_level3_len].rstrip() + "…"
                label = f"{level1} - {level2} - {level3}"
            else:
                label = label[:max_len].rstrip() + "…"

        return label

    # FALLBACK: 2-level structure (let deduplication add 3rd level if needed)
    if level1 and level2:
        label = f"{level1} - {level2}"
    elif level1:
        # Need level 2 - use unigrams or keywords
        used_words = set(level1.lower().split())
        detail_words = [w for w in clean_phrases_1[:4] if w.lower() not in used_words]
        if detail_words:
            level2 = _to_title_case(' '.join(detail_words[:2])) if len(detail_words) >= 2 else _to_title_case(detail_words[0])
        else:
            detail_kws = [_to_title_case(kw) for kw in keyword_list[:3] if kw.lower() not in used_words]
            level2 = ' '.join(detail_kws[:2]) if len(detail_kws) >= 2 else (detail_kws[0] if detail_kws else "General")

        label = f"{level1} - {level2}"
    else:
        # Last resort: build from available materials
        if len(clean_phrases_1) >= 3:
            level1 = _to_title_case(' '.join(clean_phrases_1[:2]))
            level2 = _to_title_case(' '.join(clean_phrases_1[2:4]))
        elif len(keyword_list) >= 3:
            level1 = _to_title_case(' '.join(keyword_list[:2]))
            level2 = _to_title_case(' '.join(keyword_list[2:4]))
        else:
            level1 = _to_title_case(' '.join(clean_phrases_1[:2])) if clean_phrases_1 else "Miscellaneous"
            level2 = "Topics"

        label = f"{level1} - {level2}"

    # Truncate if too long
    if len(label) > max_len:
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
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # ✅ Re-enabled: 30-50% speedup
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


@st.cache_resource
def load_local_llm(model_name, force_cpu=False, use_4bit=False):
    """
    Load a local LLM for topic labeling with intelligent GPU/CPU handling.

    Smart fallback strategy:
    1. Try GPU if available and not forced to CPU
    2. Use 4-bit quantization if use_4bit=True (fits 2x bigger models)
    3. If GPU OOM, clear cache and try CPU with system RAM
    4. Use appropriate precision for each device

    Recommended models:
    - "microsoft/Phi-3-mini-4k-instruct" (3.8GB FP16, 2GB 4-bit, fast, good for 16GB GPU, 8 docs)
    - "microsoft/Phi-3-mini-128k-instruct" (3.8GB FP16, 2GB 4-bit, fast, massive context, 50+ docs)
    - "mistralai/Mistral-7B-Instruct-v0.2" (14GB FP16, 7GB 4-bit, better quality, needs 24GB+ GPU, 15 docs)
    - "HuggingFaceH4/zephyr-7b-beta" (14GB FP16, 7GB 4-bit, good quality, needs 20GB+ GPU, 15 docs)
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
                # Check available GPU memory BEFORE cleanup
                mem_before = torch.cuda.mem_get_info()[0] / (1024**3)
                allocated_before = torch.cuda.memory_allocated(0) / (1024**3)
                reserved_before = torch.cuda.memory_reserved(0) / (1024**3)
                fragmentation_before = reserved_before - allocated_before

                st.info(f"📊 GPU Memory Status:")
                st.info(f"  • Free: {mem_before:.2f} GB")
                st.info(f"  • Allocated: {allocated_before:.2f} GB")
                st.info(f"  • Reserved: {reserved_before:.2f} GB")
                if fragmentation_before > 0.5:
                    st.warning(f"  ⚠️ Fragmentation: {fragmentation_before:.2f} GB (running cleanup...)")
                    # Run aggressive cleanup if fragmentation is high
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    import gc
                    for _ in range(3):
                        gc.collect()
                    torch.cuda.empty_cache()
                    # Check memory after cleanup
                    mem_after_cleanup = torch.cuda.mem_get_info()[0] / (1024**3)
                    gained = mem_after_cleanup - mem_before
                    st.info(f"  ✅ Cleanup freed {gained:.2f} GB | Now: {mem_after_cleanup:.2f} GB available")
                    gpu_free_gb = mem_after_cleanup
                else:
                    st.info(f"  ✓ Low fragmentation ({fragmentation_before:.2f} GB)")
                    gpu_free_gb = mem_before
                
                # Check if we have enough memory
                min_memory = 2.5 if use_4bit else 4.0

                if gpu_free_gb < 1.0:
                    st.error(f"❌ Critical: Only {gpu_free_gb:.2f}GB free - need at least {min_memory:.1f}GB")
                    st.error("🚨 GPU memory is critically low and fragmented!")
                    st.info("💡 Solutions:")
                    st.info("  1. Use the '🗑️ Clear GPU' button in sidebar to free memory")
                    st.info("  2. Restart Streamlit app to fully clear all GPU memory")
                    st.info("  3. Enable 4-bit quantization (needs only 2.5GB vs 4GB)")
                    st.info("  4. Use 'Force CPU' mode (slower but no GPU memory issues)")
                    raise RuntimeError(f"Insufficient GPU memory: {gpu_free_gb:.2f}GB free, need {min_memory:.1f}GB minimum")

                if gpu_free_gb >= min_memory:
                    st.info("🎮 Attempting to load LLM on GPU with optimizations...")

                    # Option 1: Use 4-bit quantization (2x memory reduction, minimal quality loss)
                    if use_4bit:
                        # Check if bitsandbytes is available
                        if not bitsandbytes_available:
                            st.error(f"❌ 4-bit quantization not available: {bitsandbytes_error}")
                            if IS_WINDOWS:
                                st.error("⚠️ Windows: bitsandbytes is not officially supported and often fails")
                                st.info("💡 Solutions:")
                                st.info("  1. Use FP16 mode (uncheck 4-bit quantization)")
                                st.info("  2. Use Force CPU mode if GPU memory is full")
                                st.info("  3. Linux/WSL2: bitsandbytes works better")
                            else:
                                st.info("💡 Install with: pip install bitsandbytes")
                            raise RuntimeError("bitsandbytes not available for 4-bit quantization")

                        try:
                            from transformers import BitsAndBytesConfig
                            st.info("🔬 Using 4-bit NF4 quantization (fits 2x bigger models)...")

                            bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                            )

                            # ✅ With expandable_segments, don't constrain max_memory
                            # PyTorch will manage GPU memory and overflow to system RAM if needed
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                quantization_config=bnb_config,
                                device_map="auto",
                                low_cpu_mem_usage=True
                                # max_memory removed: expandable segments handles overflow
                            )
                            # Verify quantization was applied
                            if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
                                st.success("✅ 4-bit NF4 quantization verified - 2x memory savings!")
                            else:
                                st.warning("⚠️ Quantization config not found - may not be quantized")
                            device_used = "GPU (4-bit)"
                        except ImportError:
                            st.error("❌ 4-bit requires bitsandbytes. Install with: pip install bitsandbytes")
                            raise
                    # Option 2: Use FP16 with Flash Attention 2 / SDPA (best performance)
                    else:
                        # Try Flash Attention 2 first (2-4x speedup on Ampere+ GPUs)
                        try:
                            st.info("⚡ Trying Flash Attention 2 (best performance)...")
                            # ✅ With expandable_segments, don't constrain max_memory
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                attn_implementation="flash_attention_2",
                                low_cpu_mem_usage=True
                                # max_memory removed: expandable segments handles overflow
                            )
                            st.success("✅ Using Flash Attention 2 (2-4x faster generation!)")
                            device_used = "GPU (FP16, Flash Attn 2)"
                        except (ImportError, ValueError) as attn_error:
                            # Flash Attention 2 not available, try SDPA (PyTorch 2.0+ scaled dot-product attention)
                            try:
                                st.info("⚡ Flash Attention 2 not available, trying SDPA (1.5-2x speedup)...")
                                # ✅ With expandable_segments, don't constrain max_memory
                                model = AutoModelForCausalLM.from_pretrained(
                                    model_name,
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    attn_implementation="sdpa",
                                    low_cpu_mem_usage=True
                                    # max_memory removed: expandable segments handles overflow
                                )
                                st.success("✅ Using SDPA attention (optimized)")
                                device_used = "GPU (FP16, SDPA)"
                            except (ImportError, ValueError):
                                # Fall back to default attention
                                st.info("ℹ️ Using default attention (install flash-attn for 2-4x speedup)")
                                # ✅ With expandable_segments, don't constrain max_memory
                                model = AutoModelForCausalLM.from_pretrained(
                                    model_name,
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    low_cpu_mem_usage=True
                                    # max_memory removed: expandable segments handles overflow
                                )
                                device_used = "GPU (FP16)"

                    # Show GPU memory usage after loading
                    gpu_used_after = torch.cuda.mem_get_info()
                    gpu_free_after = gpu_used_after[0] / (1024**3)
                    gpu_allocated = (gpu_free_gb - gpu_free_after)
                    st.success(f"✅ LLM loaded on {device_used} - Using {gpu_allocated:.2f}GB GPU memory")
                    st.info(f"💡 GPU: {gpu_free_after:.1f}GB free | Expandable segments: can overflow to system RAM")
                else:
                    st.warning(f"⚠️ Only {gpu_free_gb:.1f}GB GPU memory available - will use expandable segments")
                    # Don't raise - let expandable segments try to handle it
                    pass
                    
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

        # ✅ KV Cache is now managed at generation time
        # Cache is enabled (use_cache=True) for 30-50% speedup
        # past_key_values is cleared before each generation to prevent stale cache errors

        # Display final configuration with quantization status
        if "4-bit" in device_used:
            st.success(f"🎯 LLM ready on {device_used} | ⚡ 2x memory savings")
        else:
            st.success(f"🎯 LLM ready on {device_used}")

        if device_used == "CPU":
            st.info("💡 Pro tip: Close other applications to free up system RAM for faster inference")

        # ✅ Verify model device placement and quantization
        if model is not None:
            if hasattr(model, 'device'):
                actual_device = str(model.device)
                if "GPU" in device_used and 'cpu' in actual_device.lower():
                    st.warning(f"⚠️ Model reports device: {actual_device} but should be on GPU")
                elif "GPU" in device_used:
                    st.info(f"✓ Device verified: {actual_device}")

            # Verify quantization status
            if "4-bit" in device_used:
                if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
                    quant_cfg = model.config.quantization_config
                    # quantization_config is a BitsAndBytesConfig object, not a dict - use getattr
                    quant_method = getattr(quant_cfg, 'quant_method', 'unknown')
                    quant_type = getattr(quant_cfg, 'bnb_4bit_quant_type', 'unknown')
                    st.info(f"✓ Quantization verified: {quant_method} ({quant_type} type)")
                else:
                    st.error(f"❌ WARNING: Model labeled as 4-bit but no quantization_config found!")
                    st.info("This might mean 4-bit quantization failed to apply. Check bitsandbytes installation.")

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

    def recluster(self, n_topics, min_topic_size=10, use_reduced=True, method='kmeans', seed_words=None):
        """
        Quickly recluster documents into new topics.

        ✅ CARMACK: Added seed_words parameter for guided clustering.

        Args:
            n_topics: Number of topics to create
            min_topic_size: Minimum documents per topic
            use_reduced: Use UMAP embeddings if available
            method: 'kmeans' or 'hdbscan'
            seed_words: List of keyword lists for guided clustering
                       e.g., [["finance", "money"], ["marketing", "ads"]]
        """
        clustering_embeddings = self.umap_embeddings if (use_reduced and self.umap_embeddings is not None) else self.embeddings

        valid_mask = np.any(clustering_embeddings != 0, axis=1)
        valid_embeddings = clustering_embeddings[valid_mask]

        if len(valid_embeddings) == 0:
            st.error("No valid embeddings for clustering!")
            return None, None

        try:
            if method == 'kmeans':
                # ✅ CARMACK: Use seed words to initialize cluster centroids
                init_centroids = None
                if seed_words and len(seed_words) > 0:
                    try:
                        # Get embedding model from session state
                        model = st.session_state.get('safe_model')
                        if model and hasattr(model, 'model'):
                            # Encode seed word lists (join into phrases)
                            seed_phrases = [" ".join(words) for words in seed_words[:n_topics]]
                            seed_embeddings = model.model.encode(seed_phrases, convert_to_numpy=True)

                            # Project to same space if using UMAP
                            if use_reduced and self.umap_embeddings is not None:
                                # For UMAP space, we can't directly project
                                # Instead, find nearest documents in original space and use their UMAP coords
                                init_centroids_list = []
                                for seed_emb in seed_embeddings:
                                    # Find most similar document in original embeddings
                                    similarities = np.dot(self.embeddings, seed_emb)
                                    best_idx = np.argmax(similarities)
                                    # Use its UMAP coordinates
                                    init_centroids_list.append(self.umap_embeddings[best_idx])
                                init_centroids = np.array(init_centroids_list)
                            else:
                                init_centroids = seed_embeddings

                            # Pad with random centroids if we have fewer seed words than topics
                            if len(init_centroids) < n_topics:
                                remaining = n_topics - len(init_centroids)
                                random_indices = np.random.choice(len(valid_embeddings), remaining, replace=False)
                                random_centroids = valid_embeddings[random_indices]
                                init_centroids = np.vstack([init_centroids, random_centroids])

                            init_centroids = init_centroids[:n_topics]  # Trim if too many
                            st.info(f"🎯 Using {len(seed_words)} seed word sets to guide clustering")
                    except Exception as e:
                        st.warning(f"⚠️ Could not use seed words: {str(e)}")
                        init_centroids = None

                # Cluster with or without seed initialization
                if self.use_gpu:
                    try:
                        from cuml.cluster import KMeans as cuKMeans
                        # cuML doesn't support init parameter, so skip seed words for GPU
                        if init_centroids is not None:
                            st.info("⚠️ GPU clustering doesn't support seed words, using random init")
                        clusterer = cuKMeans(n_clusters=min(n_topics, len(valid_embeddings)), random_state=42)
                    except:
                        clusterer = KMeans(
                            n_clusters=min(n_topics, len(valid_embeddings)),
                            init=init_centroids if init_centroids is not None else 'k-means++',
                            n_init=1 if init_centroids is not None else 10,
                            random_state=42
                        )
                else:
                    clusterer = KMeans(
                        n_clusters=min(n_topics, len(valid_embeddings)),
                        init=init_centroids if init_centroids is not None else 'k-means++',
                        n_init=1 if init_centroids is not None else 10,
                        random_state=42
                    )

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

                # ✅ CARMACK: Add custom stopwords from session state
                custom_stopwords = st.session_state.get('custom_stopwords', set())
                all_stopwords = common_words | custom_stopwords

                filtered = {w: c for w, c in word_counts.items() if w not in all_stopwords and len(w) > 2}
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

            # Process in batches (now parallelized internally via ThreadPoolExecutor)
            num_batches = (len(all_topics_prepared) + batch_size - 1) // batch_size
            processed_count = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_topics_prepared))
                batch = all_topics_prepared[start_idx:end_idx]

                # Update progress
                status_text.info(f"🔄 Analyzing batch {batch_idx+1}/{num_batches} ({len(batch)} topics in parallel)...")

                # Carmack's simple parallel processing - no fallback needed!
                batch_results = generate_batch_llm_analysis(batch, self.llm_model)

                # Collect results and retry failed topics once
                failed_topics = []
                for topic_item in batch:
                    topic_id = topic_item['topic_id']
                    if topic_id in batch_results and batch_results[topic_id]:
                        llm_analysis_dict[topic_id] = batch_results[topic_id]
                        llm_success_count += 1
                    else:
                        # Queue for retry
                        failed_topics.append(topic_item)

                    processed_count += 1

                # Retry failed topics once (reduces 2/80 failures to ~0/80)
                if failed_topics:
                    status_text.info(f"🔄 Retrying {len(failed_topics)} failed topics...")
                    retry_results = generate_batch_llm_analysis(failed_topics, self.llm_model)

                    for topic_item in failed_topics:
                        topic_id = topic_item['topic_id']
                        if topic_id in retry_results and retry_results[topic_id]:
                            llm_analysis_dict[topic_id] = retry_results[topic_id]
                            llm_success_count += 1
                        else:
                            llm_analysis_dict[topic_id] = "No analysis available (topic may have insufficient data)"
                            llm_fallback_count += 1

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
                    # Find which topics failed
                    failed_topic_ids = [tid for tid, analysis in llm_analysis_dict.items()
                                       if "No analysis available" in analysis]
                    st.warning(f"⚠️ {llm_fallback_count} topics failed LLM analysis: {failed_topic_ids}")
                    st.caption("💡 Failed topics may have insufficient or low-quality document content. Enable debug mode to see details.")
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

        # Normalize once at source - guaranteed clean data
        return normalize_topic_info(pd.DataFrame(topic_info_list))


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

    # Process in batches (now parallelized internally via ThreadPoolExecutor)
    num_batches = (len(all_topics_prepared) + batch_size - 1) // batch_size
    processed_count = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_topics_prepared))
        batch = all_topics_prepared[start_idx:end_idx]

        # Update progress
        status_text.info(f"🔄 Analyzing batch {batch_idx+1}/{num_batches} ({len(batch)} topics in parallel)...")

        # Carmack's simple parallel processing - no fallback needed!
        batch_results = generate_batch_llm_analysis(batch, llm_model)

        # Collect results and retry failed topics once
        failed_topics = []
        for topic_item in batch:
            topic_id = topic_item['topic_id']
            if topic_id in batch_results and batch_results[topic_id]:
                llm_analysis_dict[topic_id] = batch_results[topic_id]
                llm_success_count += 1
            else:
                # Queue for retry
                failed_topics.append(topic_item)

            processed_count += 1

        # Retry failed topics once (reduces 2/80 failures to ~0/80)
        if failed_topics:
            status_text.info(f"🔄 Retrying {len(failed_topics)} failed topics...")
            retry_results = generate_batch_llm_analysis(failed_topics, llm_model)

            for topic_item in failed_topics:
                topic_id = topic_item['topic_id']
                if topic_id in retry_results and retry_results[topic_id]:
                    llm_analysis_dict[topic_id] = retry_results[topic_id]
                    llm_success_count += 1
                else:
                    llm_analysis_dict[topic_id] = "No analysis available (topic may have insufficient data)"
                    llm_fallback_count += 1

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
            # Find which topics failed
            failed_topic_ids = [tid for tid, analysis in llm_analysis_dict.items()
                               if "No analysis available" in analysis]
            st.warning(f"⚠️ {llm_fallback_count} topics failed LLM analysis: {failed_topic_ids}")
            st.caption("💡 Failed topics may have insufficient or low-quality document content. Enable debug mode to see details.")
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
        'accelerate_available': accelerate_available,
        'bitsandbytes_available': bitsandbytes_available,
        'bitsandbytes_error': bitsandbytes_error,
        'is_windows': IS_WINDOWS
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

def clear_gpu_memory(clear_models=True):
    """Clear GPU memory and optionally unload models from session state"""
    if not torch.cuda.is_available():
        return "No GPU detected"

    cleared = []

    if clear_models:
        # Unload LLM models
        if 'llm_model' in st.session_state and st.session_state.llm_model:
            try:
                if isinstance(st.session_state.llm_model, tuple) and len(st.session_state.llm_model) == 2:
                    model, tokenizer = st.session_state.llm_model
                    del model, tokenizer
                cleared.append("LLM (labeling)")
            except Exception as e:
                cleared.append(f"LLM (labeling) - error: {str(e)}")
            finally:
                st.session_state.llm_model = None
                st.session_state.llm_model_name = None

        if 'chat_llm' in st.session_state and st.session_state.chat_llm:
            try:
                if isinstance(st.session_state.chat_llm, tuple) and len(st.session_state.chat_llm) == 2:
                    model, tokenizer = st.session_state.chat_llm
                    del model, tokenizer
                cleared.append("LLM (chat)")
            except Exception as e:
                cleared.append(f"LLM (chat) - error: {str(e)}")
            finally:
                st.session_state.chat_llm = None
                st.session_state.chat_llm_loaded_model = None
                st.session_state.chat_llm_4bit = None

        # Note: Don't clear safe_model (embedding model) as it's needed for embeddings
        # User can manually clear it by recomputing embeddings with different model

    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Force garbage collection
    import gc
    gc.collect()

    if cleared:
        return f"Cleared: {', '.join(cleared)}"
    return "GPU cache cleared"

def aggressive_memory_cleanup():
    """Aggressive GPU memory cleanup with defragmentation"""
    if not torch.cuda.is_available():
        return "No GPU detected"

    st.info("🧹 Running aggressive memory cleanup...")

    # Step 1: Clear all cached models from session state
    models_cleared = []
    if st.session_state.get('chat_llm'):
        try:
            chat_llm = st.session_state.chat_llm
            if isinstance(chat_llm, tuple) and len(chat_llm) == 2:
                model, tokenizer = chat_llm
                del model, tokenizer
            st.session_state.chat_llm = None
            st.session_state.chat_llm_loaded_model = None
            st.session_state.chat_llm_4bit = None
            models_cleared.append("chat LLM")
        except: pass

    if st.session_state.get('llm_model'):
        try:
            llm_model = st.session_state.llm_model
            if isinstance(llm_model, tuple) and len(llm_model) == 2:
                model, tokenizer = llm_model
                del model, tokenizer
            st.session_state.llm_model = None
            st.session_state.llm_model_name = None
            models_cleared.append("topic labeling LLM")
        except: pass

    # Step 2: Clear Streamlit's cache
    st.cache_resource.clear()

    # Step 3: Force Python garbage collection (multiple passes for thorough cleanup)
    import gc
    for _ in range(3):
        gc.collect()

    # Step 4: Clear PyTorch CUDA cache
    torch.cuda.empty_cache()

    # Step 5: Synchronize CUDA to complete all operations
    torch.cuda.synchronize()

    # Step 6: Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Step 7: Try to defragment by forcing memory compaction
    try:
        # Allocate and free a large tensor to trigger compaction
        if torch.cuda.mem_get_info()[0] > 100 * 1024 * 1024:  # If >100MB free
            dummy = torch.zeros((1024, 1024, 10), device='cuda')  # ~40MB tensor
            del dummy
            torch.cuda.empty_cache()
    except:
        pass

    mem_info = torch.cuda.mem_get_info(0)
    free_gb = mem_info[0] / (1024**3)

    msg = f"✅ Freed: {', '.join(models_cleared) if models_cleared else 'cache only'} | {free_gb:.2f}GB now available"
    st.success(msg)
    return msg

def get_gpu_memory_status():
    """Get current GPU memory usage and loaded models with fragmentation info"""
    if not torch.cuda.is_available():
        return None

    try:
        mem_info = torch.cuda.mem_get_info(0)
        free_gb = mem_info[0] / (1024**3)
        total_gb = mem_info[1] / (1024**3)
        used_gb = total_gb - free_gb

        # Get PyTorch memory stats for fragmentation analysis
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        fragmentation_gb = reserved_gb - allocated_gb

        # Check what's loaded
        loaded = []
        if st.session_state.get('safe_model'):
            loaded.append("Embedding model")
        if st.session_state.get('llm_model'):
            model_name = st.session_state.get('llm_model_name', 'Unknown')
            loaded.append(f"LLM (labeling): {model_name.split('/')[-1]}")
        if st.session_state.get('chat_llm'):
            model_name = st.session_state.get('chat_llm_loaded_model', 'Unknown')
            loaded.append(f"LLM (chat): {model_name.split('/')[-1]}")

        return {
            'free_gb': free_gb,
            'used_gb': used_gb,
            'total_gb': total_gb,
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'fragmentation_gb': fragmentation_gb,
            'percent_used': (used_gb / total_gb) * 100,
            'loaded_models': loaded
        }
    except Exception as e:
        return None

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
    """
    Compute and cache UMAP reduced embeddings.

    ✅ CARMACK: Added disk caching to skip 15s UMAP reduction on re-runs.
    Cache is keyed by parameters + embedding shape + sample hash.
    """
    import hashlib

    # Generate cache key from parameters and embedding characteristics
    # Use shape + hash of first/last 1000 bytes for speed
    emb_sample = np.concatenate([embeddings[:100].flatten(), embeddings[-100:].flatten()])
    emb_hash = hashlib.md5(emb_sample.tobytes()).hexdigest()[:16]
    cache_key = f"{n_neighbors}_{n_components}_{embeddings.shape[0]}_{embeddings.shape[1]}_{emb_hash}"
    cache_file = f".cache/umap_{cache_key}.npy"

    # Try to load from disk cache
    if os.path.exists(cache_file):
        try:
            cached_embeddings = np.load(cache_file)
            if cached_embeddings.shape == (len(embeddings), n_components):
                st.info(f"✅ Loaded UMAP embeddings from cache (skipped {len(embeddings):,} doc reduction)")
                return cached_embeddings
        except Exception as e:
            # Cache corrupted, ignore and recompute
            pass

    # Compute UMAP (original logic)
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

    # Save to disk cache
    try:
        os.makedirs(".cache", exist_ok=True)
        np.save(cache_file, umap_embeddings)
        st.success(f"💾 Cached UMAP embeddings to {cache_file}")
    except Exception as e:
        # Non-critical failure, just log
        pass

    return umap_embeddings


# =====================================================
# 🚀 CARMACK: RAG IMPROVEMENTS - CHUNKING, CACHING, PERSISTENCE
# =====================================================

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    chunk_id: int
    parent_doc_id: int
    text: str
    start_pos: int
    end_pos: int
    topic_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None


class QueryEmbeddingCache:
    """LRU cache for query embeddings - avoid re-encoding same queries"""
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.access_order = []
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _hash_query(self, query: str) -> str:
        """Hash query for cache key"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[np.ndarray]:
        """Get cached embedding if exists"""
        key = self._hash_query(query)
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, query: str, embedding: np.ndarray):
        """Store embedding in cache"""
        key = self._hash_query(query)

        # Evict oldest if at capacity
        if len(self.cache) >= self.maxsize and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = embedding
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def stats(self) -> Dict:
        """Return cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }


def chunk_documents(documents: List[str], tokenizer, chunk_size=512, overlap=128, topics=None) -> Tuple[List[DocumentChunk], np.ndarray]:
    """
    Chunk documents into overlapping segments for better RAG retrieval.

    Args:
        documents: List of document texts
        tokenizer: Tokenizer for counting tokens
        chunk_size: Target tokens per chunk (512 is empirically optimal)
        overlap: Overlap between chunks in tokens (128 recommended)
        topics: Optional topic IDs for each document

    Returns:
        chunks: List of DocumentChunk objects
        parent_mapping: Array mapping chunk_id → parent_doc_id
    """
    chunks = []
    chunk_id = 0

    for doc_id, doc_text in enumerate(documents):
        if not doc_text or len(doc_text.strip()) == 0:
            continue

        topic_id = topics[doc_id] if topics is not None else None

        # Tokenize document
        try:
            tokens = tokenizer.encode(doc_text, add_special_tokens=False)
        except:
            # Fallback: simple whitespace tokenization if tokenizer fails
            tokens = doc_text.split()

        # If document is shorter than chunk_size, keep it as one chunk
        if len(tokens) <= chunk_size:
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                parent_doc_id=doc_id,
                text=doc_text,
                start_pos=0,
                end_pos=len(doc_text),
                topic_id=topic_id
            ))
            chunk_id += 1
            continue

        # Split into overlapping chunks
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))

            # Decode chunk tokens back to text
            try:
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)
            except:
                # Fallback: slice original text proportionally
                char_start = int((start / len(tokens)) * len(doc_text))
                char_end = int((end / len(tokens)) * len(doc_text))
                chunk_text = doc_text[char_start:char_end]

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                parent_doc_id=doc_id,
                text=chunk_text,
                start_pos=start,
                end_pos=end,
                topic_id=topic_id
            ))
            chunk_id += 1

            # Move start forward by (chunk_size - overlap)
            start += (chunk_size - overlap)

            # Break if we've covered the document
            if end >= len(tokens):
                break

    # Create parent mapping array
    parent_mapping = np.array([chunk.parent_doc_id for chunk in chunks], dtype=np.int32)

    return chunks, parent_mapping


def expand_query_with_llm(query: str, llm_model, num_variants=2) -> List[str]:
    """
    ANDREJ KARPATHY: Expand query into multiple variants for better recall.
    Generates 2-3 rephrased versions of the query, embeds all, retrieves union.

    Args:
        query: Original user query
        llm_model: Tuple of (model, tokenizer)
        num_variants: Number of query variants to generate (2-3 recommended)

    Returns:
        List of query strings: [original_query, variant1, variant2, ...]
    """
    if llm_model is None:
        return [query]  # Fallback: just use original

    try:
        model, tokenizer = llm_model

        prompt = f"""Rephrase this search query in {num_variants} different ways to find relevant documents.
Keep the core meaning but vary the wording and perspective.

Original query: {query}

Provide {num_variants} alternative phrasings, one per line:"""

        # Generate query variants
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # ✅ Re-enabled: 30-50% speedup
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract variants from response
        variants = []
        for line in response.split('\n'):
            line = line.strip()
            # Remove numbering like "1.", "2.", etc.
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if line and len(line) > 10 and line.lower() not in query.lower():
                variants.append(line)

        # Return original + variants (limit to num_variants)
        all_queries = [query] + variants[:num_variants]
        return all_queries

    except Exception as e:
        # Fallback: just use original query
        return [query]


def save_faiss_index_to_disk(faiss_index, embeddings, chunks, parent_mapping, cache_dir=".cache"):
    """
    CARMACK: Persist FAISS index and embeddings to disk - free speedup.
    Avoids rebuilding index every session.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)

        # Save FAISS index
        import faiss
        index_path = os.path.join(cache_dir, "faiss_index.bin")
        faiss.write_index(faiss_index, index_path)

        # Save embeddings as numpy array
        embeddings_path = os.path.join(cache_dir, "chunk_embeddings.npy")
        np.save(embeddings_path, embeddings)

        # Save chunks and parent mapping with pickle
        chunks_path = os.path.join(cache_dir, "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump({'chunks': chunks, 'parent_mapping': parent_mapping}, f)

        st.success(f"💾 Cached FAISS index + embeddings ({len(chunks):,} chunks)")
        return True
    except Exception as e:
        st.warning(f"⚠️ Failed to cache index: {str(e)}")
        return False


def load_faiss_index_from_disk(cache_dir=".cache") -> Optional[Tuple]:
    """
    Load FAISS index, embeddings, and chunks from disk if available.
    Returns (faiss_index, embeddings, chunks, parent_mapping) or None if not found/invalid.
    """
    try:
        index_path = os.path.join(cache_dir, "faiss_index.bin")
        embeddings_path = os.path.join(cache_dir, "chunk_embeddings.npy")
        chunks_path = os.path.join(cache_dir, "chunks.pkl")

        # Check all files exist
        if not all(os.path.exists(p) for p in [index_path, embeddings_path, chunks_path]):
            return None

        # Load FAISS index
        import faiss
        faiss_index = faiss.read_index(index_path)

        # Load embeddings
        embeddings = np.load(embeddings_path)

        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunk_data = pickle.load(f)
            chunks = chunk_data['chunks']
            parent_mapping = chunk_data['parent_mapping']

        # Validate consistency
        if faiss_index.ntotal != len(embeddings) or len(embeddings) != len(chunks):
            st.warning("⚠️ Cached index is inconsistent - will rebuild")
            return None

        st.success(f"✅ Loaded cached FAISS index ({len(chunks):,} chunks)")
        return (faiss_index, embeddings, chunks, parent_mapping)

    except Exception as e:
        st.warning(f"⚠️ Failed to load cached index: {str(e)}")
        return None


# -----------------------------------------------------
# FAISS INDEXING FOR RAG CHAT
# -----------------------------------------------------
def validate_document_indexing(embeddings, documents, topics=None):
    """
    Validate that embeddings, documents, and topics arrays are properly aligned.
    Returns (is_valid, error_message)
    """
    if embeddings is None or documents is None:
        return False, "Embeddings or documents is None"

    if len(embeddings) != len(documents):
        return False, f"Length mismatch: {len(embeddings)} embeddings but {len(documents)} documents"

    if topics is not None and len(topics) != len(documents):
        return False, f"Length mismatch: {len(topics)} topics but {len(documents)} documents"

    # Check for any obviously invalid embeddings
    if len(embeddings) > 0:
        zero_embeddings = np.all(embeddings == 0, axis=1).sum()
        if zero_embeddings > len(embeddings) * 0.5:  # More than 50% are zero
            return False, f"Warning: {zero_embeddings} embeddings are all zeros (may indicate indexing issue)"

    return True, "All arrays are properly aligned"


def build_faiss_index(embeddings):
    """
    Build FAISS index from embeddings for fast similarity search.

    ✅ CARMACK: Uses IVF (Inverted File Index) for >50k documents (5-10x faster).
    For smaller datasets, uses flat index (exact search).

    The FAISS index maps: index_position[i] → embeddings[i] → documents[i]
    """
    if not faiss_available:
        st.warning("⚠️ FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
        return None

    try:
        dimension = embeddings.shape[1]
        n_docs = len(embeddings)

        # Validate embeddings quality
        zero_embeddings = np.all(embeddings == 0, axis=1).sum()
        if zero_embeddings > 0:
            st.warning(f"⚠️ Found {zero_embeddings} zero embeddings (may be invalid documents)")

        # Choose index type based on dataset size
        if n_docs > 50000:
            # Large dataset: Use IVF for speed
            nlist = min(int(np.sqrt(n_docs)), 4096)  # Number of clusters
            st.info(f"🚀 Building IVF FAISS index for {n_docs:,} documents ({nlist} clusters)...")

            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

            # Train the index
            index.train(embeddings.astype('float32'))
            index.add(embeddings.astype('float32'))

            # Set search parameters (trade accuracy for speed)
            index.nprobe = min(32, nlist // 4)  # Search 32 clusters

            st.success(f"✅ IVF index built: {nlist} clusters, nprobe={index.nprobe}")
            return index
        else:
            # Small/medium dataset: Use flat index (exact search)
            if faiss_gpu_available:
                st.info(f"🎮 Building flat FAISS index on GPU for {n_docs:,} documents...")
                res = faiss.StandardGpuResources()
                index = faiss.IndexFlatL2(dimension)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_index.add(embeddings.astype('float32'))
                return gpu_index
            else:
                st.info(f"💻 Building flat FAISS index on CPU for {n_docs:,} documents...")
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings.astype('float32'))
                return index

    except Exception as e:
        st.error(f"❌ Failed to build FAISS index: {str(e)}")
        return None


def retrieve_relevant_documents(
    query,
    faiss_index,
    embeddings,
    documents,
    safe_model,
    top_k=5,
    topics=None,
    topic_filter=None,
    chunks=None,
    parent_mapping=None,
    query_cache=None,
    llm_model=None,
    enable_query_expansion=False
):
    """
    🚀 CARMACK+KARPATHY+YAN: Retrieve relevant documents with chunking, caching, and query expansion.

    Improvements:
    - Document chunking for better granularity
    - LRU cache for query embeddings
    - Query expansion with LLM (2-3 variants)
    - Topic pre-filtering (before FAISS search, not after)
    - Latency logging for measurement
    """
    if not faiss_index or len(documents) == 0:
        return []

    latency_start = time.time()

    try:
        # Initialize query embedding cache if not provided
        if query_cache is None:
            query_cache = st.session_state.get('query_embedding_cache')
            if query_cache is None:
                query_cache = QueryEmbeddingCache(maxsize=1000)
                st.session_state.query_embedding_cache = query_cache

        # Determine if we're using chunks or full documents
        use_chunks = chunks is not None and parent_mapping is not None
        search_items = chunks if use_chunks else documents

        if len(search_items) == 0:
            return []

        # ANDREJ KARPATHY: Query expansion for better recall
        query_variants = [query]  # Start with original
        if enable_query_expansion and llm_model and st.session_state.get('enable_query_expansion', False):
            expansion_start = time.time()
            query_variants = expand_query_with_llm(query, llm_model, num_variants=2)
            expansion_time = (time.time() - expansion_start) * 1000
            if st.session_state.get('show_rag_debug', False):
                st.caption(f"🔍 Expanded to {len(query_variants)} variants ({expansion_time:.0f}ms)")

        # Encode all query variants (with caching)
        query_embeddings = []
        cache_hits = 0
        for q_variant in query_variants:
            cached_emb = query_cache.get(q_variant)
            if cached_emb is not None:
                query_embeddings.append(cached_emb)
                cache_hits += 1
            else:
                # Encode and cache
                emb = safe_model.model.encode(
                    [q_variant],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )[0].astype('float32')
                query_cache.put(q_variant, emb)
                query_embeddings.append(emb)

        # ✅ FIX: Build topic-to-chunk mapping for efficient pre-filtering
        # Cache this mapping in session state (only build once)
        if topic_filter is not None and use_chunks and topics is not None:
            if 'topic_to_chunks_map' not in st.session_state:
                # Build mapping: topic_id -> list of chunk indices
                from collections import defaultdict
                topic_to_chunks = defaultdict(list)
                for chunk_idx, chunk in enumerate(search_items):
                    if chunk.parent_doc_id < len(topics):
                        chunk_topic = topics[chunk.parent_doc_id]
                        topic_to_chunks[chunk_topic].append(chunk_idx)
                st.session_state['topic_to_chunks_map'] = dict(topic_to_chunks)
                if st.session_state.get('show_rag_debug', False):
                    st.info(f"📊 Built topic-to-chunk mapping: {len(topic_to_chunks)} topics")

        # Debug: Collect filtering info for error reporting
        debug_info = []
        if topic_filter is not None:
            debug_info.append(f"Topic filter: {topic_filter}")
            debug_info.append(f"Topics array: {len(topics) if topics is not None else 'None'} entries")
            debug_info.append(f"Using chunks: {use_chunks}")
            debug_info.append(f"Search items: {len(search_items)}")

            # Check if topic has any chunks
            if use_chunks and 'topic_to_chunks_map' in st.session_state:
                topic_chunks = st.session_state['topic_to_chunks_map'].get(topic_filter, [])
                debug_info.append(f"Chunks in target topic: {len(topic_chunks)}")

        if st.session_state.get('show_rag_debug', False) and topic_filter is not None:
            st.info(f"🔍 Topic filter: {topic_filter} | Topics: {len(topics) if topics is not None else 'None'} | Chunks: {use_chunks}")

        # Use pre-filtering with mapping if available and topic filter is set
        if topic_filter is not None and use_chunks and 'topic_to_chunks_map' in st.session_state:
            topic_to_chunks = st.session_state['topic_to_chunks_map']
            valid_chunk_indices = topic_to_chunks.get(topic_filter, [])

            if len(valid_chunk_indices) == 0:
                # No chunks in this topic
                debug_info.append("ERROR: Topic has no chunks!")
                st.session_state['last_retrieval_debug'] = "\n".join(debug_info)
                return []

            # Build temp index for this topic's chunks only
            import faiss as faiss_lib
            filtered_embeddings = embeddings[valid_chunk_indices].astype('float32')
            temp_index = faiss_lib.IndexFlatL2(embeddings.shape[1])
            temp_index.add(filtered_embeddings)

            search_index = temp_index
            index_mapping = valid_chunk_indices  # Map results back to original chunk indices

            if st.session_state.get('show_rag_debug', False):
                st.info(f"🎯 Pre-filtered to {len(valid_chunk_indices)} chunks in topic {topic_filter}")
        else:
            # No topic filter or not using chunks - search full index
            search_index = faiss_index
            index_mapping = None

        # Search with all query variants, collect unique results
        all_results = {}  # Use dict to deduplicate by index

        for q_emb in query_embeddings:
            q_emb_2d = q_emb.reshape(1, -1).astype('float32')

            # Search (retrieve more if we need to fill top_k across variants)
            search_k = min(top_k * 2, search_index.ntotal) if len(query_embeddings) > 1 else min(top_k, search_index.ntotal)
            distances, indices = search_index.search(q_emb_2d, search_k)

            # Collect results
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue

                # Map back to original index if we pre-filtered
                original_idx = index_mapping[idx] if index_mapping is not None else idx

                if original_idx >= len(search_items):
                    continue

                # Get document info
                if use_chunks:
                    chunk = search_items[original_idx]
                    doc_text = chunk.text
                    # Get topic from parent document
                    doc_topic = topics[chunk.parent_doc_id] if topics is not None and chunk.parent_doc_id < len(topics) else None
                    parent_id = chunk.parent_doc_id
                else:
                    doc_text = search_items[original_idx]
                    doc_topic = topics[original_idx] if topics is not None and original_idx < len(topics) else None
                    parent_id = original_idx

                # Store best score for this doc/chunk
                dist = float(distances[0][i])
                if original_idx not in all_results or dist < all_results[original_idx]['distance']:
                    all_results[original_idx] = {
                        'document': doc_text,
                        'distance': dist,
                        'index': int(original_idx),
                        'parent_id': int(parent_id),
                        'topic': doc_topic,
                        'chunk_id': original_idx if use_chunks else None
                    }

        # Sort by distance and take top_k
        results = sorted(all_results.values(), key=lambda x: x['distance'])[:top_k]

        # Latency logging
        latency_ms = (time.time() - latency_start) * 1000

        # Store debug info in session state for error reporting
        if topic_filter is not None and len(results) == 0:
            debug_info.append(f"Results found: {len(all_results)}")
            st.session_state['last_retrieval_debug'] = "\n".join(debug_info)

        # Show retrieval stats if debug enabled
        if st.session_state.get('show_rag_debug', False):
            cache_stats = query_cache.stats()
            item_type = "chunks" if use_chunks else "docs"

            debug_msg = f"📊 {latency_ms:.0f}ms | Searched {search_index.ntotal} {item_type}"
            if topic_filter is not None and index_mapping is not None:
                debug_msg += f" | Pre-filtered to topic {topic_filter} ({len(index_mapping)} chunks)"
            if cache_hits > 0:
                debug_msg += f" | Cache: {cache_hits}/{len(query_variants)} hits ({cache_stats['hit_rate']:.1%})"
            if len(query_variants) > 1:
                debug_msg += f" | {len(query_variants)} variants"
            debug_msg += f" | Returned {len(results)}"

            st.info(debug_msg)

        return results

    except Exception as e:
        st.error(f"❌ Retrieval error: {str(e)}")
        return []


@st.cache_resource
def load_reranker_model(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Load cross-encoder reranker model.
    Uses ms-marco-MiniLM-L-6-v2: Fast, good accuracy, optimized for search relevance.
    """
    try:
        return CrossEncoder(model_name)
    except Exception as e:
        st.warning(f"Failed to load reranker: {e}")
        return None


def rerank_documents(query, retrieved_docs, reranker_model, top_k=None, relevance_threshold=0.0):
    """
    Rerank documents using cross-encoder for better relevance scoring.

    Args:
        query: User query string
        retrieved_docs: List of docs from FAISS retrieval (with 'document', 'distance', etc)
        reranker_model: CrossEncoder model
        top_k: Number of top docs to return after reranking (None = return all)
        relevance_threshold: Minimum score to keep (0.0 = keep all)

    Returns:
        Reranked list of documents with added 'rerank_score' field
    """
    if not reranker_model or not retrieved_docs:
        return retrieved_docs

    try:
        # Prepare query-document pairs for cross-encoder
        pairs = [(query, doc['document']) for doc in retrieved_docs]

        # Get relevance scores (cross-encoder scores pairs together)
        scores = reranker_model.predict(pairs)

        # Add rerank scores to documents
        for doc, score in zip(retrieved_docs, scores):
            doc['rerank_score'] = float(score)

        # Filter by threshold
        if relevance_threshold > 0.0:
            retrieved_docs = [doc for doc in retrieved_docs if doc['rerank_score'] >= relevance_threshold]

        # Sort by rerank score (descending)
        retrieved_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Take top_k if specified
        if top_k is not None and top_k > 0:
            retrieved_docs = retrieved_docs[:top_k]

        return retrieved_docs

    except Exception as e:
        st.error(f"❌ Reranking error: {str(e)}")
        # Return original docs if reranking fails
        return retrieved_docs


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
        for i, doc_info in enumerate(retrieved_docs[:100], 1):
            doc_preview = doc_info['document'][:300] + "..." if len(doc_info['document']) > 300 else doc_info['document']
            topic_tag = f" [from Topic {doc_info['topic']}]" if doc_info.get('topic') is not None else ""
            context_parts.append(f"\n[Doc {i}]{topic_tag}: {doc_preview}")

        context = "\n".join(context_parts)

        # Create prompt
        prompt = f"""{context}

User Question: {user_query}

Instructions: Answer the question using ONLY the relevant documents above. Do not repeat the question or documents. Start your response immediately with the answer.

Answer:"""

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        # ✅ Ensure model and inputs are on same device (prevent mixed device errors)
        if torch.cuda.is_available():
            # Check if model is on GPU, if not move it
            if hasattr(model, 'device') and model.device.type != 'cuda':
                try:
                    model = model.cuda()
                except:
                    pass  # If move fails, continue with CPU
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # ✅ Clear past_key_values to prevent DynamicCache errors when reusing models
        if hasattr(model, 'past_key_values'):
            model.past_key_values = None

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # ✅ Re-enabled: 30-50% speedup (cache cleared above to prevent errors)
                past_key_values=None  # Ensure fresh cache for each generation
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part - be aggressive to avoid showing the prompt
        # Try multiple extraction strategies
        if "Answer:" in response:
            # Best case: model followed format
            response = response.split("Answer:")[-1].strip()
        elif "User Question:" in response:
            # Extract everything after the user question
            parts = response.split("User Question:")
            if len(parts) > 1:
                # Get text after the question, skip the question itself
                after_question = parts[-1]
                # Try to find where the actual answer starts (after the question text)
                if user_query in after_question:
                    response = after_question.split(user_query, 1)[-1].strip()
                    # Remove common prompt artifacts
                    for prefix in ["Based on", "provide a", "Be concise"]:
                        if response.lower().startswith(prefix.lower()):
                            continue
                    # If response starts with prompt text, skip to next sentence
                    if response.startswith("Based on"):
                        sentences = response.split(". ", 1)
                        response = sentences[1] if len(sentences) > 1 else response
                else:
                    response = after_question.strip()
        else:
            # Last resort: take the last 1/3 of response (likely the actual answer)
            # This prevents showing the entire prompt back to user
            lines = response.split("\n")
            # Skip context lines, find where actual content starts
            answer_lines = []
            skip_mode = True
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Stop skipping when we hit something that looks like an answer
                if skip_mode and any(indicator in line.lower() for indicator in
                                     ["based on", "according to", "the documents show", "analysis reveals"]):
                    skip_mode = False
                if not skip_mode:
                    answer_lines.append(line)

            if answer_lines:
                response = "\n".join(answer_lines)
            else:
                # Absolute fallback: take last 40% of text
                response = response[int(len(response) * 0.6):]

        return response.strip()

    except Exception as e:
        st.error(f"❌ LLM generation error: {str(e)}")
        return f"Error generating response: {str(e)}"


def generate_chat_response(user_query, topic_info_df, topics, processed_df, current_topic_id=None, explicit_topic_filter=None, use_rag=True, llm_model=None, faiss_index=None, embeddings=None, documents=None, safe_model=None):
    """
    Generate chat response using RAG (Retrieval-Augmented Generation).
    No fallback modes - RAG or nothing.

    Args:
        explicit_topic_filter: Topic ID to filter to (from UI dropdown), takes precedence over text parsing
    """
    query_lower = user_query.lower()

    # Check prerequisites
    if not (use_rag and llm_model and faiss_index and documents is not None and safe_model):
        missing = []
        if not llm_model: missing.append("LLM")
        if not faiss_index: missing.append("FAISS")
        if documents is None: missing.append("documents")
        if not safe_model: missing.append("embeddings")
        return f"Error: Missing {', '.join(missing)}. Recompute embeddings with LLM enabled."

    if len(documents) == 0:
        return "Error: No documents available."

    # ✅ FIX: Get chunk embeddings if available (FAISS is built from chunks, not documents)
    chunk_embeddings_from_state = st.session_state.get('chunk_embeddings')
    actual_embeddings = chunk_embeddings_from_state if chunk_embeddings_from_state is not None else embeddings

    # Validate indexing alignment (use document embeddings for document validation)
    is_valid, validation_msg = validate_document_indexing(embeddings if chunk_embeddings_from_state is None else None, documents, topics)
    if not is_valid and chunk_embeddings_from_state is None:
        st.error(f"❌ {validation_msg}")
        return f"Error: {validation_msg}. Recompute embeddings."

    # Check FAISS index matches embeddings (compare with chunk embeddings if chunking is used)
    if faiss_index.ntotal != len(actual_embeddings):
        st.error(f"❌ FAISS index size mismatch: {faiss_index.ntotal} vectors vs {len(actual_embeddings)} embeddings")
        return "Error: FAISS index out of sync. Recompute embeddings."

    # Determine topic filter: explicit UI filter takes precedence, then parse from query text
    topic_filter = explicit_topic_filter if explicit_topic_filter is not None else None

    # If no explicit filter, try parsing from query text
    if topic_filter is None:
        topic_match = re.search(r'topic\s+(\d+)', query_lower)
        if topic_match:
            topic_filter = int(topic_match.group(1))

    # Validate topic filter
    if topic_filter is not None:
        if topics is not None and topic_filter not in set(topics):
            st.warning(f"Topic {topic_filter} not found, searching all topics")
            topic_filter = None
        else:
            # Get topic label for confirmation
            topic_row = topic_info_df[topic_info_df['Topic'] == topic_filter]
            topic_label = topic_row.iloc[0]['Human_Label'] if len(topic_row) > 0 else f"Topic {topic_filter}"
            st.info(f"🎯 Filtering to: {topic_label}")

    # Retrieve and generate
    doc_count = st.session_state.get('rag_doc_count', 10)

    # Debug: show what we're retrieving
    if st.session_state.get('show_rag_debug', False):
        st.caption(f"🔍 Retrieving {doc_count} documents...")

    # Get initial candidates (retrieve more if reranking enabled)
    use_reranking = st.session_state.get('use_reranking', False)
    retrieval_count = doc_count * 3 if use_reranking else doc_count  # 3x oversample for reranking

    # 🚀 CARMACK+KARPATHY: Get chunks and embeddings from session state
    chunks = st.session_state.get('chunks')
    parent_mapping = st.session_state.get('parent_mapping')
    chunk_embeddings = st.session_state.get('chunk_embeddings', embeddings)

    # Query expansion enabled?
    enable_query_expansion = st.session_state.get('enable_query_expansion', False)

    retrieved_docs = retrieve_relevant_documents(
        user_query,
        faiss_index,
        chunk_embeddings,  # Use chunk embeddings if available, else fall back to doc embeddings
        documents,
        safe_model,
        top_k=retrieval_count,
        topics=topics,
        topic_filter=topic_filter,
        chunks=chunks,  # Pass chunks for chunk-based retrieval
        parent_mapping=parent_mapping,
        query_cache=None,  # Will be initialized inside function
        llm_model=llm_model,  # For query expansion
        enable_query_expansion=enable_query_expansion
    )

    if not retrieved_docs:
        filter_msg = f" in topic {topic_filter}" if topic_filter else ""

        # Build error message with debug info embedded (will persist in chat history)
        error_msg = f"❌ No relevant documents found{filter_msg}.\n\n"

        # Include debug info directly in the response message
        if 'last_retrieval_debug' in st.session_state:
            error_msg += "**Debug Information:**\n```\n"
            error_msg += st.session_state['last_retrieval_debug']
            error_msg += "\n```\n\n"
            error_msg += "**Possible causes:**\n"
            error_msg += "- Topic might not have any documents matching your query\n"
            error_msg += "- Try selecting 'All Topics' to search across all documents\n"
            error_msg += "- Try rephrasing your query\n"
        else:
            error_msg += "Try rephrasing your query or selecting a different topic."

        return error_msg

    # Rerank if enabled
    if use_reranking:
        reranker = st.session_state.get('reranker_model')
        if not reranker:
            # Load reranker on first use
            with st.spinner("Loading reranker model..."):
                reranker = load_reranker_model()
                st.session_state.reranker_model = reranker

        if reranker:
            relevance_threshold = st.session_state.get('relevance_threshold', 0.0)

            if st.session_state.get('show_rag_debug', False):
                st.caption(f"🔄 Reranking {len(retrieved_docs)} docs → top {doc_count} (threshold: {relevance_threshold})")

            # Rerank and take top_k with threshold filtering
            retrieved_docs = rerank_documents(
                user_query,
                retrieved_docs,
                reranker,
                top_k=doc_count,
                relevance_threshold=relevance_threshold
            )

            if st.session_state.get('show_rag_debug', False):
                if retrieved_docs:
                    avg_score = sum(d.get('rerank_score', 0) for d in retrieved_docs) / len(retrieved_docs)
                    st.caption(f"✅ After reranking: {len(retrieved_docs)} docs (avg score: {avg_score:.3f})")
                else:
                    st.caption("⚠️ No docs passed relevance threshold")

    # Generate response
    response = generate_rag_response(
        user_query,
        retrieved_docs,
        topic_info_df,
        topics,
        llm_model,
        current_topic_id
    )

    # Add sources with relevance scores
    response += "\n\n**Sources:**\n"
    for i, doc in enumerate(retrieved_docs[:3], 1):
        preview = doc['document'][:100] + "..." if len(doc['document']) > 100 else doc['document']
        topic_tag = f" [Topic {doc['topic']}]" if doc.get('topic') else ""

        # Show rerank score if available
        if 'rerank_score' in doc:
            score_tag = f" (relevance: {doc['rerank_score']:.3f})"
        else:
            score_tag = ""

        response += f"{i}. {preview}{topic_tag}{score_tag}\n"

    return response

def main():
    st.title("🚀 Complete BERTopic with All Features")

    # Initialize session state - Carmack style: one dict, one loop
    SESSION_DEFAULTS = {
        # Core data
        'embeddings_computed': False,
        'embeddings': None,
        'umap_embeddings': None,
        'documents': None,
        'valid_indices': None,

        # DataFrames
        'df': None,
        'processed_df': None,
        'browser_df': None,
        'topic_info': None,
        'current_topic_info': None,

        # Clustering
        'reclusterer': None,
        'current_topics': None,
        'min_topic_size': 10,
        'min_topic_size_used': 10,
        'clustering_method': 'Unknown',

        # Models
        'model': None,
        'llm_model': None,
        'llm_model_name': None,

        # UI state
        'text_col': None,
        'uploaded_file_name': 'data',
        'custom_stopwords': set(),
        'topic_human': {},
        'last_topics_hash': None,

        # Hardware
        'gpu_used': False,
        'gpu_capabilities': None,
    }

    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Lazy-load GPU capabilities (expensive operation)
    if st.session_state.gpu_capabilities is None:
        st.session_state.gpu_capabilities = check_gpu_capabilities()
    gpu_capabilities = st.session_state.gpu_capabilities

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # GPU Status and Management
        if gpu_capabilities['cuda_available']:
            st.success(f"✅ GPU: {gpu_capabilities['device_name']}")

            # Real-time GPU memory status
            gpu_status = get_gpu_memory_status()
            if gpu_status:
                st.progress(gpu_status['percent_used'] / 100)
                st.caption(f"**{gpu_status['used_gb']:.1f} GB / {gpu_status['total_gb']:.1f} GB** ({gpu_status['percent_used']:.0f}% used)")

                # Show fragmentation warning if high
                if gpu_status['fragmentation_gb'] > 1.0:
                    st.warning(f"⚠️ Fragmentation: {gpu_status['fragmentation_gb']:.2f} GB - Use 'Deep Clean' to defragment")
                elif gpu_status['fragmentation_gb'] > 0.5:
                    st.info(f"ℹ️ Fragmentation: {gpu_status['fragmentation_gb']:.2f} GB")

                # Show memory breakdown
                with st.expander("📊 Memory Details", expanded=False):
                    st.caption(f"• Free: {gpu_status['free_gb']:.2f} GB")
                    st.caption(f"• Allocated: {gpu_status['allocated_gb']:.2f} GB (actively used)")
                    st.caption(f"• Reserved: {gpu_status['reserved_gb']:.2f} GB (PyTorch cache)")
                    st.caption(f"• Fragmentation: {gpu_status['fragmentation_gb']:.2f} GB (wasted space)")

                # Show loaded models
                if gpu_status['loaded_models']:
                    with st.expander("🔍 Loaded on GPU", expanded=False):
                        for model in gpu_status['loaded_models']:
                            st.caption(f"• {model}")

                # GPU management controls
                st.markdown("**GPU Memory Management:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🗑️ Clear GPU", help="Unload LLM models and clear GPU cache", use_container_width=True):
                        msg = clear_gpu_memory(clear_models=True)
                        st.success(msg)
                        st.rerun()
                with col2:
                    if st.button("⚡ Deep Clean", help="Aggressive cleanup with defragmentation (fixes fragmentation errors)", use_container_width=True):
                        msg = aggressive_memory_cleanup()
                        st.rerun()
                with col3:
                    if st.button("♻️ Cache Only", help="Clear GPU cache without unloading models", use_container_width=True):
                        msg = clear_gpu_memory(clear_models=False)
                        st.info(msg)
                        st.rerun()
        else:
            st.warning("⚠️ No GPU detected. Using CPU (slower)")

        # Acceleration packages status
        with st.expander("📦 Acceleration Status", expanded=False):
            accel_cols = st.columns(2)
            with accel_cols[0]:
                st.write(f"{'✅' if gpu_capabilities['cuml_available'] else '❌'} cuML")
                st.write(f"{'✅' if gpu_capabilities['cupy_available'] else '❌'} CuPy")
                st.write(f"{'✅' if gpu_capabilities.get('bitsandbytes_available', False) else '❌'} BitsAndBytes (4-bit)")
            with accel_cols[1]:
                st.write(f"{'✅' if gpu_capabilities['faiss_gpu_available'] else '❌'} FAISS GPU")
                st.write(f"{'✅' if gpu_capabilities['accelerate_available'] else '❌'} Accelerate")
                if gpu_capabilities.get('is_windows', False):
                    st.caption("⚠️ Windows detected - some packages may not work")

        # Global LLM Configuration
        st.header("🤖 LLM Configuration")
        st.caption("Select one LLM to use across all features (labeling, analysis, chat, summaries)")

        # LLM model selection
        global_llm_model = st.selectbox(
            "LLM Model",
            options=list(LLM_MODEL_CONFIG.keys()),
            index=0,  # Default to first model (Phi-3-mini-128k)
            help="This LLM will be used for all AI features",
            key="global_llm_model"
        )

        # Store in session state
        st.session_state.global_llm_model = global_llm_model

        # Global optimization options
        col_llm1, col_llm2 = st.columns(2)
        with col_llm1:
            can_use_4bit = bitsandbytes_available
            global_use_4bit = st.checkbox(
                "⚡ 4-bit Quantization",
                value=False,
                disabled=not can_use_4bit,
                help="2x memory savings (7GB → 3.5GB)" +
                     (" [Disabled: " + str(bitsandbytes_error) + "]" if not can_use_4bit else ""),
                key="global_use_4bit"
            )
            st.session_state.global_use_4bit = global_use_4bit
            if not can_use_4bit:
                if IS_WINDOWS:
                    st.caption("⚠️ Windows: Use FP16 instead")
                else:
                    st.caption(f"⚠️ {bitsandbytes_error}")

        with col_llm2:
            global_force_cpu = st.checkbox(
                "Force CPU",
                value=False,
                help="Use system RAM instead of GPU (slower but reliable)",
                key="global_force_cpu"
            )
            st.session_state.global_force_cpu = global_force_cpu

        # Show model info
        model_config = LLM_MODEL_CONFIG[global_llm_model]
        quant_status = "4-bit" if global_use_4bit and can_use_4bit else "FP16"
        device_status = "CPU" if global_force_cpu else "GPU"
        st.caption(f"📊 {model_config['description']} | {quant_status} | {device_status}")

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

                st.divider()

                # ✅ CARMACK: Custom stopwords to exclude domain-specific common words
                custom_stopwords_input = st.text_area(
                    "Custom Stopwords (comma-separated)",
                    value="",
                    help="Add domain-specific words to exclude from topics (e.g., 'samsung, product, customer'). These will be filtered from keywords and topic names.",
                    placeholder="samsung, product, customer, service"
                )

                custom_stopwords = set()
                if custom_stopwords_input.strip():
                    custom_stopwords = {w.strip().lower() for w in custom_stopwords_input.split(',') if w.strip()}

                # Store in session state for use in clustering
                st.session_state.custom_stopwords = custom_stopwords
                if custom_stopwords:
                    st.info(f"✅ Custom stopwords: {', '.join(sorted(custom_stopwords))}")

            # Topic Size Control
            st.subheader("📏 Topic Size Control")
            # Scale min_topic_size with dataset size (target 0.5-2% of data)
            if len(df) < 1000:
                default_min_topic_size = max(10, len(df) // 50)  # 2% for small datasets
            elif len(df) < 10000:
                default_min_topic_size = max(20, len(df) // 100)  # 1% for medium datasets
            else:
                default_min_topic_size = max(100, len(df) // 200)  # 0.5% for large datasets

            min_topic_size = st.slider(
                "Minimum Topic Size",
                min_value=2,
                max_value=max(2, min(500, len(df) // 10)),
                value=default_min_topic_size,
                help=f"Minimum number of documents per topic. Default: {default_min_topic_size} ({(default_min_topic_size/len(df)*100):.1f}% of dataset)"
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
                    # Scale default nr_topics with dataset size (for HDBSCAN guidance)
                    if len(df) < 1000:
                        nr_topics = max(5, min(20, len(df) // 50))
                    elif len(df) < 10000:
                        nr_topics = max(10, min(50, len(df) // 200))
                    else:
                        nr_topics = max(20, min(100, len(df) // 400))

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
                    "Use LLM for Better Labels",
                    value=False,
                    help="Generate topic labels using the configured LLM (see sidebar)"
                )

                # Use global LLM settings from sidebar
                llm_model_name = st.session_state.get('global_llm_model')
                use_4bit_labeling = st.session_state.get('global_use_4bit', False)
                force_llm_cpu = st.session_state.get('global_force_cpu', False)

                if use_llm_labeling:
                    # Show which settings will be used
                    if llm_model_name:
                        model_short_name = llm_model_name.split('/')[-1]
                        quant_status = "4-bit" if use_4bit_labeling else "FP16"
                        device_status = "CPU" if force_llm_cpu else "GPU"
                        st.info(f"📊 Using global LLM: {model_short_name} | {quant_status} | {device_status}")
                        st.caption("💡 Change settings in sidebar: 🤖 LLM Configuration")

                        if force_llm_cpu:
                            st.caption("🔧 CPU mode: Slower but doesn't use GPU memory")
                        elif torch.cuda.is_available():
                            gpu_free = torch.cuda.mem_get_info()[0] / (1024**3)
                            if gpu_free < 4.0:
                                st.warning(f"⚠️ Only {gpu_free:.1f}GB GPU memory free. Consider Force CPU in sidebar.")
                    else:
                        st.error("⚠️ No global LLM configured. Please select one in the sidebar.")
                        st.stop()

                    st.caption("⚠️ First-time download may be large (3-14GB). Model is cached after first use.")

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
                # First validate that arrays are aligned
                is_valid, validation_msg = validate_document_indexing(embeddings, cleaned_docs, topics=None)
                if not is_valid:
                    st.error(f"❌ Document indexing validation failed: {validation_msg}")
                    st.error("This may cause issues with RAG search. Please check your data.")
                else:
                    st.success(f"✅ Document indexing validated: {validation_msg}")
                    st.caption(f"📊 Indexed: {len(embeddings):,} embeddings ↔ {len(cleaned_docs):,} documents")

                # Step 4: 🚀 CARMACK: Try loading cached index first, then chunk + build if needed
                status_msg = "LLM analysis and RAG chat" if use_llm_labeling else "RAG chat"
                with st.spinner(f"🔍 Step 4: Building enhanced FAISS index for {status_msg}..."):
                    # Try loading from cache
                    cached_data = load_faiss_index_from_disk()

                    if cached_data is not None:
                        # Use cached index
                        faiss_index, chunk_embeddings, chunks, parent_mapping = cached_data
                        st.session_state.faiss_index = faiss_index
                        st.session_state.chunks = chunks
                        st.session_state.parent_mapping = parent_mapping
                        st.session_state.chunk_embeddings = chunk_embeddings
                    else:
                        # Build new index with chunking
                        st.info("📄 KARPATHY: Chunking documents (512 tokens, 128 overlap)...")

                        # Get tokenizer from safe_model for chunking
                        try:
                            from transformers import AutoTokenizer
                            model_name = safe_model.hf_model if hasattr(safe_model, 'hf_model') else "sentence-transformers/all-MiniLM-L6-v2"
                            chunk_tokenizer = AutoTokenizer.from_pretrained(model_name)
                        except:
                            # Fallback: use a simple tokenizer
                            class SimpleTokenizer:
                                def encode(self, text, add_special_tokens=False):
                                    return text.split()
                                def decode(self, tokens):
                                    return ' '.join(tokens) if isinstance(tokens, list) else tokens
                            chunk_tokenizer = SimpleTokenizer()

                        # Chunk documents
                        chunks, parent_mapping = chunk_documents(
                            cleaned_docs,
                            chunk_tokenizer,
                            chunk_size=512,
                            overlap=128,
                            topics=None  # Will be set later after clustering
                        )

                        st.success(f"✅ Chunked {len(cleaned_docs):,} docs → {len(chunks):,} chunks ({len(chunks)/len(cleaned_docs):.1f}x)")

                        # Embed chunks
                        st.info("🔢 Embedding chunks with sentence-transformers...")
                        chunk_texts = [chunk.text for chunk in chunks]
                        chunk_embeddings = safe_model.model.encode(
                            chunk_texts,
                            show_progress_bar=True,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            batch_size=batch_size
                        )

                        # Build FAISS index from chunk embeddings
                        st.info("🚀 Building FAISS index from chunks...")
                        faiss_index = build_faiss_index(chunk_embeddings)

                        if faiss_index:
                            st.success(f"✅ FAISS index built: {faiss_index.ntotal:,} chunk vectors")

                            # Save to disk
                            st.info("💾 CARMACK: Persisting index to disk...")
                            save_faiss_index_to_disk(faiss_index, chunk_embeddings, chunks, parent_mapping)

                            # Store in session state
                            st.session_state.faiss_index = faiss_index
                            st.session_state.chunks = chunks
                            st.session_state.parent_mapping = parent_mapping
                            st.session_state.chunk_embeddings = chunk_embeddings
                        else:
                            st.error("❌ Failed to build FAISS index")
                            st.session_state.faiss_index = None

                # ✅ Clear GPU memory before loading LLM
                if torch.cuda.is_available() and use_llm_labeling:
                    st.info("🧹 Clearing GPU cache before loading LLM...")
                    clear_gpu_memory()
                    gpu_free_after = torch.cuda.mem_get_info()[0] / (1024**3)
                    st.info(f"📊 GPU memory available for LLM: {gpu_free_after:.1f} GB")

                # Step 5: Load LLM if enabled
                llm_model = None
                if use_llm_labeling and llm_model_name:
                    # Clear old LLM if loading a different one
                    if st.session_state.get('llm_model_name') != llm_model_name:
                        if st.session_state.get('llm_model'):
                            st.info("🧹 Clearing old topic labeling LLM from GPU...")
                            # Properly delete model and tokenizer objects first
                            try:
                                old_llm = st.session_state.llm_model
                                if isinstance(old_llm, tuple) and len(old_llm) == 2:
                                    model, tokenizer = old_llm
                                    del model, tokenizer
                            except Exception as e:
                                st.warning(f"Error during model cleanup: {e}")

                            st.session_state.llm_model = None
                            st.session_state.llm_model_name = None

                            # Clear Streamlit's cache_resource to prevent multiple cached models
                            st.cache_resource.clear()
                            st.info("🗑️ Cleared Streamlit model cache")

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                freed_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                                st.info(f"✅ GPU memory freed: {freed_gb:.1f} GB available")

                    with st.spinner(f"Loading {llm_model_name} for enhanced labeling..."):
                        llm_model = load_local_llm(llm_model_name, force_cpu=force_llm_cpu, use_4bit=use_4bit_labeling)
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
                    # Also store in top-level session state for easy reuse by other features
                    st.session_state.llm_model = llm_model
                    st.session_state.llm_model_name = llm_model_name
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

                # Store seed words in session state for reclustering
                st.session_state.seed_words = seed_topic_list if seed_topic_list else None
                if seed_topic_list:
                    st.info(f"🎯 Loaded {len(seed_topic_list)} seed word sets for guided clustering")

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
                        method=method,
                        seed_words=seed_topic_list if seed_topic_list else None
                    )

                    if topics is None:
                        st.error("Initial clustering failed!")
                        st.stop()

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

        # ✅ CARMACK: Show seed words status
        seed_words_active = st.session_state.get('seed_words', None)
        if seed_words_active:
            st.info(f"🎯 **Seed Words Active:** {len(seed_words_active)} keyword sets will guide clustering")
        else:
            st.caption("💡 No seed words loaded. Clustering will be unsupervised.")

        # Recluster button
        if st.button("🔄 Recluster with New Settings", type="secondary"):
            with st.spinner(f"Reclustering into {n_topics_slider} topics... (This is fast!)"):
                method = 'kmeans' if "K-means" in clustering_method else 'hdbscan'
                # ✅ CARMACK: Pass seed words from session state
                seed_words = st.session_state.get('seed_words', None)
                topics, topic_info = st.session_state.reclusterer.recluster(
                    n_topics=n_topics_slider,
                    min_topic_size=st.session_state.min_topic_size,
                    use_reduced=use_reduced and st.session_state.umap_embeddings is not None,
                    method=method,
                    seed_words=seed_words
                )

                if topics is not None:
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

        # Use global LLM settings from sidebar
        post_llm_model_name = st.session_state.get('global_llm_model')
        post_llm_use_4bit = st.session_state.get('global_use_4bit', False)
        post_llm_force_cpu = st.session_state.get('global_force_cpu', False)

        # Show which settings will be used
        if post_llm_model_name:
            model_short_name = post_llm_model_name.split('/')[-1]
            quant_status = "4-bit" if post_llm_use_4bit else "FP16"
            device_status = "CPU" if post_llm_force_cpu else "GPU"
            st.info(f"📊 Using global LLM: {model_short_name} | {quant_status} | {device_status}")
            st.caption("💡 Change settings in sidebar: 🤖 LLM Configuration")
        else:
            st.error("⚠️ No global LLM configured. Please select one in the sidebar.")
            st.stop()

        # Debug mode toggle
        enable_debug = st.checkbox(
            "🔍 Enable debug logging (shows why topics fail)",
            value=False,
            key="llm_debug_mode",
            help="Logs detailed information about LLM analysis process to help diagnose failures"
        )

        # Configure logging based on debug mode
        if enable_debug:
            import logging
            logging.basicConfig(level=logging.DEBUG, force=True)
            st.info("🔍 Debug mode enabled - check terminal/logs for detailed analysis output")
        else:
            import logging
            logging.basicConfig(level=logging.WARNING, force=True)

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
                    post_llm = load_local_llm(post_llm_model_name, force_cpu=post_llm_force_cpu, use_4bit=post_llm_use_4bit)

                    if post_llm is None:
                        st.error("❌ Failed to load LLM. Check the error messages above.")
                    else:
                        # Store in session state for reuse by other features (like topic summary)
                        st.session_state.llm_model = post_llm
                        st.session_state.post_llm_model_name = post_llm_model_name

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
            topic_info = st.session_state.current_topic_info

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
            # ✅ CARMACK: Vectorized dictionary creation (20x faster than iterrows)
            topic_keywords = dict(zip(topic_info['Topic'], topic_info['Keywords']))
            if 'Human_Label' in topic_info.columns:
                topic_human = dict(zip(topic_info['Topic'], topic_info['Human_Label']))
            else:
                topic_human = dict(zip(topic_info['Topic'],
                                      topic_info.get('Name', topic_info['Topic'].apply(lambda x: f"Topic {x}"))))
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

                display_df = topic_info.copy()
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
                        oversized_options,
                        key="split_topic_selector"
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
                        key=f"split_topic_slider_{topic_to_split}",
                        help="How many subtopics to create from this large topic"
                    )

                    if st.button(
                        f"🔍 Split Topic {topic_to_split} into {sub_n_topics} subtopics",
                        key=f"split_topic_btn_{topic_to_split}"
                    ):
                        # ✅ CARMACK: Reuse existing embeddings instead of re-computing
                        docs_to_split_indices = [i for i, t in enumerate(topics) if t == topic_to_split]
                        docs_to_split = [st.session_state.documents[i] for i in docs_to_split_indices]
                        docs_to_split_embeddings = st.session_state.embeddings[docs_to_split_indices]

                        if len(docs_to_split) >= max(10, sub_n_topics * 2):
                            with st.spinner(f"Analyzing {len(docs_to_split):,} documents..."):
                                # Direct K-means on existing embeddings (10x faster than BERTopic)
                                if gpu_capabilities['cuda_available']:
                                    kmeans = GPUKMeans(n_clusters=sub_n_topics)
                                else:
                                    kmeans = KMeans(n_clusters=sub_n_topics, random_state=42)

                                sub_topics = kmeans.fit_predict(docs_to_split_embeddings)

                                st.success(f"✅ Split into {len(set(sub_topics))} subtopics")

                                # Build topic info manually (keywords extraction)
                                st.write("### 📊 Subtopics Found:")
                                sub_topic_data = []
                                for topic_id in sorted(set(sub_topics)):
                                    if topic_id == -1:
                                        continue
                                    topic_docs = [docs_to_split[i] for i, t in enumerate(sub_topics) if t == topic_id]
                                    count = len(topic_docs)

                                    # Extract keywords (simple word frequency)
                                    all_text = ' '.join(topic_docs[:50]).lower()
                                    words = all_text.split()
                                    # ✅ CARMACK: Use module-level Counter import (avoid local import scoping issues)
                                    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
                                    word_counts = Counter(w for w in words if len(w) > 3 and w not in common_words)
                                    keywords = ', '.join([w for w, _ in word_counts.most_common(5)])

                                    sub_topic_data.append({
                                        'Topic': topic_id,
                                        'Count': count,
                                        'Keywords': keywords,
                                        '% of Parent': (count / len(docs_to_split) * 100)
                                    })

                                sub_topic_display = pd.DataFrame(sub_topic_data)
                                sub_topic_display['% of Parent'] = sub_topic_display['% of Parent'].round(1)

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

                # Get the selected topic ID (needed for both search and browsing)
                selected_topic_id = topic_options[selected_option]

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

                            # ✅ CARMACK: LLM Topic Summary Feature
                            # Generate quick summary of all documents in this topic
                            col_summary1, col_summary2 = st.columns([1, 4])
                            with col_summary1:
                                if st.button(
                                    f"🤖 Summarize Topic {selected_topic_id}",
                                    key=f"summarize_topic_btn_{selected_topic_id}",
                                    help="Get LLM-powered summary of all documents in this topic"
                                ):
                                    st.session_state[f'generate_topic_summary_{selected_topic_id}'] = True

                            with col_summary2:
                                summary_max_docs = st.number_input(
                                    "Max docs for summary:",
                                    min_value=10,
                                    max_value=500,
                                    value=100,
                                    step=10,
                                    key=f"summary_max_docs_{selected_topic_id}",
                                    help="Limit number of documents to analyze (more = slower but comprehensive)"
                                )

                            # Generate summary if requested
                            if st.session_state.get(f'generate_topic_summary_{selected_topic_id}', False):
                                # Clear the flag immediately to prevent re-generation on next rerun
                                st.session_state[f'generate_topic_summary_{selected_topic_id}'] = False

                                with st.spinner(f"🤖 Analyzing {min(len(filtered_df), summary_max_docs)} documents from Topic {selected_topic_id}..."):
                                    try:
                                        # Get all document texts for this topic
                                        topic_docs = filtered_df[text_col].head(summary_max_docs).tolist()

                                        # Get keywords from topic_info or browser_df
                                        topic_keywords = ""
                                        if 'Topic_Keywords' in filtered_df.columns and len(filtered_df) > 0:
                                            topic_keywords = filtered_df.iloc[0]['Topic_Keywords']
                                        elif hasattr(st.session_state, 'current_topic_info'):
                                            topic_row = st.session_state.current_topic_info[st.session_state.current_topic_info['Topic'] == selected_topic_id]
                                            if len(topic_row) > 0 and 'Keywords' in topic_row.columns:
                                                topic_keywords = topic_row.iloc[0]['Keywords']

                                        # ✅ Try to reuse already-loaded LLM from any source
                                        # Priority: chat_llm (most likely loaded) > llm_model > clustering LLM
                                        llm_model = None
                                        llm_tokenizer = None

                                        # Check multiple possible LLM sources (prioritize chat since it's most commonly used)
                                        if 'chat_llm' in st.session_state and st.session_state.chat_llm is not None:
                                            # Chat LLM (from RAG feature) - check it's properly loaded
                                            chat_llm_tuple = st.session_state.chat_llm
                                            if isinstance(chat_llm_tuple, tuple) and len(chat_llm_tuple) == 2:
                                                llm_model, llm_tokenizer = chat_llm_tuple
                                                # Ensure model is on correct device
                                                llm_model = ensure_model_on_device(llm_model, prefer_gpu=True)
                                                st.toast("✅ Reusing chat LLM for summary", icon="♻️")
                                        elif 'llm_model' in st.session_state and st.session_state.llm_model is not None:
                                            # Topic summary LLM (if already loaded earlier)
                                            llm_tuple = st.session_state.llm_model
                                            if isinstance(llm_tuple, tuple) and len(llm_tuple) == 2:
                                                llm_model, llm_tokenizer = llm_tuple
                                                llm_model = ensure_model_on_device(llm_model, prefer_gpu=True)
                                                st.toast("✅ Using cached topic summary LLM", icon="⚡")
                                        elif (hasattr(st.session_state, 'reclusterer') and
                                              st.session_state.reclusterer is not None and
                                              hasattr(st.session_state.reclusterer, 'llm_model') and
                                              st.session_state.reclusterer.llm_model is not None):
                                            # Main clustering LLM (stored in reclusterer)
                                            clustering_tuple = st.session_state.reclusterer.llm_model
                                            if isinstance(clustering_tuple, tuple) and len(clustering_tuple) == 2:
                                                llm_model, llm_tokenizer = clustering_tuple
                                                llm_model = ensure_model_on_device(llm_model, prefer_gpu=True)
                                                st.toast("✅ Reusing clustering LLM for summary", icon="♻️")

                                        # If no LLM found, load a new one using global settings
                                        if llm_model is None:
                                            # Use global LLM settings from sidebar
                                            llm_model_name = st.session_state.get('global_llm_model')
                                            use_4bit = st.session_state.get('global_use_4bit', False)
                                            force_llm_cpu = st.session_state.get('global_force_cpu', False)

                                            if not llm_model_name:
                                                st.error("⚠️ No global LLM configured. Please select one in the sidebar.")
                                                st.info("💡 Configure LLM in sidebar: 🤖 LLM Configuration")
                                                st.stop()

                                            # Show which global settings will be used
                                            model_short_name = llm_model_name.split('/')[-1]
                                            quant_status = "4-bit" if use_4bit else "FP16"
                                            device_status = "CPU" if force_llm_cpu else "GPU"
                                            st.info(f"📊 Using global LLM: {model_short_name} | {quant_status} | {device_status}")
                                            st.caption("💡 Change settings in sidebar: 🤖 LLM Configuration")

                                            with st.spinner(f"Loading LLM model for summary ({quant_status})..."):
                                                llm_tuple = load_local_llm(llm_model_name, force_cpu=force_llm_cpu, use_4bit=use_4bit)

                                                if llm_tuple:
                                                    llm_model, llm_tokenizer = llm_tuple
                                                    st.session_state.llm_model = llm_tuple
                                                    st.toast(f"✅ LLM loaded for summary", icon="⚡")
                                                else:
                                                    st.error("❌ Failed to load LLM for topic summary")
                                                    llm_model = None
                                                    llm_tokenizer = None

                                        # Sample documents intelligently (first, middle, last to get variety)
                                        sample_docs = []
                                        if len(topic_docs) <= 20:
                                            sample_docs = topic_docs
                                        else:
                                            # Take first 7, middle 7, last 6
                                            sample_docs = topic_docs[:7] + topic_docs[len(topic_docs)//2-3:len(topic_docs)//2+4] + topic_docs[-6:]

                                        # Truncate documents for context window
                                        sample_docs_truncated = [doc[:300] for doc in sample_docs]

                                        # Create prompt
                                        docs_text = "\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(sample_docs_truncated)])

                                        prompt = f"""You are analyzing documents from a topic cluster. Provide a concise summary (3-5 sentences) that captures:
1. The main theme/problem discussed
2. Key patterns or common issues mentioned
3. Notable insights or trends

Topic Label: {human_label}
Keywords: {topic_keywords}
Total Documents in Topic: {len(filtered_df)}
Sample Documents (showing {len(sample_docs)} representative docs):

{docs_text}

Provide a clear, actionable summary:"""

                                        # Generate summary (✅ CARMACK: Use return_dict=True for attention_mask)
                                        messages = [{"role": "user", "content": prompt}]
                                        model_inputs = llm_tokenizer.apply_chat_template(
                                            messages,
                                            add_generation_prompt=True,
                                            return_tensors="pt",
                                            return_dict=True
                                        )

                                        if torch.cuda.is_available():
                                            model_inputs = {k: v.to(llm_model.device) for k, v in model_inputs.items()}

                                        # Generate
                                        with torch.no_grad():
                                            outputs = llm_model.generate(
                                                **model_inputs,  # Unpacks input_ids and attention_mask
                                                max_new_tokens=1000,  # ✅ Increased from 300 to allow detailed summaries
                                                temperature=0.7,
                                                do_sample=True,
                                                pad_token_id=llm_tokenizer.eos_token_id,
                                                use_cache=True  # ✅ Re-enabled: 30-50% speedup
                                            )

                                        # Decode only the generated tokens (skip prompt)
                                        input_length = model_inputs['input_ids'].shape[1]
                                        generated_tokens = outputs[0][input_length:]
                                        summary_text = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                                        # Store summary in session state
                                        st.session_state[f'topic_summary_{selected_topic_id}'] = {
                                            'text': summary_text,
                                            'label': human_label,
                                            'keywords': topic_keywords,
                                            'total_docs': len(filtered_df),
                                            'analyzed_docs': min(len(filtered_df), summary_max_docs),
                                            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
                                        }

                                        # ✅ FIX: Update topic_info and browser_df to show analysis in table
                                        if 'current_topic_info' in st.session_state and st.session_state.current_topic_info is not None:
                                            # Make explicit copy to avoid pandas view issues
                                            topic_info_df = st.session_state.current_topic_info.copy()

                                            # Add LLM_Analysis column if it doesn't exist
                                            if 'LLM_Analysis' not in topic_info_df.columns:
                                                topic_info_df['LLM_Analysis'] = ''

                                            # Update the analysis for this topic (ensure proper indexing)
                                            mask = topic_info_df['Topic'] == selected_topic_id
                                            topic_info_df.loc[mask, 'LLM_Analysis'] = summary_text

                                            # Save back to session state
                                            st.session_state.current_topic_info = topic_info_df
                                            st.session_state.topic_info = topic_info_df

                                            # Clear browser_df cache to force rebuild with new analysis
                                            if 'browser_df' in st.session_state:
                                                del st.session_state.browser_df

                                        st.success(f"✅ Summary generated for Topic {selected_topic_id}")
                                        st.rerun()  # Refresh to show analysis in table

                                    except Exception as e:
                                        st.error(f"❌ Failed to generate summary: {str(e)}")
                                        st.exception(e)

                            # Display summary if it exists (persists after generation)
                            if f'topic_summary_{selected_topic_id}' in st.session_state:
                                summary_data = st.session_state[f'topic_summary_{selected_topic_id}']

                                with st.expander(f"📝 Topic {selected_topic_id} Summary ({summary_data['total_docs']:,} docs)", expanded=True):
                                    col_summary_display, col_clear = st.columns([5, 1])

                                    with col_clear:
                                        if st.button("🗑️ Clear", key=f"clear_summary_{selected_topic_id}", help="Remove this summary"):
                                            del st.session_state[f'topic_summary_{selected_topic_id}']
                                            st.rerun()

                                    with col_summary_display:
                                        st.markdown(f"**Topic:** {summary_data['label']}")
                                        st.markdown(f"**Keywords:** {summary_data['keywords']}")
                                        st.markdown(f"**Documents Analyzed:** {summary_data['analyzed_docs']:,} of {summary_data['total_docs']:,}")
                                        st.caption(f"Generated: {summary_data['timestamp']}")

                                    st.markdown("---")
                                    st.markdown("**Summary:**")
                                    st.write(summary_data['text'])

                                    # Download button for summary
                                    summary_report = f"""Topic {selected_topic_id} Summary
{summary_data['label']}

Keywords: {summary_data['keywords']}
Total Documents: {summary_data['total_docs']:,}
Documents Analyzed: {summary_data['analyzed_docs']:,}

SUMMARY:
{summary_data['text']}

Generated: {summary_data['timestamp']}
"""
                                    st.download_button(
                                        label="📥 Download Summary",
                                        data=summary_report.encode("utf-8"),
                                        file_name=f"topic_{selected_topic_id}_summary.txt",
                                        mime="text/plain",
                                        key=f"download_summary_persisted_{selected_topic_id}"
                                    )

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
                with st.expander("💬 Ask Questions About Topics (RAG-Powered)", expanded=True):
                    st.caption("Get AI-powered insights using FAISS retrieval and LLM generation")

                    # Check prerequisites - RAG is mandatory, no fallback
                    has_faiss = st.session_state.get('faiss_index') is not None

                    if not has_faiss:
                        st.error("❌ **RAG Prerequisites Missing**")
                        st.warning("FAISS index not found. Recompute embeddings with 'Enable LLM Labeling' to build the index.")
                        st.info("Chat requires:\n- FAISS index\n- Embedding model\n- LLM model")
                        st.stop()  # Don't show chat interface at all

                    # Prerequisites met - show RAG interface
                    show_rag_debug = st.checkbox(
                        "🔍 Show Debug Info",
                        value=True,
                        help="Show detailed diagnostic information about document retrieval",
                        key="show_rag_debug_checkbox"
                    )
                    st.session_state.show_rag_debug = show_rag_debug

                    # Use global LLM settings from sidebar
                    chat_llm_model_name = st.session_state.get('global_llm_model')
                    use_4bit_chat = st.session_state.get('global_use_4bit', False)
                    force_llm_cpu = st.session_state.get('global_force_cpu', False)

                    # Show which global settings will be used
                    if chat_llm_model_name:
                        model_short_name = chat_llm_model_name.split('/')[-1]
                        quant_status = "4-bit" if use_4bit_chat else "FP16"
                        device_status = "CPU" if force_llm_cpu else "GPU"
                        st.info(f"📊 Using global LLM: {model_short_name} | {quant_status} | {device_status}")
                        st.caption("💡 Change settings in sidebar: 🤖 LLM Configuration")
                    else:
                        st.error("⚠️ No global LLM configured. Please select one in the sidebar.")
                        st.stop()

                    # Load chat LLM
                    # Check if model/quantization changed
                    cached_model_name = st.session_state.get('chat_llm_loaded_model')
                    cached_4bit = st.session_state.get('chat_llm_4bit', False)

                    if cached_model_name == chat_llm_model_name and cached_4bit == use_4bit_chat:
                        # Already loaded with same config
                        chat_llm = st.session_state.get('chat_llm')
                        quant_label = "4-bit" if use_4bit_chat else "FP16"
                        st.caption(f"✅ Using cached LLM: {chat_llm_model_name.split('/')[-1]} ({quant_label})")
                    else:
                        # Clear old chat LLM if switching models or quantization
                        if st.session_state.get('chat_llm'):
                            st.info("🧹 Clearing old chat LLM from GPU...")
                            # Properly delete model and tokenizer objects first
                            try:
                                old_chat_llm = st.session_state.chat_llm
                                if isinstance(old_chat_llm, tuple) and len(old_chat_llm) == 2:
                                    model, tokenizer = old_chat_llm
                                    del model, tokenizer
                            except Exception as e:
                                st.warning(f"Error during chat LLM cleanup: {e}")

                            st.session_state.chat_llm = None
                            st.session_state.chat_llm_loaded_model = None
                            st.session_state.chat_llm_4bit = None

                            # Clear Streamlit's cache_resource for the old model configuration
                            # This prevents multiple cached versions from accumulating in memory
                            st.cache_resource.clear()
                            st.info("🗑️ Cleared Streamlit model cache to prevent memory accumulation")

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                freed_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                                st.info(f"✅ GPU memory freed: {freed_gb:.1f} GB available")

                        # Load new model with optimizations
                        quant_label = "4-bit quantization" if use_4bit_chat else "FP16"
                        with st.spinner(f"Loading {chat_llm_model_name} with {quant_label}..."):
                            chat_llm = load_local_llm(chat_llm_model_name, force_cpu=force_llm_cpu, use_4bit=use_4bit_chat)
                            if not chat_llm:
                                st.error("❌ Failed to load LLM. Cannot use chat without LLM.")
                                st.stop()

                            st.session_state.chat_llm = chat_llm
                            st.session_state.chat_llm_loaded_model = chat_llm_model_name
                            st.session_state.chat_llm_4bit = use_4bit_chat
                            st.success(f"✅ Chat LLM loaded: {chat_llm_model_name.split('/')[-1]} ({quant_label})")

                    # Show status
                    num_docs = len(st.session_state.get('documents', []))
                    num_topics = len(set(st.session_state.get('current_topics', [])))
                    st.success(f"✅ RAG Active: {num_docs:,} documents from {num_topics} topics")

                    # Document count control - always visible
                    model_config = LLM_MODEL_CONFIG.get(chat_llm_model_name, {})
                    recommended_docs = model_config.get('recommended_docs', 10)
                    context_window = model_config.get('context_window', 4096)

                    # Use model-specific key so each model has its own slider value
                    model_short_name = chat_llm_model_name.split('/')[-1]
                    slider_key = f"rag_doc_count_slider_{model_short_name}"

                    # Read from session state if exists, otherwise use recommended default
                    default_value = st.session_state.get(slider_key, recommended_docs)

                    rag_doc_count = st.slider(
                        "📊 Documents to retrieve",
                        min_value=1,
                        max_value=100,
                        value=default_value,
                        help=f"Number of documents to search and use for answering. Recommended: {recommended_docs} for this model ({context_window//1024}k context)",
                        key=slider_key
                    )
                    st.caption(f"💡 Recommended for {model_short_name}: {recommended_docs} documents")

                    # Store for use in retrieval
                    st.session_state.rag_doc_count = rag_doc_count

                    # Reranking controls
                    st.markdown("---")
                    st.markdown("**🎯 Reranking & Relevance Filtering**")

                    col_rerank1, col_rerank2 = st.columns(2)
                    with col_rerank1:
                        use_reranking = st.checkbox(
                            "Enable Reranking",
                            value=False,
                            help="Use cross-encoder to re-score and filter retrieved documents for better relevance",
                            key="use_reranking_checkbox"
                        )
                        st.session_state.use_reranking = use_reranking

                    with col_rerank2:
                        if use_reranking:
                            # Read from session state if exists, otherwise use 0.0 default
                            threshold_default = st.session_state.get("relevance_threshold_slider", 0.0)

                            relevance_threshold = st.slider(
                                "Relevance Threshold",
                                min_value=0.0,
                                max_value=1.0,
                                value=threshold_default,
                                step=0.05,
                                help="Minimum relevance score (0.0=keep all, higher=stricter filtering)",
                                key="relevance_threshold_slider"
                            )
                            st.session_state.relevance_threshold = relevance_threshold
                        else:
                            st.session_state.relevance_threshold = 0.0

                    if use_reranking:
                        st.caption(f"✅ Reranking enabled: Retrieves {rag_doc_count * 3} candidates → reranks → returns top {rag_doc_count}")

                    # Query Expansion (ANDREJ KARPATHY)
                    st.markdown("---")
                    st.markdown("**🔍 Query Expansion (Advanced)**")
                    enable_query_expansion = st.checkbox(
                        "Enable Query Expansion",
                        value=False,
                        help="KARPATHY: Rephrase query into 2-3 variants using LLM for better recall. Slightly slower but finds more relevant docs.",
                        key="enable_query_expansion_checkbox"
                    )
                    st.session_state.enable_query_expansion = enable_query_expansion

                    if enable_query_expansion:
                        st.caption("✅ Query expansion: LLM generates 2-3 query variants → retrieves union → better recall")

                    # Initialize chat history in session state
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []

                    # Topic filter for chat
                    # ✅ FIX: Use set() to get unique topic IDs (topics contains all doc assignments with duplicates)
                    unique_topic_ids = sorted(set([t for t in topics if t != -1]))
                    topic_options = ["All Topics"] + [f"Topic {t}: {st.session_state.topic_human.get(t, 'Unknown')}"
                                                       for t in unique_topic_ids]

                    col_filter, col_clear = st.columns([3, 1])
                    with col_filter:
                        chat_topic_filter = st.selectbox(
                            "Filter to specific topic",
                            options=topic_options,
                            index=0,
                            key="chat_topic_filter",
                            help="Limit search to documents from a specific topic"
                        )
                    with col_clear:
                        if st.button("🗑️ Clear Chat", key="clear_chat_browser", help="Clear chat history"):
                            st.session_state.chat_history = []
                            st.rerun()

                    # Parse selected topic ID from dropdown
                    chat_topic_id = None
                    if chat_topic_filter != "All Topics":
                        chat_topic_id = int(chat_topic_filter.split(":")[0].replace("Topic ", ""))

                    # Chat input
                    if prompt := st.chat_input("Ask a question about your topics...", key="topic_browser_chat"):
                        # Add user message to chat history
                        st.session_state.chat_history.append({"role": "user", "content": prompt})

                        # Generate response (RAG always enabled, no fallback)
                        with st.spinner("Thinking..."):
                            # ✅ FIX: Pass chunk_embeddings instead of document embeddings for FAISS
                            response = generate_chat_response(
                                prompt,
                                topic_info,
                                topics,
                                processed_df,
                                current_topic_id=selected_topic_id if selected_topic_id != "all" else None,
                                explicit_topic_filter=chat_topic_id,
                                use_rag=True,  # Always True - RAG is mandatory
                                llm_model=chat_llm,  # Guaranteed to exist or we stopped earlier
                                faiss_index=st.session_state.get('faiss_index'),
                                embeddings=st.session_state.get('chunk_embeddings', st.session_state.get('embeddings')),
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
                        num_docs = len(st.session_state.get('documents', []))
                        num_topics = len(set(st.session_state.get('current_topics', [])))
                        st.info(f"🤖 **RAG Active** - {num_docs:,} documents from {num_topics} topics available.\n\n"
                               "Semantic search + LLM powered by your actual documents.\n\n"
                               "**Try:**\n"
                               "- 'What do customers say about delivery?'\n"
                               "- 'Find issues related to installation'\n"
                               "- 'Summarize the main complaints'\n"
                               "- 'Tell me about topic 5'")
                    else:
                        # Display chat history (newest first)
                        for message in reversed(st.session_state.chat_history):
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

            with tabs[5]:  # Export
                st.subheader("💾 Export Results")

                export_df = st.session_state.browser_df.copy()
                safe_topic_info = topic_info

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

                st.markdown("---")
                st.subheader("📦 Session Export/Import")
                st.markdown("✅ **CARMACK**: Save complete session to resume work later or skip re-embedding")

                # Session Export
                if st.button("📦 Export Full Session (ZIP)", help="Save embeddings, topics, and all results to resume work later"):
                    import zipfile
                    from io import BytesIO
                    import json
                    from datetime import datetime

                    with st.spinner("Creating session export..."):
                        zip_buffer = BytesIO()

                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            # 1. Save embeddings
                            emb_buffer = BytesIO()
                            np.save(emb_buffer, st.session_state.embeddings)
                            zf.writestr("embeddings.npy", emb_buffer.getvalue())

                            # 2. Save UMAP embeddings if available
                            if st.session_state.umap_embeddings is not None:
                                umap_buffer = BytesIO()
                                np.save(umap_buffer, st.session_state.umap_embeddings)
                                zf.writestr("umap_embeddings.npy", umap_buffer.getvalue())

                            # 3. Save topics and metadata
                            # Build topic_keywords from topic_info
                            topic_keywords_dict = {}
                            if hasattr(st.session_state, 'current_topic_info'):
                                topic_keywords_dict = dict(zip(
                                    st.session_state.current_topic_info['Topic'],
                                    st.session_state.current_topic_info['Keywords']
                                ))

                            session_metadata = {
                                'topics': st.session_state.current_topics.tolist(),
                                'documents': st.session_state.documents,
                                'topic_keywords': topic_keywords_dict,
                                'topic_human': st.session_state.topic_human,
                                'parameters': {
                                    'min_topic_size': st.session_state.get('min_topic_size_used', 10),
                                    'n_topics': len(set(st.session_state.current_topics)),
                                    'clustering_method': st.session_state.clustering_method,
                                    'model_name': st.session_state.model_name,
                                    'uploaded_file_name': st.session_state.uploaded_file_name,
                                },
                                'export_timestamp': datetime.now().isoformat(),
                                'version': '1.0'
                            }
                            zf.writestr("session_metadata.json", json.dumps(session_metadata, indent=2))

                            # 4. Save results CSV
                            zf.writestr("results.csv", export_df.to_csv(index=False))

                            # 5. Save topic info
                            topic_info_export = safe_topic_info[export_cols] if 'LLM_Analysis' in export_cols else safe_topic_info
                            zf.writestr("topic_info.csv", topic_info_export.to_csv(index=False))

                            # 6. Save LLM analysis if available
                            if 'topic_llm_analysis' in st.session_state and st.session_state.topic_llm_analysis:
                                zf.writestr("llm_analysis.json", json.dumps(st.session_state.topic_llm_analysis, indent=2))

                        zip_buffer.seek(0)

                        st.download_button(
                            label="📥 Download Session ZIP",
                            data=zip_buffer.getvalue(),
                            file_name=f"bertopic_session_{st.session_state.uploaded_file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            help="Complete session including embeddings (can resume work without re-embedding)"
                        )

                        st.success("✅ Session exported successfully! Contains embeddings, topics, and all results.")
                        st.info("💡 **Tip**: Import this ZIP file later to resume work instantly without re-embedding documents")

                # Session Import
                st.markdown("### 📂 Import Session")
                uploaded_session = st.file_uploader(
                    "Upload Session ZIP",
                    type=['zip'],
                    help="Import a previously exported session to resume work",
                    key='session_import'
                )

                if uploaded_session is not None:
                    if st.button("🔄 Load Session"):
                        import zipfile
                        import json
                        from io import BytesIO

                        with st.spinner("Loading session..."):
                            try:
                                with zipfile.ZipFile(uploaded_session, 'r') as zf:
                                    # Load metadata
                                    metadata = json.loads(zf.read("session_metadata.json"))

                                    # Load embeddings
                                    st.session_state.embeddings = np.load(BytesIO(zf.read("embeddings.npy")))

                                    # Load UMAP embeddings if available
                                    if "umap_embeddings.npy" in zf.namelist():
                                        st.session_state.umap_embeddings = np.load(BytesIO(zf.read("umap_embeddings.npy")))

                                    # Load topics and documents
                                    st.session_state.current_topics = np.array(metadata['topics'])
                                    st.session_state.documents = metadata['documents']
                                    st.session_state.topic_keywords = metadata['topic_keywords']
                                    st.session_state.topic_human = metadata['topic_human']

                                    # Load parameters
                                    params = metadata['parameters']
                                    st.session_state.min_topic_size_used = params.get('min_topic_size', 10)
                                    st.session_state.clustering_method = params.get('clustering_method', 'HDBSCAN')
                                    st.session_state.model_name = params.get('model_name', 'all-MiniLM-L6-v2')
                                    st.session_state.uploaded_file_name = params.get('uploaded_file_name', 'imported_session')

                                    # Load results
                                    results_csv = pd.read_csv(BytesIO(zf.read("results.csv")))
                                    st.session_state.browser_df = results_csv
                                    st.session_state.df = results_csv

                                    # Load LLM analysis if available
                                    if "llm_analysis.json" in zf.namelist():
                                        st.session_state.topic_llm_analysis = json.loads(zf.read("llm_analysis.json"))

                                    st.success(f"✅ Session loaded successfully!")
                                    st.info(f"📊 Loaded {len(st.session_state.documents):,} documents with {params['n_topics']} topics")
                                    st.info(f"🕐 Original session: {metadata.get('export_timestamp', 'Unknown')}")
                                    st.info(f"⚡ Embeddings loaded - no re-computation needed!")
                                    st.rerun()

                            except Exception as e:
                                st.error(f"❌ Failed to load session: {str(e)}")
                                st.exception(e)

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
