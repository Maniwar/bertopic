"""
Carmack's Simple LLM Analysis
No batching. No parsing. Just parallel calls.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st


def analyze_topics_simple(topics_dict, llm_model, max_workers=4):
    """
    Analyze topics with LLM. One topic = one call. Run in parallel.

    Simple. Fast. Reliable. No parsing complexity.

    Args:
        topics_dict: Dict of {topic_id: {'label': str, 'docs': list}}
        llm_model: Tuple of (model, tokenizer)
        max_workers: Parallel workers (default 4)

    Returns:
        Dict of {topic_id: analysis_text}
    """
    if not topics_dict or llm_model is None:
        return {}

    results = {}
    total = len(topics_dict)

    # Progress tracking
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all topics
        future_to_topic = {
            executor.submit(
                _analyze_single_topic,
                topic_id,
                info['label'],
                info['docs'],
                llm_model
            ): topic_id
            for topic_id, info in topics_dict.items()
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_topic):
            topic_id = future_to_topic[future]
            completed += 1

            try:
                analysis = future.result()
                results[topic_id] = analysis if analysis else "No analysis available"
            except Exception as e:
                results[topic_id] = "No analysis available"
                st.caption(f"⚠️ Topic {topic_id} failed: {str(e)[:50]}")

            # Update progress
            progress = completed / total
            progress_bar.progress(progress)
            status_text.info(f"🔄 Analyzed {completed}/{total} topics")

    # Clean up
    progress_bar.empty()
    status_text.empty()

    success_count = sum(1 for v in results.values() if v != "No analysis available")
    st.success(f"✅ Analyzed {success_count}/{total} topics successfully")

    return results


def _analyze_single_topic(topic_id, label, docs, llm_model):
    """
    Analyze one topic. Called in parallel by ThreadPoolExecutor.

    Simple prompt. Simple parsing. No state tracking.
    """
    if not docs or llm_model is None:
        return None

    model, tokenizer = llm_model

    # Clean docs
    cleaned = [str(d).strip() for d in docs[:8] if d and str(d).strip()]
    if not cleaned:
        return None

    # Simple prompt - just ask for analysis
    docs_text = "\n".join([f"- {doc[:200]}" for doc in cleaned[:8]])

    prompt = f"""Topic: {label}

Sample documents:
{docs_text}

Provide a one-sentence analysis of what users are saying in this topic.

Analysis:"""

    # Generate
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Simple extraction - just take everything after "Analysis:"
    if "Analysis:" in response:
        analysis = response.split("Analysis:")[-1].strip()
    else:
        # LLM might just respond directly
        analysis = response.strip()

    # Clean up
    analysis = analysis.split('\n')[0].strip()  # First line only
    analysis = analysis.strip('"\'.,;[](){}')

    # Validate
    if len(analysis) > 10 and len(analysis) < 500:
        return analysis

    return None


# Drop-in replacement for your current code
def analyze_topics_for_reclusterer(topics_dict, labels_dict, llm_model):
    """
    Drop-in replacement for FastReclusterer's LLM analysis.

    Usage:
        # In FastReclusterer._extract_topic_keywords, replace batch code with:
        if self.llm_model is not None:
            topic_data = {
                topic_id: {'label': labels_dict[topic_id], 'docs': docs}
                for topic_id, docs in topics_dict.items()
                if topic_id != -1
            }
            llm_analysis_dict = analyze_topics_simple(topic_data, self.llm_model)
    """
    topic_data = {
        topic_id: {'label': labels_dict.get(topic_id, f"Topic {topic_id}"), 'docs': docs}
        for topic_id, docs in topics_dict.items()
        if topic_id != -1
    }

    return analyze_topics_simple(topic_data, llm_model)
