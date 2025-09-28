#!/usr/bin/env python3
"""
Test script to verify sentiment-based topic separation
Creates sample positive and negative customer service feedback
"""

import pandas as pd
import streamlit as st

# Sample data with clear positive and negative customer service feedback
test_data = {
    'feedback': [
        # Positive customer service
        "Excellent customer service! The support team was incredibly helpful and resolved my issue quickly.",
        "Outstanding support experience. The representative was knowledgeable and patient.",
        "Great customer service, very responsive and professional staff.",
        "The customer support team went above and beyond to help me. Truly excellent service!",
        "Amazing customer service experience. Quick response and perfect solution.",
        "Wonderful support team! They were so helpful and friendly throughout.",
        "Best customer service I've ever experienced. Fast and efficient help.",
        "Superb customer support. The agent was extremely competent and courteous.",
        "Fantastic service from the support team. Problem solved in minutes!",
        "Exceptional customer care. Very impressed with the level of service.",

        # Negative customer service
        "Poor customer service. Nobody could help me with my issue.",
        "Terrible support experience. Waited hours and got no resolution.",
        "Awful customer service. The representative was rude and unhelpful.",
        "Worst customer support ever. Complete waste of time.",
        "Very disappointed with the customer service. No one seemed to care.",
        "Horrible experience with support. Still waiting for a solution.",
        "Bad customer service. The agent didn't understand my problem at all.",
        "Frustrating support experience. Multiple transfers with no help.",
        "Unacceptable customer service. Issue still unresolved after days.",
        "Pathetic support team. Completely incompetent and dismissive.",

        # Positive product quality
        "The product quality is excellent. Very well made and durable.",
        "Outstanding product! Exceeds all my expectations for quality.",
        "Great quality item. Worth every penny I paid.",
        "Superior product quality. Built to last and works perfectly.",
        "Amazing quality! This product is exactly what I needed.",

        # Negative product quality
        "Product quality is terrible. Broke after just one use.",
        "Poor quality item. Not worth the money at all.",
        "Bad product quality. Materials feel cheap and flimsy.",
        "Awful quality. The product fell apart immediately.",
        "Very poor quality. Disappointed with this purchase.",

        # Positive delivery
        "Fast delivery! Arrived earlier than expected.",
        "Excellent shipping. Well packaged and on time.",
        "Great delivery service. Quick and reliable.",
        "Perfect delivery experience. Item arrived in perfect condition.",
        "Amazing shipping speed. Got it the next day!",

        # Negative delivery
        "Slow delivery. Took weeks to arrive.",
        "Terrible shipping. Package was damaged.",
        "Poor delivery service. Still waiting after a month.",
        "Bad shipping experience. Wrong item delivered.",
        "Awful delivery. Package was lost in transit.",

        # Mixed/Neutral
        "The product is okay but customer service could be better.",
        "Good product quality but shipping was delayed.",
        "Average experience overall. Nothing special.",
        "Service was fine, nothing to complain about.",
        "Standard experience, met basic expectations."
    ]
}

# Create DataFrame
df = pd.DataFrame(test_data)

# Save to CSV
df.to_csv('test_sentiment_data.csv', index=False)
print(f"Created test_sentiment_data.csv with {len(df)} samples")
print("\nData distribution:")
print("- Positive customer service: 10 samples")
print("- Negative customer service: 10 samples")
print("- Positive product quality: 5 samples")
print("- Negative product quality: 5 samples")
print("- Positive delivery: 5 samples")
print("- Negative delivery: 5 samples")
print("- Mixed/Neutral: 5 samples")
print("\nTotal: 45 samples")
print("\nUse this file to test if positive and negative sentiments are properly separated into different topics.")