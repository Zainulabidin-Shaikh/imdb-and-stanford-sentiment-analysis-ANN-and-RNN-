"""
Streamlit Application for Sentiment Analysis - Inference Only
Load pre-trained models and demonstrate predictions (no training in UI)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os
import sys
from pathlib import Path
import pickle
import tensorflow as tf

# Add project modules to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from config import *
from data_loader import DataLoader
from ann_model import ANNSentimentClassifier
from rnn_model import RNNSentimentClassifier
from evaluation import ModelEvaluator

# Configure Streamlit page
st.set_page_config(
    page_title="🎭 Sentiment Analysis Demo",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .model-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .prediction-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .prediction-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class SentimentAnalysisApp:
    """Streamlit app for sentiment analysis inference"""

    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.initialize_session_state()
        self.load_available_models()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = 'imdb'
        if 'ann_model' not in st.session_state:
            st.session_state.ann_model = None
        if 'rnn_model' not in st.session_state:
            st.session_state.rnn_model = None
        if 'tokenizer' not in st.session_state:
            st.session_state.tokenizer = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = None

    def load_available_models(self):
        """Check for available pre-trained models"""
        available_datasets = []

        for dataset_name in DATASETS.keys():
            model_dir = MODELS_DIR / dataset_name
            best_models_file = model_dir / "best_models.json"

            if best_models_file.exists():
                available_datasets.append(dataset_name)

        st.session_state.available_datasets = available_datasets

    def load_models(self, dataset_name):
        """Load pre-trained models for specified dataset"""
        model_dir = MODELS_DIR / dataset_name
        best_models_file = model_dir / "best_models.json"

        if not best_models_file.exists():
            st.error(f"No trained models found for {dataset_name}. Please run training first.")
            st.info("Run: python train_models.py --dataset " + dataset_name)
            return False

        try:
            # Load model info
            with open(best_models_file, 'r') as f:
                model_info = json.load(f)

            st.session_state.model_info = model_info

            # Load tokenizer
            with open(model_info['tokenizer_path'], 'rb') as f:
                tokenizer = pickle.load(f)
            st.session_state.tokenizer = tokenizer

            # Load ANN model
            if os.path.exists(model_info['ann_info']['model_path']):
                ann_model = ANNSentimentClassifier()
                ann_model.load_model(model_info['ann_info']['model_path'])
                st.session_state.ann_model = ann_model

            # Load RNN model
            if os.path.exists(model_info['rnn_info']['model_path']):
                rnn_model = RNNSentimentClassifier()
                rnn_model.load_model(model_info['rnn_info']['model_path'])
                st.session_state.rnn_model = rnn_model

            st.session_state.models_loaded = True
            return True

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False

    def run(self):
        """Main application runner"""

        # Header
        st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Demo</h1>', 
                   unsafe_allow_html=True)

        st.markdown("""
        ### Interactive Demo of Pre-trained ANN vs RNN Models

        This application demonstrates pre-trained sentiment analysis models:
        - 📊 **Model Performance**: Compare trained ANN and RNN models
        - 🔮 **Interactive Prediction**: Test models with your own text
        - 📈 **Performance Metrics**: View comprehensive evaluation results
        - ⚖️ **Model Comparison**: Side-by-side analysis
        """)

        # Sidebar for model selection
        self.create_sidebar()

        # Check if models are available
        if not st.session_state.available_datasets:
            st.error("No trained models found!")
            st.info("""
            To get started:
            1. Run training first: `python train_models.py --dataset imdb`
            2. Wait for training to complete
            3. Reload this page
            """)
            return

        # Main content based on selection
        if not st.session_state.models_loaded:
            st.warning("Please select a dataset and load models using the sidebar.")
            return

        # Main content
        if st.session_state.page == "Model Overview":
            self.model_overview_page()
        elif st.session_state.page == "Interactive Demo":
            self.interactive_demo_page()
        elif st.session_state.page == "Model Comparison":
            self.model_comparison_page()
        elif st.session_state.page == "Performance Analysis":
            self.performance_analysis_page()

    def create_sidebar(self):
        """Create sidebar for model selection"""
        st.sidebar.title("Model Selection")

        # Available datasets
        if st.session_state.available_datasets:
            selected_dataset = st.sidebar.selectbox(
                "Choose Trained Dataset",
                st.session_state.available_datasets,
                format_func=lambda x: DATASETS[x]["name"],
                index=0
            )

            # Load models button
            if st.sidebar.button("Load Models", key="load_models_btn"):
                with st.spinner("Loading pre-trained models..."):
                    success = self.load_models(selected_dataset)
                    if success:
                        st.sidebar.success("Models loaded successfully!")
                        st.session_state.selected_dataset = selected_dataset
                    else:
                        st.sidebar.error("Failed to load models")

        else:
            st.sidebar.error("No trained models available")
            st.sidebar.info("Run training first:\n`python train_models.py`")

        # Navigation
        if st.session_state.models_loaded:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Navigation")

            pages = [
                "Model Overview",
                "Interactive Demo", 
                "Model Comparison",
                "Performance Analysis"
            ]

            st.session_state.page = st.sidebar.selectbox(
                "Select Page",
                pages,
                index=0
            )

        # Model status
        st.sidebar.markdown("---")
        st.sidebar.subheader("Status")

        if st.session_state.models_loaded:
            st.sidebar.success("✅ Models Loaded")
            if st.session_state.model_info:
                best_model = st.session_state.model_info['best_model_type'].upper()
                st.sidebar.info(f"🏆 Best: {best_model}")
        else:
            st.sidebar.warning("⏳ Models Not Loaded")

    def model_overview_page(self):
        """Model overview and information page"""
        st.markdown('<h2 class="section-header">📊 Model Overview</h2>', 
                   unsafe_allow_html=True)

        model_info = st.session_state.model_info

        # Dataset information
        st.subheader(f"Dataset: {DATASETS[st.session_state.selected_dataset]['name']}")

        col1, col2 = st.columns(2)

        # ANN Model Info
        with col1:
            st.subheader("🔹 ANN Model")
            ann_info = model_info['ann_info']
            ann_metrics = ann_info['metrics']

            st.markdown(f"""
            <div class="model-info">
            <h4>Architecture</h4>
            <ul>
                <li>Embedding Dimension: {ann_info['config']['embedding_dim']}</li>
                <li>Hidden Units: {ann_info['config']['hidden_units']}</li>
                <li>Dropout Rate: {ann_info['config']['dropout_rate']}</li>
                <li>Batch Size: {ann_info['config']['batch_size']}</li>
            </ul>

            <h4>Performance</h4>
            <ul>
                <li>Accuracy: {ann_metrics['accuracy']:.4f}</li>
                <li>F1-Score: {ann_metrics['f1_score']:.4f}</li>
                <li>Precision: {ann_metrics['precision']:.4f}</li>
                <li>Recall: {ann_metrics['recall']:.4f}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # RNN Model Info
        with col2:
            st.subheader("🔹 RNN Model")
            rnn_info = model_info['rnn_info']
            rnn_metrics = rnn_info['metrics']

            st.markdown(f"""
            <div class="model-info">
            <h4>Architecture</h4>
            <ul>
                <li>Embedding Dimension: {rnn_info['config']['embedding_dim']}</li>
                <li>LSTM Units: {rnn_info['config']['lstm_units']}</li>
                <li>Bidirectional: {'Yes' if rnn_info['config']['bidirectional'] else 'No'}</li>
                <li>Dropout Rate: {rnn_info['config']['dropout_rate']}</li>
            </ul>

            <h4>Performance</h4>
            <ul>
                <li>Accuracy: {rnn_metrics['accuracy']:.4f}</li>
                <li>F1-Score: {rnn_metrics['f1_score']:.4f}</li>
                <li>Precision: {rnn_metrics['precision']:.4f}</li>
                <li>Recall: {rnn_metrics['recall']:.4f}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # Best model highlight
        best_model = model_info['best_model_type']
        st.markdown("---")
        st.subheader("🏆 Best Performing Model")

        if best_model == 'ann':
            best_metrics = ann_metrics
            model_name = "Artificial Neural Network (ANN)"
        else:
            best_metrics = rnn_metrics
            model_name = "Recurrent Neural Network (RNN)"

        st.success(f"**{model_name}** achieved the highest F1-Score: **{best_metrics['f1_score']:.4f}**")

        # Training timestamp
        st.info(f"Models trained on: {model_info['comparison_timestamp']}")

    def interactive_demo_page(self):
        """Interactive demo page for testing models"""
        st.markdown('<h2 class="section-header">🔮 Interactive Demo</h2>', 
                   unsafe_allow_html=True)

        st.subheader("Test Sentiment Analysis")

        # Text input
        user_text = st.text_area(
            "Enter text to analyze sentiment:",
            height=100,
            placeholder="Type your review, comment, or any text here...",
            help="Enter any text and see how both models predict its sentiment"
        )

        if user_text.strip():
            # Preprocess text
            data_loader = DataLoader(st.session_state.selected_dataset)
            data_loader.tokenizer = st.session_state.tokenizer
            processed_text = data_loader.preprocess_single_text(user_text)

            st.markdown("---")

            # Get predictions from both models
            col1, col2 = st.columns(2)

            # ANN predictions
            if st.session_state.ann_model:
                with col1:
                    st.subheader("🔹 ANN Prediction")

                    ann_proba = st.session_state.ann_model.predict_proba(processed_text)[0]
                    ann_pred = "Positive" if ann_proba[1] > 0.5 else "Negative"
                    ann_confidence = max(ann_proba)

                    # Display prediction with styling
                    if ann_pred == "Positive":
                        st.markdown(f"""
                        <div class="prediction-positive">
                        <h3>😊 Positive Sentiment</h3>
                        <p>Confidence: {ann_confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-negative">
                        <h3>😞 Negative Sentiment</h3>
                        <p>Confidence: {ann_confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability visualization
                    fig_ann = go.Figure([go.Bar(
                        x=["Negative", "Positive"],
                        y=ann_proba,
                        marker_color=['#ff6b6b', '#51cf66'],
                        text=[f"{p:.1%}" for p in ann_proba],
                        textposition='auto'
                    )])
                    fig_ann.update_layout(
                        title="ANN Confidence Scores",
                        yaxis_title="Probability",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_ann, use_container_width=True)

            # RNN predictions
            if st.session_state.rnn_model:
                with col2:
                    st.subheader("🔹 RNN Prediction")

                    rnn_proba = st.session_state.rnn_model.predict_proba(processed_text)[0]
                    rnn_pred = "Positive" if rnn_proba[1] > 0.5 else "Negative"
                    rnn_confidence = max(rnn_proba)

                    # Display prediction with styling
                    if rnn_pred == "Positive":
                        st.markdown(f"""
                        <div class="prediction-positive">
                        <h3>😊 Positive Sentiment</h3>
                        <p>Confidence: {rnn_confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-negative">
                        <h3>😞 Negative Sentiment</h3>
                        <p>Confidence: {rnn_confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability visualization
                    fig_rnn = go.Figure([go.Bar(
                        x=["Negative", "Positive"],
                        y=rnn_proba,
                        marker_color=['#ff6b6b', '#51cf66'],
                        text=[f"{p:.1%}" for p in rnn_proba],
                        textposition='auto'
                    )])
                    fig_rnn.update_layout(
                        title="RNN Confidence Scores",
                        yaxis_title="Probability",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_rnn, use_container_width=True)

            # Agreement/Disagreement analysis
            if st.session_state.ann_model and st.session_state.rnn_model:
                st.markdown("---")
                st.subheader("🤝 Model Agreement")

                if ann_pred == rnn_pred:
                    st.success(f"✅ Both models agree: **{ann_pred}** sentiment")
                    confidence_diff = abs(ann_confidence - rnn_confidence)
                    st.info(f"Confidence difference: {confidence_diff:.1%}")
                else:
                    st.warning(f"⚠️ Models disagree: ANN says **{ann_pred}**, RNN says **{rnn_pred}**")
                    st.info("This might indicate ambiguous or nuanced sentiment in the text.")

        # Sample texts for testing
        st.markdown("---")
        st.subheader("💡 Try These Sample Texts")

        sample_texts = [
            "This movie was absolutely fantastic! Amazing acting and incredible plot twists.",
            "I really didn't enjoy this film. The storyline was confusing and poorly executed.",
            "The restaurant was decent. Nothing extraordinary but the service was okay.",
            "Best product I've ever purchased! Exceeded all my expectations completely.",
            "Total waste of money. Poor quality and terrible customer support experience.",
            "The book was interesting but had some pacing issues in the middle chapters.",
            "Outstanding performance by the entire cast. Highly recommend watching this!",
            "Not sure how I feel about this. It was okay but nothing special really."
        ]

        for i, sample in enumerate(sample_texts):
            if st.button(f"📝 Test Sample {i+1}", key=f"sample_{i}"):
                st.text_area("Sample Text:", value=sample, height=60, key=f"sample_display_{i}")

    def model_comparison_page(self):
        """Model comparison page"""
        st.markdown('<h2 class="section-header">⚖️ Model Comparison</h2>', 
                   unsafe_allow_html=True)

        model_info = st.session_state.model_info
        ann_metrics = model_info['ann_info']['metrics']
        rnn_metrics = model_info['rnn_info']['metrics']

        # Metrics comparison
        st.subheader("Performance Metrics Comparison")

        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'ANN Model': [
                ann_metrics['accuracy'],
                ann_metrics['precision'], 
                ann_metrics['recall'],
                ann_metrics['f1_score']
            ],
            'RNN Model': [
                rnn_metrics['accuracy'],
                rnn_metrics['precision'],
                rnn_metrics['recall'], 
                rnn_metrics['f1_score']
            ]
        }

        comparison_df = pd.DataFrame(metrics_data)
        comparison_df['Difference (RNN - ANN)'] = comparison_df['RNN Model'] - comparison_df['ANN Model']

        # Style the dataframe
        st.dataframe(
            comparison_df.style.format({
                'ANN Model': '{:.4f}',
                'RNN Model': '{:.4f}',
                'Difference (RNN - ANN)': '{:.4f}'
            }).highlight_max(axis=1, subset=['ANN Model', 'RNN Model']),
            use_container_width=True
        )

        # Visual comparison
        st.subheader("Visual Performance Comparison")

        metrics_dict = {"ANN": ann_metrics, "RNN": rnn_metrics}
        fig_comparison = self.evaluator.create_metrics_comparison(metrics_dict)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Winner analysis
        st.subheader("🏆 Performance Analysis")

        best_model = model_info['best_model_type']
        if best_model == 'ann':
            winner_metrics = ann_metrics
            winner_name = "ANN"
            loser_metrics = rnn_metrics
            loser_name = "RNN"
        else:
            winner_metrics = rnn_metrics
            winner_name = "RNN"
            loser_metrics = ann_metrics
            loser_name = "ANN"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                f"{winner_name} Accuracy",
                f"{winner_metrics['accuracy']:.4f}",
                delta=f"{winner_metrics['accuracy'] - loser_metrics['accuracy']:.4f}"
            )

        with col2:
            st.metric(
                f"{winner_name} F1-Score", 
                f"{winner_metrics['f1_score']:.4f}",
                delta=f"{winner_metrics['f1_score'] - loser_metrics['f1_score']:.4f}"
            )

        with col3:
            st.metric(
                f"{winner_name} Precision",
                f"{winner_metrics['precision']:.4f}",
                delta=f"{winner_metrics['precision'] - loser_metrics['precision']:.4f}"
            )

        # Model characteristics
        st.subheader("🔍 Model Characteristics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ANN Model Strengths:**")
            st.write("• Faster inference time")
            st.write("• Simpler architecture") 
            st.write("• Less memory usage")
            st.write("• Good for shorter texts")

        with col2:
            st.markdown("**RNN Model Strengths:**")
            st.write("• Better sequence understanding")
            st.write("• Handles longer texts well")
            st.write("• Captures word order")
            st.write("• Better context awareness")

    def performance_analysis_page(self):
        """Detailed performance analysis page"""
        st.markdown('<h2 class="section-header">📈 Performance Analysis</h2>', 
                   unsafe_allow_html=True)

        model_info = st.session_state.model_info

        # Training information
        st.subheader("🕐 Training Information")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ANN Model Training:**")
            ann_config = model_info['ann_info']['config']
            st.json({
                "Epochs": ann_config['epochs'],
                "Batch Size": ann_config['batch_size'],
                "Learning Rate": ann_config['learning_rate'],
                "Hidden Units": ann_config['hidden_units'],
                "Dropout Rate": ann_config['dropout_rate']
            })

        with col2:
            st.markdown("**RNN Model Training:**")
            rnn_config = model_info['rnn_info']['config']
            st.json({
                "Epochs": rnn_config['epochs'],
                "Batch Size": rnn_config['batch_size'], 
                "Learning Rate": rnn_config['learning_rate'],
                "LSTM Units": rnn_config['lstm_units'],
                "Bidirectional": rnn_config['bidirectional']
            })

        # Performance summary
        st.subheader("📊 Performance Summary")

        ann_metrics = model_info['ann_info']['metrics']
        rnn_metrics = model_info['rnn_info']['metrics']

        # Create comprehensive metrics table
        detailed_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'ANN': [ann_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1_score']],
            'RNN': [rnn_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1_score']],
        })

        detailed_metrics['Best Model'] = detailed_metrics.apply(
            lambda row: 'ANN' if row['ANN'] > row['RNN'] else 'RNN', axis=1
        )
        detailed_metrics['Performance Gap'] = abs(detailed_metrics['ANN'] - detailed_metrics['RNN'])

        st.dataframe(
            detailed_metrics.style.format({
                'ANN': '{:.4f}',
                'RNN': '{:.4f}',
                'Performance Gap': '{:.4f}'
            }).apply(lambda x: ['background-color: #d4edda' if v == detailed_metrics.loc[x.name, 'Best Model'] 
                               else '' for v in ['ANN', 'RNN']], axis=1, subset=['ANN', 'RNN']),
            use_container_width=True
        )

        # Usage recommendations
        st.subheader("💡 Usage Recommendations")

        best_model = model_info['best_model_type']

        if best_model == 'ann':
            st.success("""
            **Recommended: ANN Model**

            The ANN model performed better on this dataset. Use it when:
            • You need faster predictions
            • Working with shorter texts
            • Memory efficiency is important
            • Simple deployment is preferred
            """)
        else:
            st.success("""
            **Recommended: RNN Model**

            The RNN model performed better on this dataset. Use it when:
            • Analyzing longer texts
            • Word order and sequence matter
            • You need better context understanding
            • Higher accuracy is more important than speed
            """)

        # Model files information
        st.subheader("📁 Model Files")

        st.info(f"""
        **Model Files Location:**
        • ANN Model: `{model_info['ann_info']['model_path']}`
        • RNN Model: `{model_info['rnn_info']['model_path']}`
        • Tokenizer: `{model_info['tokenizer_path']}`
        • Best Model Info: `saved_models/{st.session_state.selected_dataset}/best_models.json`
        """)

# Run the application
if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.run()
