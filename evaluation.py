"""
Evaluation and visualization utilities for sentiment analysis models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Any
import streamlit as st
from wordcloud import WordCloud
from config import VIZ_CONFIG

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""

    def __init__(self):
        self.colors = VIZ_CONFIG["color_palette"]

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Comprehensive model evaluation"""

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }

        if y_pred_proba is not None:
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics["roc_auc"] = auc(fpr, tpr)

            # PR AUC
            metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba)

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str] = None) -> go.Figure:
        """Create interactive confusion matrix heatmap"""

        if class_names is None:
            class_names = ["Negative", "Positive"]

        cm = confusion_matrix(y_true, y_pred)

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=True
        ))

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=500,
            height=400
        )

        return fig

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str = "Model") -> go.Figure:
        """Create ROC curve plot"""

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(color=self.colors[0], width=2)
        ))

        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=500,
            height=400,
            showlegend=True
        )

        return fig

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model") -> go.Figure:
        """Create Precision-Recall curve plot"""

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{model_name} (AUC = {pr_auc:.3f})',
            line=dict(color=self.colors[1], width=2),
            fill='tonexty'
        ))

        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=500,
            height=400,
            showlegend=True
        )

        return fig

    def plot_training_history(self, history: Dict[str, List[float]], 
                            model_name: str = "Model") -> go.Figure:
        """Plot training history (loss and accuracy)"""

        epochs = list(range(1, len(history['loss']) + 1))

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['loss'], 
                      mode='lines', name='Training Loss',
                      line=dict(color=self.colors[0])),
            row=1, col=1
        )

        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_loss'], 
                          mode='lines', name='Validation Loss',
                          line=dict(color=self.colors[1])),
                row=1, col=1
            )

        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['accuracy'], 
                      mode='lines', name='Training Accuracy',
                      line=dict(color=self.colors[2]), showlegend=False),
            row=1, col=2
        )

        if 'val_accuracy' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_accuracy'], 
                          mode='lines', name='Validation Accuracy',
                          line=dict(color=self.colors[3]), showlegend=False),
                row=1, col=2
            )

        fig.update_layout(
            title=f"{model_name} Training History",
            width=1000,
            height=400
        )

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)

        return fig

    def create_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create comparative metrics visualization"""

        models = list(metrics_dict.keys())
        metric_names = list(metrics_dict[models[0]].keys())

        fig = go.Figure()

        x_pos = np.arange(len(metric_names))
        bar_width = 0.35

        for i, model in enumerate(models):
            values = [metrics_dict[model][metric] for metric in metric_names]

            fig.add_trace(go.Bar(
                x=[f"{metric}" for metric in metric_names],
                y=values,
                name=model,
                marker_color=self.colors[i % len(self.colors)],
                text=[f"{val:.3f}" for val in values],
                textposition='auto'
            ))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            width=800,
            height=500,
            showlegend=True
        )

        return fig

    def plot_class_distribution(self, labels: np.ndarray, 
                              class_names: List[str] = None) -> go.Figure:
        """Plot class distribution pie chart"""

        if class_names is None:
            class_names = ["Negative", "Positive"]

        unique, counts = np.unique(labels, return_counts=True)

        fig = go.Figure(data=[go.Pie(
            labels=[class_names[i] for i in unique],
            values=counts,
            hole=0.3,
            marker_colors=self.colors[:len(unique)]
        )])

        fig.update_layout(
            title="Class Distribution",
            width=400,
            height=400
        )

        return fig

    def plot_sequence_length_distribution(self, sequences: List[List[int]], 
                                        title: str = "Sequence Length Distribution") -> go.Figure:
        """Plot sequence length distribution"""

        lengths = [len([x for x in seq if x != 0]) for seq in sequences]

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=lengths,
            nbinsx=50,
            name="Sequence Lengths",
            marker_color=self.colors[0],
            opacity=0.7
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Sequence Length",
            yaxis_title="Frequency",
            width=600,
            height=400
        )

        # Add statistics annotations
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)

        fig.add_vline(x=mean_length, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_length:.1f}")
        fig.add_vline(x=median_length, line_dash="dash", line_color="green",
                     annotation_text=f"Median: {median_length:.1f}")

        return fig

    def create_word_cloud(self, texts: List[str], sentiment: str = "positive") -> WordCloud:
        """Create word cloud for sentiment analysis"""

        # Combine all texts
        combined_text = " ".join(texts)

        # Color scheme based on sentiment
        if sentiment.lower() == "positive":
            colormap = "Greens"
        else:
            colormap = "Reds"

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5
        ).generate(combined_text)

        return wordcloud

    def display_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    class_names: List[str] = None) -> pd.DataFrame:
        """Generate and display classification report as DataFrame"""

        if class_names is None:
            class_names = ["Negative", "Positive"] 

        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )

        # Convert to DataFrame for better display
        df_report = pd.DataFrame(report).transpose()

        return df_report

def create_metric_cards(metrics: Dict[str, float]) -> None:
    """Create metric cards for Streamlit display"""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Accuracy",
            value=f"{metrics.get('accuracy', 0):.3f}",
            delta=None
        )

    with col2:
        st.metric(
            label="Precision", 
            value=f"{metrics.get('precision', 0):.3f}",
            delta=None
        )

    with col3:
        st.metric(
            label="Recall",
            value=f"{metrics.get('recall', 0):.3f}",
            delta=None
        )

    with col4:
        st.metric(
            label="F1-Score",
            value=f"{metrics.get('f1_score', 0):.3f}",
            delta=None
        )

def compare_models_table(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create comparison table for multiple models"""

    df = pd.DataFrame(metrics_dict).transpose()

    # Round values for better display
    df = df.round(4)

    # Add rank column for each metric
    for col in df.columns:
        df[f"{col}_rank"] = df[col].rank(ascending=False)

    return df
