# 🎭 Sentiment Analysis Deep Learning Project

A complete end-to-end sentiment analysis project comparing ANN vs RNN models. **Train models in terminal, then use the interactive web app for fast predictions and analysis.**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Terminal)
```bash
# Train models on IMDB dataset (recommended)
python train_models.py --dataset imdb

# Or train on Stanford Sentiment Treebank
python train_models.py --dataset stanford_sentiment

# Train only specific model
python train_models.py --dataset imdb --model ann
python train_models.py --dataset imdb --model rnn
```

### 3. Run Interactive Demo (Web App)
```bash
streamlit run app.py
```

**That's it! 🎉 The web app will load your trained models for fast inference.**

## 🎯 Project Philosophy

### Why This Approach?
- ⚡ **Fast UI**: No waiting hours for training in browser
- 🎯 **Optimized Training**: Best hyperparameters, automatic model selection
- 🔥 **Professional UX**: Users want to test models, not train them
- 💾 **Reusable Models**: Train once, use multiple times

## 📊 What You Get

### 🖥️ Terminal Training
- **Optimized Hyperparameters**: Pre-configured for best performance
- **Automatic Comparison**: Trains both models and selects the best
- **Progress Tracking**: Real-time training progress and metrics
- **Model Saving**: Saves best models automatically

### 🌐 Web Application  
- **Model Overview**: Performance metrics and architecture details
- **Interactive Demo**: Test models with your own text instantly
- **Model Comparison**: Side-by-side analysis of ANN vs RNN
- **Performance Analysis**: Comprehensive evaluation metrics

## 📁 Project Structure

```
sentiment_analysis_project/
├── train_models.py                 # 🚂 Terminal training script
├── app.py                          # 🌐 Streamlit web application  
├── config.py                       # ⚙️ Configuration settings
├── requirements.txt                # 📦 Dependencies
├── data_loader.py                  # 📊 Data processing
├── ann_model.py                    # 🧠 ANN model implementation
├── rnn_model.py                    # 🔄 RNN model implementation  
├── evaluation.py                   # 📈 Evaluation utilities
├── saved_models/                   # 💾 Trained model files
│   ├── imdb/                      
│   │   ├── best_models.json        # Model info and paths
│   │   ├── tokenizer.pkl           # Text tokenizer
│   │   ├── ann_model_*.h5          # Trained ANN model
│   │   └── rnn_model_*.h5          # Trained RNN model
│   └── stanford_sentiment/
└── README.md                       # 📚 This file
```

## 🧠 Model Architectures

### ANN (Artificial Neural Network)
```
Input (text) → Embedding → Global Average Pooling → Dense Layers → Output
• Embedding: 128 dimensions
• Hidden layers: [256, 128, 64] 
• Regularization: Dropout, Batch Normalization, L2
• Fast inference, good for shorter texts
```

### RNN (Recurrent Neural Network)  
```
Input (text) → Embedding → Bidirectional LSTM → Dense → Output
• Embedding: 128 dimensions
• LSTM units: 128 (bidirectional)
• Regularization: Spatial Dropout, Recurrent Dropout
• Better for longer texts, understands sequence
```

## 📊 Expected Performance

### IMDB Movie Reviews Dataset
- **Training Time**: 15-25 minutes (both models)
- **ANN Accuracy**: ~87-90%
- **RNN Accuracy**: ~89-92%
- **Dataset Size**: 50,000 reviews

### Stanford Sentiment Treebank
- **Training Time**: 10-18 minutes (both models)  
- **ANN Accuracy**: ~83-86%
- **RNN Accuracy**: ~85-88%
- **Dataset Size**: ~67,000 sentences

## 🎮 Web App Features

### 📊 Model Overview
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Architecture Details**: Layer configurations and hyperparameters
- **Training Information**: When and how models were trained
- **Best Model Identification**: Automatic winner selection

### 🔮 Interactive Demo  
- **Instant Predictions**: Test any text with both models
- **Confidence Scores**: Visual probability distributions
- **Model Agreement**: See when models agree/disagree
- **Sample Texts**: Pre-loaded examples for quick testing

### ⚖️ Model Comparison
- **Side-by-Side Metrics**: Performance comparison table
- **Visual Charts**: Interactive comparison visualizations  
- **Strength Analysis**: When to use each model
- **Performance Gaps**: Detailed difference analysis

### 📈 Performance Analysis
- **Training Configuration**: Hyperparameters used
- **Detailed Metrics**: Comprehensive performance breakdown
- **Usage Recommendations**: Which model to use when
- **File Locations**: Where models are saved

## 🛠️ Advanced Usage

### Custom Training Configuration
```python
# Edit train_models.py to customize hyperparameters
ann_config = {
    'embedding_dim': 128,      # Embedding size
    'hidden_units': [256, 128, 64],  # Network depth
    'dropout_rate': 0.3,       # Regularization
    'learning_rate': 0.001,    # Learning speed
    'batch_size': 64,          # Training batch size
    'epochs': 25               # Training epochs
}
```

### Multiple Dataset Training
```bash
# Train on both datasets
python train_models.py --dataset imdb
python train_models.py --dataset stanford_sentiment

# Web app will let you switch between them
streamlit run app.py
```

### Model Comparison
```bash
# Train different configurations and compare
python train_models.py --dataset imdb --model ann
python train_models.py --dataset imdb --model rnn

# View results in web app
streamlit run app.py
```

## 🎯 Use Cases

### 🎓 Educational
- **Learn Deep Learning**: Compare ANN vs RNN architectures
- **Understand NLP**: Text preprocessing and tokenization
- **Model Evaluation**: Comprehensive metrics analysis
- **Interactive Learning**: Test models with your own examples

### 💼 Professional  
- **Portfolio Project**: Showcase ML engineering skills
- **Client Demos**: Interactive model demonstration
- **Research Tool**: Experiment with different architectures
- **Production Template**: Deploy sentiment analysis systems

### 🔬 Research
- **Architecture Comparison**: ANN vs RNN performance analysis
- **Dataset Analysis**: Compare performance across datasets
- **Hyperparameter Impact**: Test different configurations
- **Model Interpretability**: Understand prediction confidence

## 🔧 Troubleshooting

### Common Issues

**No models found error:**
```bash
# Make sure you trained models first
python train_models.py --dataset imdb
```

**Training too slow:**
```bash  
# Reduce batch size or epochs in train_models.py
# Or use a smaller dataset
```

**Memory errors:**
```bash
# Reduce batch_size in config.py
# Or reduce model complexity
```

**Poor performance:**
```bash
# Try different dataset
# Increase training epochs
# Adjust learning rate
```

## 📦 Dependencies

### Core Requirements
- **streamlit**: Web application framework
- **tensorflow**: Deep learning models
- **numpy, pandas**: Data processing
- **plotly**: Interactive visualizations
- **scikit-learn**: Evaluation metrics
- **nltk**: Text preprocessing

### Optional Enhancements
- **datasets**: Hugging Face datasets (Stanford Sentiment)
- **wordcloud**: Word cloud generation
- **optuna**: Hyperparameter optimization

## 🚀 Deployment Options

### Local Development
```bash
git clone <repository>
cd sentiment_analysis_project
pip install -r requirements.txt
python train_models.py --dataset imdb
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python train_models.py --dataset imdb
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Web app hosting with model files
- **AWS/GCP**: Scalable cloud deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Train and test your changes
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the deep learning framework
- **Streamlit Team** for the amazing web app framework  
- **Stanford NLP Group** for the Sentiment Treebank dataset
- **Andrew Maas et al.** for the IMDB Movie Reviews dataset
- **Hugging Face** for dataset access and tools

## 📞 Support

Having issues? Check out:

1. **Training Issues**: See training script output for errors
2. **Model Loading**: Ensure models exist in `saved_models/` directory
3. **Web App Issues**: Check Streamlit logs in terminal
4. **Performance**: Try different datasets or configurations

---

**Ready to analyze sentiment like a pro! 🎭✨**

### Quick Commands Summary:
```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (15-25 minutes)
python train_models.py --dataset imdb

# 3. Demo (instant)
streamlit run app.py
```
