# 🛡️ PhishGuard - Human vs AI Phishing Detection Study

[![Live Demo](https://img.shields.io/badge/Live-Demo-00f0ff?style=for-the-badge&logo=github)](https://hhhpraise.github.io/phishing-detection-study/)
[![Portfolio](https://img.shields.io/badge/View-Portfolio-00ff88?style=for-the-badge)](https://hhhpraise.github.io/portfolio)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

> An interactive research platform comparing human and AI capabilities in detecting phishing emails across three experimental modes: Human-Only, AI-Only, and Hybrid collaboration.


---

## 📊 Research Overview

PhishGuard is a cybersecurity research platform that investigates the effectiveness of human-AI collaboration in identifying phishing emails. The study examines:

- **Human vs AI Performance**: Who detects phishing more accurately?
- **Decision Time Analysis**: Does AI assistance speed up or slow down decisions?
- **Trust Calibration**: Do humans appropriately trust AI predictions?
- **Hybrid Effectiveness**: Does human-AI collaboration outperform either alone?
- **Confidence Analysis**: Are high-confidence predictions more reliable?

### 🎯 Key Research Questions

1. How does human accuracy compare to AI in phishing detection?
2. Does showing AI predictions improve or bias human judgment?
3. When do humans override AI recommendations, and why?
4. What is the optimal model for human-AI collaboration in cybersecurity?

---

## ✨ Features

### 🤖 **AI-Powered Detection**
- Real-time phishing detection using TensorFlow.js
- Browser-based ML model (no server required)
- Confidence scoring for each prediction
- Trained on 2,000+ real phishing and legitimate emails

### 🎮 **Interactive Testing Modes**

#### Mode 1: Human Only
- Make decisions independently without AI assistance
- Baseline for human performance measurement

#### Mode 2: AI Only (Sequential)
- Human decides first, then sees AI prediction
- Measures willingness to change mind based on AI

#### Mode 3: Hybrid (Collaborative)
- AI prediction shown before human decision
- Evaluates AI influence on human judgment

### 📈 **Real-Time Data Collection**
- Automatic saving to Google Sheets via API
- Anonymous participant tracking
- Comprehensive metrics: accuracy, time, confidence
- Backup storage with local download option

### 🎨 **Modern UI/UX**
- Cybersecurity-themed dark interface
- Smooth animations and transitions
- Fully responsive (desktop, tablet, mobile)
- Accessible and intuitive design

---

## 🚀 Live Demo

**Try it now:** [https://hhhpraise.github.io/phishing-detection-study/](https://hhhpraise.github.io/phishing-detection-study/)

**Duration:** 15-20 minutes  
**Task:** Classify 15 emails as phishing or safe  
**No personal data collected** - Completely anonymous

---

## 🛠️ Technology Stack

### Machine Learning
- **Framework**: TensorFlow / Keras
- **Training**: Python with GPU optimization (CUDA)
- **Deployment**: TensorFlow.js (browser-based inference)
- **Model**: Neural Network with TF-IDF features
- **Performance**: 85%+ accuracy on test set

### Frontend
- **Languages**: HTML5, CSS3, JavaScript (ES6+)
- **ML Library**: TensorFlow.js
- **Design**: Custom CSS with dark cybersecurity theme
- **Responsive**: Mobile-first approach

### Data Collection
- **Storage**: Google Sheets API
- **Authentication**: Google Apps Script
- **Backup**: Browser localStorage
- **Format**: Structured JSON to spreadsheet

### Deployment
- **Hosting**: GitHub Pages
- **Version Control**: Git/GitHub
- **CI/CD**: Automatic deployment on push

---

## 📊 Dataset

**Source**: Phishing Email Validation Dataset  
**Provider**: Sofia Tech Park - AI & CAD Systems Laboratory  
**Size**: 2,000 emails (balanced phishing/safe)  
**Format**: CSV with full email text and binary labels  
**Split**: 80% training (1,600) / 20% testing (400)

### Data Preprocessing
- Text normalization and cleaning
- URL pattern extraction
- TF-IDF vectorization (5,000 features)
- Stratified sampling for balanced splits

---

## 🧪 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.5% |
| **Precision** | 89.2% |
| **Recall** | 85.8% |
| **F1-Score** | 87.5% |
| **ROC-AUC** | 0.92 |

### Confusion Matrix
```
                Predicted
               Safe  Phishing
Actual Safe     178      22
      Phishing   28     172
```

---

## 📁 Project Structure

```
phishing-detection-study/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and split data
│   └── test_emails_selected.json  # 15 curated test emails
├── models/
│   ├── phishing_model.h5       # Trained Keras model
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   └── training_history.png    # Training curves
├── src/
│   ├── data_exploration.py     # Dataset analysis
│   ├── data_preprocessing.py   # Data cleaning pipeline
│   ├── train_model.py          # Model training script
│   ├── evaluate_model.py       # Model evaluation
│   ├── convert_to_tfjs.py      # TensorFlow.js conversion
│   └── select_test_emails.py   # Test email curation
├── web/
│   ├── index.html              # Main application
│   ├── styles.css              # Enhanced UI styling
│   ├── app.js                  # Core application logic
│   ├── phishguard_model.js     # ML model interface
│   ├── google_sheets.js        # Data collection API
│   ├── config.js               # Configuration
│   └── model/                  # TensorFlow.js model files
├── assets/                     # Images and media
├── README.md
└── requirements.txt
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js (optional, for development)
- CUDA-enabled GPU (optional, for training)

### For ML Development

1. **Clone the repository**
```bash
git clone https://github.com/hhhpraise/phishing-detection-study.git
cd phishing-detection-study
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
python src/convert_to_tfjs.py
```

### For Web Development

1. **Local testing**
```bash
python -m http.server 8000
# Visit http://localhost:8000/web
```

2. **Deploy to GitHub Pages**
- Push to `main` branch
- Enable GitHub Pages in repository settings
- Select branch and folder (root or /docs)

---

## 📊 Data Collection Schema

Data is automatically saved to Google Sheets with the following fields:

| Field | Description | Type |
|-------|-------------|------|
| `timestamp` | Response timestamp | ISO DateTime |
| `participant_id` | Anonymous UUID | String |
| `session_id` | Unique session ID | String |
| `email_id` | Test email identifier | String |
| `email_ground_truth` | Actual label | phishing/safe |
| `test_mode` | Experimental condition | human_only/ai_only/hybrid |
| `human_decision` | User's classification | phishing/safe |
| `human_confidence` | User's confidence | 1-5 scale |
| `ai_prediction` | AI's classification | phishing/safe |
| `ai_confidence` | AI's confidence | 0.0-1.0 |
| `decision_time_seconds` | Response time | Float |
| `agreed_with_ai` | User-AI agreement | Boolean |
| `changed_mind` | Changed after AI reveal | Boolean |
| `user_feedback` | Optional comments | String |
| `country` | User location | String |
| `device_type` | Device category | Desktop/Mobile/Tablet |

---

## 🔐 Privacy & Ethics

- ✅ **No Personal Data**: Only anonymous UUIDs collected
- ✅ **Informed Consent**: Clear consent process before participation
- ✅ **Voluntary**: Participants can stop at any time
- ✅ **Secure Storage**: Data stored in private Google Sheets
- ✅ **Research Purpose**: Data used only for cybersecurity research
- ✅ **Transparency**: Full methodology disclosed

---

## 📈 Preliminary Findings

*Note: Findings will be updated after collecting sufficient data*

### Expected Outcomes
- AI likely to have higher raw accuracy than average users
- Expert users may match or exceed AI performance
- Hybrid mode expected to show improved accuracy
- AI influence may introduce bias in certain scenarios
- High-confidence AI predictions likely more reliable

### Analysis Plan
- Descriptive statistics and visualizations
- Statistical significance testing (t-tests, ANOVA)
- Error analysis and pattern identification
- Confidence calibration analysis
- Demographic comparisons

---

## 🎓 Use Cases & Applications

### Research
- Human-AI collaboration studies
- Trust in AI systems
- Decision-making under uncertainty
- Cybersecurity education effectiveness

### Industry
- Training tool for security awareness
- Benchmark for phishing detection systems
- User interface design for AI-assisted tools
- Trust calibration in AI recommendations

### Education
- Interactive learning platform
- Phishing awareness training
- Critical thinking development
- AI literacy education

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue with detailed description
2. **Suggest Features**: Share ideas for improvements
3. **Improve Documentation**: Fix typos or add clarity
4. **Code Contributions**: Fork, create branch, submit PR

### Development Guidelines
- Follow existing code style
- Comment complex logic
- Test thoroughly before submitting
- Update documentation as needed

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{phishguard2024,
  author = {Your Name},
  title = {PhishGuard: Human vs AI Phishing Detection Study},
  year = {2024},
  url = {https://github.com/hhhpraise/phishing-detection-study},
  note = {Interactive research platform for studying human-AI collaboration in cybersecurity}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Sofia Tech Park - AI & CAD Systems Laboratory
- **Inspiration**: Growing need for effective phishing detection
- **Tools**: TensorFlow, TensorFlow.js, Google Sheets API
- **Community**: Open-source ML and cybersecurity communities

---

## 📞 Contact & Links

- **Portfolio**: [https://hhhpraise.github.io/portfolio](https://hhhpraise.github.io/portfolio)
- **GitHub**: [@hhhpraise](https://github.com/hhhpraise)
- **Live Demo**: [PhishGuard Study](https://hhhpraise.github.io/phishing-detection-study/)

---

## 🗺️ Roadmap

### Current Version (v1.0)
- [x] Core ML model training and deployment
- [x] Three experimental modes implemented
- [x] Google Sheets data collection
- [x] Responsive web interface
- [x] GitHub Pages deployment

### Future Enhancements (v2.0)
- [ ] Multi-language support (Chinese, Spanish, etc.)
- [ ] Advanced AI models (BERT, transformers)
- [ ] Real-time analytics dashboard
- [ ] Gamification elements (leaderboards, achievements)
- [ ] Email simulation builder
- [ ] API for researchers
- [ ] Mobile app version

### Research Extensions
- [ ] Longitudinal study (tracking improvement over time)
- [ ] Expert vs novice comparison
- [ ] Cultural differences in trust and detection
- [ ] Explainable AI integration
- [ ] Active learning from human feedback

---

## 📊 Project Status

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-85%25-green)
![Model Accuracy](https://img.shields.io/badge/model%20accuracy-87.5%25-blue)
![Participants](https://img.shields.io/badge/participants-TBD-orange)

**Status**: Active Development & Data Collection  
**Last Updated**: January 2025  
**Version**: 1.0.0

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Help advance cybersecurity research by participating in the study**

[![Take the Study](https://img.shields.io/badge/Take%20the%20Study-Start%20Now-00f0ff?style=for-the-badge&logo=shield)](https://hhhpraise.github.io/phishing-detection-study/)

Made with ❤️ for cybersecurity research

</div>

---

## 🐛 Known Issues

- **Issue #1**: Mobile keyboard may obscure input fields on small screens
  - *Workaround*: Scroll down after tapping input
- **Issue #2**: WeChat browser may have CORS limitations
  - *Status*: Under investigation

Report new issues [here](https://github.com/hhhpraise/phishing-detection-study/issues)


---

**Disclaimer**: This is a research project. The AI model is for educational purposes and should not be used as the sole means of phishing detection in production environments.