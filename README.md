# URL Shield 🛡️

URL Shield is an intelligent web protection system that uses machine learning to classify URLs into different threat categories (Benign, Defacement, Malware, Phishing, Spam).

## Features 🌟

- **URL Analysis**: Extract 21 different features from URLs
- **Machine Learning**: Neural network-based classification
- **User Interface**: Clean and intuitive Streamlit-based GUI
- **Multiple Interfaces**: Both GUI and command-line options
- **High Accuracy**: Trained on a large dataset of various URL types

## Installation 🔧

1. Clone the repository: 

git clone https://github.com/kanavbajaj/url-shield.git
cd url-shield

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 💻

### GUI Interface
```bash
cd GUI
streamlit run gui.py
```
### Command Line Interface
```bash:README.md
cd scripts
python predict_args.py -i <url>
```
## Project Structure 📁
url-shield/
├── GUI/ # Streamlit web interface
├── scripts/ # Core functionality and training scripts
├── notebooks/ # Development and analysis notebooks
├── models/ # Trained model files
└── FinalDataset/ # Training data and features

## Contributing 🤝

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

