# 🌏 Korean-English Translation Model  

This repository contains a **Transformer-based** model for Korean to English translation. The model was trained on conversational data to provide natural translations between **Korean** and **English** languages.  

## 📌 Overview  
This project implements a **Neural Machine Translation (NMT)** system using the **Transformer** architecture. It was trained on a dataset of **100,000 conversational Korean-English sentence pairs** from **AI-HUB**, a free community resource created to promote AI development in Korea.  

## ✨ Features  

✅ Transformer-based architecture for **high-quality translations**  
✅ Trained on **conversational data** for natural translations  
✅ Easy **configuration** through YAML files using **Hydra**  
✅ Simple **inference interface**  
✅ Achieved **🔥 29.9 BLEU Score**  

## 📂 Repository Structure  
```bash
ko-en-translation/
│── main.py                   # Entry point
│── dataset.py                # Data preprocessing
│── transformer.py            # Transformer model implementation
│── trainer.py                # Training script
│── translation.py            # Inference script
│── requirements.txt          # Dependencies
│── LICENSE                   # License file
│── README.md                 # Documentation
│── conf/                     # Configuration files
│   ├── config.yaml
│   ├── model/
│   │   ├── transformer.yaml
│   ├── dataset/
│   │   ├── ai_hub_conversation_100K.yaml
│   ├── inference/
│   │   ├── translation.yaml
│   ├── trainer/
│   │   ├── default.yaml
│── data/
│   ├── 대화체.xlsx             # Dataset file
```

## ⚡ Installation  
Clone the repository:  
```bash
git clone https://github.com/minseoc03/ko-en-translation.git
cd ko-en-translation
```
Install the required dependencies:  
```bash
pip install -r requirements.txt
```

## 🚀 Usage  
### 🔄 Translation  
To translate **Korean text to English**, run `main.py`:  
```bash
python main.py
```

### ⚙️ Configuration with Hydra  
Modify configuration parameters in `conf/inference/translation.yaml` or override them from the command line:  
```bash
python main.py inference.translation.src_text="안녕하세요. 어떻게 지내세요?"
```

## 📦 Pre-trained Model  
To use a **pre-trained model**, create a `pretrained/` folder and place the following files inside:  
```
Transformer.pt
Transformer_history.pt
```

## 📖 Examples  
#### ✅ Basic example  
Modify `src_text` in `conf/inference/translation.yaml`, then run:  
```bash
python main.py
```
Command line override examples:  
```bash
python main.py inference.translation.src_text="안녕하세요. 반갑습니다."
python main.py inference.translation.src_text="저는 한국어를 공부하고 있습니다."
```

## 🏗️ Model Details  
- **Architecture:** Transformer  
- **Training Data:** 100,000 Korean-English conversational sentence pairs from AI-HUB  
- **Tokenizer:** [HuggingFace Helsinki-NLP/opus-mt-ko-en](https://huggingface.co/Helsinki-NLP/opus-mt-ko-en)  

## 📜 License  
This project is licensed under the **[MIT License](LICENSE)**.  

## 🙌 Acknowledgements  
- **Dataset** : [AI-HUB Korean-English Conversation Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126)  
- **Tokenizer** : [HuggingFace Helsinki-NLP/opus-mt-ko-en](https://huggingface.co/Helsinki-NLP/opus-mt-ko-en)  
