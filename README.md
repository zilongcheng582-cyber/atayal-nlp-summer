## Low-Resource Language NLP Attempt: Ataya
A low-resource NLP pipeline for the Zeyuan dialect of Atayal (澤敖利泰雅語),
based on aligned text and audio data from Taiwan's K-12 indigenous language curriculum.

## Project Overview
This project builds a complete NLP pipeline for Atayal, an endangered Formosan language
spoken by the Atayal people of Taiwan. Using 688 aligned text-audio sentence pairs,
we develop corpus analysis tools, a tokenization pipeline, a character-level LSTM
language model, and audio feature extraction.

## Results
- Corpus: 688 sentences, 10,588 total words, 1,851 unique words, 61 unique characters
- LSTM trained for 10 epochs, final loss: 0.3163
- Model successfully generates Atayal-like text
- MFCC features extracted from aligned audio samples (3–41 seconds)


## Repository Structure
```
notebooks/
├── 01_corpus_analysis.ipynb      ← corpus statistics and character mapping
├── 02_tokenization.ipynb         ← text encoding and sequence generation├── 03_lstm_language_model.ipynb  ← character-level LSTM training
└── 04_mfcc_analysis.ipynb        ← audio feature extraction and visualization
data/
├── raw/                          ← 12 chapters of Atayal text (.pyu files)
└── processed/                    ← cleaned text (688 sentences)
audio/
└── samples/                      ← aligned wav samples for MFCC analysis
src/
├── preprocess.py                 ← data loading and cleaning pipeline
├── model.py                      ← AtayalLSTM model definition
└── train.py                      ← training script
```


## Data Source
Aligned text and audio from Taiwan's 十二年國教 Atayal (Zeyuan dialect) curriculum.
Text and audio are sentence-level aligned across 12 chapters.

## Setup
Clone the repo and open any notebook directly in Google Colab.
All notebooks automatically download data from this repository — no manual setup needed.

## Tech Stack
- Python 3
- PyTorch
- librosa
- Google Colab

## Future Work
- Expand corpus with more Atayal dialect data
- Build speech recognition pipeline using MFCC + text alignment
- Explore cross-dialect comparison (other Atayal dialects)
- Fine-tune model with larger dataset

## Author
First-year CS undergraduate NLP attempt.
