# Anomalies detection in limit order books data

This repository contains the source code and resources for the master's thesis **"Anomalies Detection in Limit Order Books Data"**, which focuses on detecting suspicious or manipulative trading behaviors (especially **spoofing**) in historical financial market data using unsupervised machine learning.

## 📌 Abstract

Modern financial markets are fast-paced, complex, and increasingly targeted by sophisticated forms of manipulation.
This thesis focuses on anomaly detection in time series derived from limit order books, aiming to identify manipulative behavior known as **spoofing**.
Due to the absence of annotated data, unsupervised machine learning methods are applied to real historical data.
Six methods are implemented:

- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- Fully Connected Autoencoder
- Convolutional Autoencoder
- Transformer-based Autoencoder

The models are evaluated using the less commonly known metrics **Excess Mass** and **Mass Volume**, with the Isolation Forest and Transformer models achieving the best results.
By combining the most effective models, a robust tool is created, capable of detecting suspicious behavior without manual annotation.
The proposed solution efficiently identifies high-risk areas for subsequent expert analysis and thus offers a practical contribution to detecting illicit practices in financial markets.

## 🛠️ Technologies Used

- **Python 3.11**
- [NumPy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Scikit-learn](https://pypi.org/project/scikit-learn/)
- [PyTorch](https://pypi.org/project/torch/)
- [Weights & Biases (WandB)](https://pypi.org/project/wandb/)
- [Plotly](https://pypi.org/project/plotly/)
- [Dash](https://dash.plotly.com/)

## 🚀 Getting Started

1. **Clone the repository** to your local machine using the following command:

```bash
git clone --recursive https://github.com/SpeekeR99/DP_2024_2025_Zappe.git
cd DP_2024_2025_Zappe
```

2. **Create a virtual environment** and **install dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `call venv\Scripts\activate.bat`
pip install -r requirements.txt
```

3. **Run** the pipeline

The full pipeline includes:

- Downloading the raw message style data from the exchange server
- Preprocessing the data
  - Reconstructing the order book
  - Feature extraction
- Model training and evaluation
- Visualization of the results

Steps:

- Run the `src/A7/download_eobi.py` script to download the raw messages
- Use the `src/data_preprocess/json-detailed2lobster.py` and `src/data_preprocess/augment_lobster.py` scripts to transform and enrich the data
- Train models using:
  - `src/anomaly_detection/models/autoencoder.py` for the autoencoder models
  - `src/anomaly_detection/models/if_ocsvm_lof.py` for the scikit-learn models
- Evaluation happens automatically during the training process 
- Finally, run `src/anomaly_detection/models/ensemble.py` for the final ensemble model and visualization of the detection results and evaluation metrics
- Optionally, use `src/visualization/visuals.py` to visualize the results in an interactive web application

## 📁 Repository Structure

```plaintext
data/                     # Original and preprocessed LOB datasets (.json, .csv)
doc/                      # Thesis document (.pdf)
img/                      # All figures, visualizations, evaluation results
lib/
  ├── diffi/              # Feature importance algorithm
  └── eval/               # EM/MV metric implementations
models/                   # Trained models (empty by default)
src/
  ├── A7/                 # EOBI and Orderbook data downloaders
  ├── data_preprocess/    # Data conversion and feature engineering
  ├── meta_centrum/       # Scripts for remote training (MetaCentrum)
  └── anomaly_detection/
       ├── models/        # All ML model training scripts
       ├── eval/          # Metric definitions and evaluation routines
       ├── data/          # Data loaders and utilities
       └── analysis/      # Feature selection and visualization tools
  └── visualization/      # Interactive dashboard
res/                      # Output anomaly detection results
requirements.txt          # Python dependencies
```

---

# Detekce anomálií v datech z knih limitních objednávek

Tento repozitář obsahuje zdrojový kód a související materiály k diplomové práci **„Detekce anomálií v datech z knih limitních objednávek“**, která se zaměřuje na detekci podezřelého nebo manipulativního chování (zejména **spoofingu**) v historických datech z finančních trhů pomocí metod učení bez učitele.

## 📌 Abstrakt

Moderní finanční trhy jsou rychlé, komplexní a stále častěji se stávají cílem sofistikovaných forem manipulace.
Tato práce se zaměřuje na detekci anomálií v časových řadách odvozených z knih limitních objednávek s cílem rozpoznat manipulativní chování zvané **spoofing**.
Vzhledem k absenci anotovaných dat jsou použity metody strojového učení bez učitele aplikované na reálná historická data.
V práci je implementováno šest metod:

- izolační les
- lokální faktor odlehlosti
- jednotřídní SVM
- plně propojený autoenkodér
- konvoluční autoenkodér
- transformer autoenkodér

Modely jsou evaluovány pomocí méně známých metrik **Excess Mass** a **Mass Volume**, přičemž nejlépe si vedou modely izolační les a transformer.
Kombinací nejvýkonnějších modelů vznikl robustní nástroj schopný odhalit podezřelé chování bez ruční anotace.
Navržené řešení efektivně identifikuje rizikové oblasti pro následnou expertní analýzu a představuje tak praktický přínos pro detekci nelegálních praktik na finančních trzích.

## 🛠️ Použité technologie

- **Python 3.11**
- [NumPy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Scikit-learn](https://pypi.org/project/scikit-learn/)
- [PyTorch](https://pypi.org/project/torch/)
- [Weights & Biases (WandB)](https://pypi.org/project/wandb/)
- [Plotly](https://pypi.org/project/plotly/)
- [Dash](https://dash.plotly.com/)

## 🚀 Jak začít

1. **Naklonujte repozitář** do lokálního stroje pomocí příkazu:

```bash
git clone --recursive https://github.com/SpeekeR99/DP_2024_2025_Zappe.git
cd DP_2024_2025_Zappe
```

2. **Vytvořte virtuální prostředí** a **nainstalujte závislosti**

```bash
python -m venv venv
source venv/bin/activate  # Na Windows použijte `call venv\Scripts\activate.bat`
pip install -r requirements.txt
```

3. **Spusťte** pipeline

Celá pipeline zahrnuje:

- Stahování surových dat ve stylu zpráv z burzovního serveru
- Předzpracování dat
  - Rekonstrukce knihy objednávek
  - Extrakce příznaků
- Trénink a vyhodnocení modelu
- Vizualizace výsledků

Kroky:

- Spusťte skript `src/A7/download_eobi.py` pro stažení surových zpráv
- Použijte skripty `src/data_preprocess/json-detailed2lobster.py` a `src/data_preprocess/augment_lobster.py` pro transformaci a obohacení dat
- Natrénujte modely pomocí:
  - `src/anomaly_detection/models/autoencoder.py` pro autoenkodér modely
  - `src/anomaly_detection/models/if_ocsvm_lof.py` pro scikit-learn modely
- Vyhodnocení probíhá automaticky během tréninkového procesu
- Nakonec spusťte `src/anomaly_detection/models/ensemble.py` pro finální soubor modelů a vizualizaci výsledků detekce a hodnotících metrik
- Volitelně použijte `src/visualization/visuals.py` pro vizualizaci výsledků v interaktivní webové aplikaci

## 📁 Struktura repozitáře

```plaintext
data/                     # Původní a předzpracovaná LOB data (.json, .csv)
doc/                      # Dokument diplomové práce (.pdf)
img/                      # Všechny obrázky, vizualizace, hodnotící výsledky
lib/
  ├── diffi/              # Algoritmus pro důležitost příznaků
  └── eval/               # Implementace EM/MV metrik
models/                   # Natrénované modely (ve výchozím nastavení prázdné)
src/
  ├── A7/                 # EOBI a downloader knih objednávek
  ├── data_preprocess/    # Konverze dat a inženýrství příznaků
  ├── meta_centrum/       # Skripty pro vzdálený trénink (MetaCentrum)
  └── anomaly_detection/
	   ├── models/        # Všechny skripty pro trénink ML modelů
	   ├── eval/          # Definice metrik a vyhodnocovací rutiny
	   ├── data/          # Načítání dat a utility
	   └── analysis/      # Nástroje pro výběr příznaků a vizualizaci
  └── visualization/      # Interaktivní dashboard
res/                      # Výstupní výsledky detekce anomálií
requirements.txt          # Závislosti Pythonu
```
