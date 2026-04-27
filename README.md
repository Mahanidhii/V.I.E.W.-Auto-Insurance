# V.I.E.W. (Visual Inspection and Estimation Workflow) 
### Context-Aware Automated Vehicle Insurance Assessor Using a Multi-Modal ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow)

V.I.E.W. is an end-to-end multi-modal machine learning application designed to automate vehicle damage assessment and financial claim estimation for the insurance industry. By combining Deep Learning for computer vision and gradient boosting for tabular data, the pipeline delivers accurate, context-aware, and financially robust repair estimates.

## Key Features

The pipeline is split into two distinct yet integrated phases:

### Phase 1: Computer Vision Assessment
- **Damage Classification:** Utilizes a fine-tuned **EfficientNet-B0** model to analyze uploaded vehicle images.
- **Categorization:** Classifies damage into specific types (e.g., Scratch, Dent, Broken Glass) and maps them to severity levels (Minor, Moderate, Severe).

### Phase 2: Claim Estimation
- **Predictive Modeling:** Uses a **CatBoost Regressor** trained on contextual tabular data (vehicle age, make, fuel type, incident severity, etc.) to predict accurate claim amounts.
- **Actuarial Rule-Based Engine:** A hybrid engine validates the ML model's prediction against strict industry constraints (brand multipliers, depreciation curves, collision severity) to ensure logically sound estimates and avoid anomalies.

## Architecture

1. **Vision Model:** Transfer learning using `EfficientNet-B0`. Unfrozen top layers for fine-tuning on vehicle damage textures.
2. **Regression Engine:** `CatBoost` algorithm handling categorical and numerical features with engineered columns (e.g., premium-deductible ratio, age-based depreciation).
3. **Hybrid Verification:** Combines the CatBoost output (60%) with a deterministic Rule-Based baseline (40%) to handle edge cases effectively.
4. **User Interface:** Interactive UI built with `Streamlit`.

## Project Structure
```text
├── app/
│   ├── main.py                     # Streamlit application entry 
│   ├── pages/
│   │   ├── home.py                 # Landing page
│   │   ├── 1_vision_assessment.py  # Image upload & damage 
│   │   └── 2_claim_estimation.py   # Context input & claim 
│   └── utils/
│       ├── hybrid_engine.py        # Actuarial constraints & ML 
│       └── visualizer.py           # Charts & results 
├── src/
│   ├── dataset_parser.py           # Data preprocessing and 
│   ├── train_vision.py             # EfficientNet-B0 training 
│   └── train_regression.py         # CatBoost model training & 
├── models/                         # Saved .h5 and .cbm models + 
├── data/                           # Tabular CSVs and processed 
├── model_eval/                     # Evaluation metrics and 
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Context-Aware Automated Vehicle Insurance Assessor..."
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

To launch the Streamlit user interface, navigate to the project root and run:
```bash
streamlit run app/main.py
```

### Training the Models

If you wish to retrain the models with new data:

1. **Train the Vision Model (EfficientNet):**
   ```bash
   python src/train_vision.py
   ```
2. **Train the Regression Model (CatBoost):**
   ```bash
   python src/train_regression.py
   ```

## Technologies & Libraries

- **Frameworks:** Streamlit, TensorFlow / Keras, CatBoost, Scikit-Learn
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly, Altair
- **Languages:** Python
