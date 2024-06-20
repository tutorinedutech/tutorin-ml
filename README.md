
# Tutorin-ML

## Project Overview

This project is a machine learning-based recommendation system, specifically designed for mentor-mentee matching.

## Directory Structure

```plaintext
- .gitignore
- README.md
- converted_model_ranking.tflite
- converted_model_retrieval.tflite
- embedding.json
- mentor_embeddings.json
- requirements.txt
- setup.py
- data_source/
  - Data_RecomenderSystem.xlsx
- embedding/
  - candidate_embedding.py
- notebooks/
  - recomendation-system-input-user (1).ipynb
  - train-model-siamese.ipynb
- save_models/
  - candidate_models/
  - query_models/
  - ranking_models/
  - retrieval_models/
- src/
  - apps/
    - app.py
  - data/
    - data_prep.py
  - evaluation/
    - evaluation.py
    - ranking_eval.py
  - model/
    - ranking_model.py
    - retrieval_model.py
  - tflite/
    - ranking_tflite.py
    - retrieval_tflite.py
  - train/
    - ranking_training.py
    - retrieval_training.py
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repository_url>
   cd tutorin-ml-main
   ```

2. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up the project:**
   ```sh
   python setup.py install
   ```

## Usage

1. **Data Preparation:**
   - Use the scripts in `src/data/` for data cleaning and preprocessing.
   - `data_loader.py` loads the data, and `pre_processing.py` handles initial preprocessing tasks.

2. **Embedding:**
   - Generate candidate embeddings using `embedding/candidate_embedding.py`.

3. **Model Training:**
   - Train the retrieval model using `src/train/retrieval_training.py`.
   - Train the ranking model using `src/train/ranking_training.py`.
   - You can also use the provided Jupyter notebooks in `notebooks/` for interactive training and experimentation.

4. **Evaluation:**
   - Evaluate model performance using scripts in `src/evaluation/`.
   - `ranking_eval.py` and `evaluation.py` contain evaluation metrics and procedures.

5. **Deployment:**
   - Deploy the models using TensorFlow Lite scripts in `src/tflite/`.
   - `ranking_tflite.py` and `retrieval_tflite.py` convert the trained models to TensorFlow Lite format.

6. **Application:**
   - Run the application using `src/apps/app.py`.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Acknowledgments

- List any resources, libraries, or contributors you'd like to thank.
