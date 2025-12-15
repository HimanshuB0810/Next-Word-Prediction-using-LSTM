# Next Word Prediction using LSTM

This project implements a Next Word Prediction model using Deep Learning techniques, specifically Long Short-Term Memory (LSTM) networks. The model is trained on the text of Shakespeare's *Hamlet* and deployed as an interactive web application using Streamlit.

## 📌 Project Overview

The goal of this project is to predict the most likely next word in a sequence of text. It utilizes natural language processing (NLP) for data preprocessing and a sequential LSTM neural network for the prediction task.

## 📂 Project Structure

- **`app.py`**: The main Streamlit application file. It loads the trained model and tokenizer to provide an interactive interface for users to input text and get predictions.
- **`experiments.ipynb`**: A Jupyter Notebook containing the complete pipeline:
  - Data collection (downloading *Hamlet* from NLTK Gutenberg corpus).
  - Data preprocessing (tokenization, creating n-gram sequences, padding).
  - Model architecture definition and training.
  - Evaluation and testing.
  - Saving the trained model (`next_word_lstm.h5`) and tokenizer (`tokenizer.pkl`).
- **`hamlet.txt`**: The text corpus used for training the model.
- **`requirements.txt`**: List of Python dependencies required to run the project.
- **`next_word_lstm.h5`**: The trained Keras model file (generated after running the notebook).
- **`tokenizer.pkl`**: The pickled tokenizer object (generated after running the notebook).

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or download the source code.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Training the Model
To generate the model and tokenizer files, run the Jupyter Notebook:
1.  Open `experiments.ipynb` in Jupyter or Google Colab.
2.  Run all cells to download the data, preprocess it, and train the LSTM network.
3.  This will create `next_word_lstm.h5` and `tokenizer.pkl` in your directory.

### 2. Running the Application
Once the model is trained, you can launch the web application:
```bash
streamlit run app.py
````

This will open a local web server (usually at `http://localhost:8501`) where you can interact with the predictor.

## 🧠 Model Architecture

The model is built using TensorFlow/Keras with the following layers:

1.  **Embedding Layer**: Converts word indices into dense vectors of fixed size.
2.  **LSTM Layer 1**: A Long Short-Term Memory layer with 150 units (returns sequences).
3.  **Dropout Layer**: Applied with a rate of 0.2 to prevent overfitting.
4.  **LSTM Layer 2**: A second LSTM layer with 100 units.
5.  **Dense Layer**: The output layer with a softmax activation function to predict the probability of the next word among the total vocabulary.

## 📝 Example

**Input Sequence:** \> "To be or not to"

**Prediction:** \> "be"

## 📦 Dependencies

  * Python 3.x
  * TensorFlow
  * Streamlit
  * NumPy
  * Pandas
  * Scikit-learn
  * NLTK
  * Matplotlib
  * TensorBoard


