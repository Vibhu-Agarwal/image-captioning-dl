# Image Captioning with Deep Learning

This project implements an image captioning model using a deep learning architecture. The model is trained to generate descriptive captions for given images. It uses an encoder-decoder framework with a pre-trained ResNet-50 for image feature extraction and a simple RNN for caption generation.

## Features

- **Encoder-Decoder Architecture**: Utilizes a CNN encoder to extract image features and an RNN decoder to generate captions.
- **Pre-trained Encoder**: Leverages a pre-trained ResNet-50 model for robust image feature extraction.
- **Simple RNN Decoder**: A straightforward RNN for generating captions word by word.
- **Flickr8k Dataset**: Trained on the popular Flickr8k dataset.
- **Checkpointing**: Saves model checkpoints during training to resume from the last saved state.
- **Inference**: Provides scripts to generate captions for new images and visualize the results.
- **Customizable**: The architecture is modular and can be extended with more advanced components like LSTMs, GRUs, or attention mechanisms.

## Dependencies

The project is built with Python and uses the following libraries:

- `torch`
- `torchvision`
- `nltk`
- `datasets`
- `matplotlib`
- `pillow`

For a detailed list of dependencies and their versions, please refer to the `pyproject.toml` file.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/image-captioning-dl.git
    cd image-captioning-dl
    ```

2.  **Install the dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    *Note: A `requirements.txt` file is not provided. You can generate one from `pyproject.toml` or install the dependencies manually.*

## Usage

### Training

To train the model, run the `main.py` script.

```bash
python main.py
```

- The script will download the Flickr8k dataset.
- It will create and cache a vocabulary from the training captions.
- Training progress, including loss, will be logged to TensorBoard in the `runs` directory.
- Model checkpoints will be saved in the `model_checkpoints` directory.

You can modify the training parameters like `NUM_EPOCHS`, `LEARNING_RATE`, etc., in the `model_run.py` file.

### Inference

The `img_captioning.ipynb` notebook provides a complete walkthrough of the project, from data loading and model building to training and inference. It's a great place to start to understand the end-to-end workflow.

You can also use the `inference.py` script to generate captions for images. The notebook demonstrates how to:

1.  Load a trained model from a checkpoint.
2.  Preprocess an image.
3.  Generate a caption.
4.  Display the image with the generated caption.

## File Descriptions

- **`main.py`**: The main script to start the training process.
- **`dataset.py`**: Defines the `Flickr8KDataset` class for loading and preprocessing the data.
- **`model.py`**: Contains the `EncoderCNN` and `DecoderRNN` model definitions.
- **`model_run.py`**: Manages the training loop, optimization, and checkpointing.
- **`language_utils.py`**: Includes functions for text tokenization and vocabulary creation.
- **`image_preprocess.py`**: Provides utilities for image transformation.
- **`inference.py`**: Contains functions for generating captions and visualizing the results.
- **`utils.py`**: Helper functions for saving/loading checkpoints and device selection.
- **`custom_types.py`**: Defines a custom abstract base class for datasets.
- **`img_captioning.ipynb`**: A Jupyter notebook with a comprehensive, step-by-step guide to the project.

## Model Architecture

The model follows an encoder-decoder architecture:

### Encoder

- **Model**: `EncoderCNN`
- **Backbone**: Pre-trained ResNet-50
- **Process**: The ResNet-50 processes an input image and extracts a feature vector from the last convolutional layer. This feature vector is then passed through a fully connected layer to reduce its dimensionality, creating a rich representation of the image's content.

### Decoder

- **Model**: `DecoderRNN`
- **Type**: Simple Recurrent Neural Network (RNN)
- **Process**: The decoder is an RNN that takes the image feature vector from the encoder as its initial hidden state. It then generates the caption word by word. At each time step, the RNN takes the previously generated word as input and produces the next word in the sequence. The process starts with a special `<start>` token and ends when an `<end>` token is generated or the maximum caption length is reached.

## Future Work

- **Implement Attention**: Incorporate an attention mechanism to allow the decoder to focus on different parts of the image at each step of caption generation.
- **Use Advanced Decoders**: Replace the simple RNN with more powerful variants like LSTM or GRU to better capture long-term dependencies in the text.
- **Experiment with Embeddings**: Try different pre-trained word embeddings (e.g., GloVe, Word2Vec) to improve language understanding.
- **Web Application**: Deploy the trained model as a web application where users can upload images and get captions.
- **Beam Search**: Implement beam search for the decoder to generate more accurate and fluent captions instead of the current greedy search approach.