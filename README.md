# Sign Language Recognition

This project is designed to recognize American Sign Language (ASL) letters using a trained neural network model. The goal is to identify individual ASL hand signs and predict the corresponding alphabet letter.

## Objective

The main objective of this project is to create a deep learning model that can classify images of hand gestures representing letters in ASL. Given an image of a hand sign, the model predicts the corresponding letter from A to Z (excluding 'J').

## Tech Stack

- **TensorFlow**: A deep learning framework used to train the model and perform inference.
- **Keras**: High-level neural networks API, running on top of TensorFlow, used for building and training the model.
- **NumPy**: A library for numerical computing in Python, used for handling image data and array operations.
- **Pandas**: Although not directly used in this code, it is a common dependency for data manipulation tasks.

## Installation

To run this project, ensure that you have the following Python libraries installed:

```bash
pip install tensorflow numpy pandas
```

## Usage

1. **Load Pretrained Model**: 
   The model used for classification is pre-trained and stored in a file called `SignClass.keras`. The model is loaded using Keras' `load_model` function.

2. **Image Input**:
   An image of a hand sign is loaded using TensorFlow's `load_img` method. The image is then processed to match the input size expected by the model (28x28 pixels) and normalized for better performance.

3. **Model Prediction**:
   The processed image is passed to the model for classification. The model predicts the class (letter) of the hand sign by computing the probabilities for each class, and the class with the highest probability is chosen as the predicted letter.

4. **Output**:
   The predicted letter is printed as output.

## Code Walkthrough

### Step 1: Import Required Libraries

```python
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import keras
```

These libraries are used for image processing (`load_img`, `img_to_array`), numerical computations (`numpy`), and loading the model (`keras.models.load_model`).

### Step 2: Load the Pretrained Model

```python
load = keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/0. SignClass/0. SignClass.keras')
```

Here, the model is loaded from a `.keras` file that contains the pre-trained neural network for ASL sign classification.

### Step 3: Define Alphabet (Excluding 'J')

```python
letters = [chr(i) for i in range(ord('A'), ord('Z')+1) if i != ord('J')]  # Exclude 'J' (class 9)
```

A list of letters (from 'A' to 'Z', excluding 'J') is created to map the model's output classes to corresponding alphabet letters.

### Step 4: Preprocess the Input Image

```python
img = load_img('/content/2.png', color_mode='grayscale', target_size=(28, 28))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255
```

The image is loaded in grayscale mode, resized to `28x28` pixels, and converted to a NumPy array. The array is expanded to match the input shape expected by the model, and pixel values are normalized by dividing by 255.

### Step 5: Make Predictions

```python
predicted_class_index = np.argmax(load.predict(img_array), axis=1)[0]
```

The image is passed through the model for prediction. The `np.argmax` function is used to obtain the index of the class with the highest probability.

### Step 6: Output the Predicted Letter

```python
predicted_letter = letters[predicted_class_index]
print(f"Predicted letter: {predicted_letter}")
```

The predicted class index is mapped to the corresponding letter in the `letters` list, and the result is printed.

## Example

For an input image `/content/2.png`, the output could be:

```
Predicted letter: A
```

This means that the hand gesture in the image was recognized as the letter 'A' in ASL.

## Notes

- The model excludes 'J' due to its complexity in hand sign recognition, which might require a more advanced technique to handle effectively.
- The input image should be preprocessed to match the expected format (grayscale and resized to 28x28 pixels).
- Make sure to provide the correct path to the model file and input image.

## Future Improvements

- Expand the model to include numbers and more complex hand gestures.
- Improve preprocessing to handle variations in lighting, hand positioning, and background noise.
- Implement a real-time recognition system using webcam input.

## Acknowledgments
[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/code?datasetId=3258&sortBy=voteCount)

[CNN using Keras(100% Accuracy)](https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy)

