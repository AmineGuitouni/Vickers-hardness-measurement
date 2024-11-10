# Corner Detection with U-Net

This project is an AI model that uses the U-Net architecture to detect corners in images. The model is trained on labeled data and is capable of identifying specific corners in an image, like those captured through a microscope or a similar setup. The repository includes scripts for training, labeling, creating an API, and visualizing labeled data.

## Project Structure

- **`main.py`**: The main script to train the U-Net model on labeled image data.
- **`config.py`**: A configuration file defining training parameters (e.g., learning rate, batch size, number of epochs).
- **`api.py`**: A script to create an API endpoint for making predictions with the trained model, allowing easy integration with applications.
- **`labeling.py`**: A utility for labeling images, helping to create and organize a dataset with ground truth labels.
- **`view_labeling.py`**: A script to visualize labeled images and verify that labels align correctly with the corners.

## Requirements

To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies:

- PyTorch
- FastAPI
- OpenCV
- uvicorn