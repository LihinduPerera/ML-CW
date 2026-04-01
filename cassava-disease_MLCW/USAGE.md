

CASSAVA LEAF DISEASE CLASSIFIER
===================================

A machine learning system for classifying cassava leaf diseases from images.

SUPPORTED DISEASE CLASSES:
----------------------------
1. cbb   - Cassava Bacterial Blight
2. cbsd  - Cassava Brown Streak Disease
3. cgm   - Cassava Green Mite
4. cmd   - Cassava Mosaic Disease
5. healthy - Healthy Cassava Leaf

METHODS OF USE:
-----------------

1. PYTHON API (In Notebook):
   --------------------------
   result = cassava_classifier.predict('path/to/image.jpg')
   print(result['class_full'])
   print(result['confidence'])

2. COMMAND LINE:
   --------------
   python predict.py image.jpg
   python predict.py image.jpg -o result.json

3. INTERACTIVE MODE:
   ------------------
   Run the Jupyter notebook and use the file upload widget,
   or call predict_from_input() for command-line interaction.

REQUIRED FILES:
-----------------
- outputs/feature_scaler.pkl
- outputs/pca_model.pkl
- outputs/random_forest_optimized.pkl (or outputs/random_forest_model.pkl)

OUTPUT FORMAT:
----------------
{
    "class": "cmd",
    "class_full": "Cassava Mosaic Disease",
    "confidence": 0.92,
    "probabilities": {
        "cbb": 0.02,
        "cbsd": 0.03,
        "cgm": 0.01,
        "cmd": 0.92,
        "healthy": 0.02
    }
}

REQUIREMENTS:
----------------
- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn 1.0+
- OpenCV
- NumPy

PERFORMANCE:
---------------
- Inference time: ~0.5-1.0 seconds per image (with GPU)
- Accuracy: See model evaluation results
- Memory: ~2GB (with VGG16 model loaded)

