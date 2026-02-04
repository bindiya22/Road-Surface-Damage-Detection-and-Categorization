Road Surface Damage Detection and Categorization using Hybrid Deep Learning Techniques

This project aims to detect various different corruptions that we can identify on the surface of roads such as transverse, longitudinal, alligator, potholes
and other corruptions. All three models were extensively trained on the dataset with tuned parameters. This design helps achieve faster inference compared to
standalone transformers or CNN models, which makes it suitable for real-time monitoring and inspection of road surfaces. The proposed model performs reliably
with a precision of 0.86 and a recall of 0.73. Evaluation using F1-score resulted in a value of 0.75 while the ensemble accuracy yielded a value of 80%, which
indicates that the proposed model possesses a strong detection and categorization capability.

Models used - YOLOv12, MaxViT, Swin |
Dataset used - https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022 |
Hardware used in training- Transformer models: Kaggle and Google Collab T4 GPU & YOLO: NVIDIA GPU
