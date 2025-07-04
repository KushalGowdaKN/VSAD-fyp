**🎥 Video Anomaly Detection System**\
An efficient, lightweight deep learning-based system for detecting anomalies in surveillance video data using ResNet-18 and LSTM. The project addresses the challenge of accurately identifying abnormal events across large and diverse video datasets, enabling automated monitoring and effective alert systems for real-world security applications.

**📌 Problem Statement**\
Despite significant progress in traditional and deep learning methods, there is still no definitive solution for video surveillance anomaly detection systems that can effectively handle large datasets containing various types of anomalies and their variations. In addition to limitations in existing detection approaches, system administrators often lack proper software tools to monitor and review flagged anomalies. This project aims to develop an efficient, lightweight automated system capable of detecting anomalies in video data and providing essential tools for effective monitoring and analysis.

**🎯 Objectives**\
✅ Build a lightweight and effective deep learning model for analyzing pre-recorded video data and identifying abnormal events.

✅ Train the model on a balanced dataset covering diverse normal and abnormal behaviors for accurate anomaly classification.

✅ Utilize ResNet-18 for spatial feature extraction and integrate temporal modeling using LSTM for detecting activity-based anomalies.

✅ Ensure efficient processing of large video datasets without relying on high-end hardware or constant human supervision.

✅ Design a user-friendly alert interface for timely review, interpretation, and response to flagged anomalies.

**🧠 Model Architecture**
* ResNet-18: Used for extracting spatial features from individual video frames.

* LSTM (Long Short-Term Memory): Applied for temporal sequence modeling to identify activity-based anomalies over time.

* The model is trained using Kaggle Notebooks to leverage GPU acceleration.

**📂 Dataset**

* A large-scale real-world surveillance dataset consisting of over 1900 videos across 14 different classes of activities, including both normal and anomalous events.

* In this project, we focused on 4 specific anomaly classes:
  - Fighting
  -  Arson
  -  Stealing
  -  Explosion.

* These classes were selected to train the model for detecting high-risk and visually distinguishable anomalous behaviors.

**⚙️ Tools & Technologies**
* Python, PyTorch

* ResNet-18, LSTM

* OpenCV (for frame extraction)

* Kaggle Notebook (for model training with GPU support)

**📊 Future Work**
* Real-time anomaly detection and alerting

* Enhanced UI dashboard for flagged anomaly review

* Optimization for edge deployment (e.g., Jetson Nano, Raspberry Pi)

**🧾 License**
This project is for academic and research purposes.
