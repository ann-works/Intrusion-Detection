# Real-Time Network Intrusion Detection System
A real-time network intrusion detection system to classify and predict malicious traffic from time-series log data.
##  Project Overview

This project implements a real-time network intrusion detection system using a sequential deep learning model. The primary goal is to not only classify traffic as normal or malicious but also to distinguish between different types of attacks, such as **BotAttacks** and **PortScans**. The project leverages a Bidirectional LSTM to analyze sequences of network logs, allowing it to capture temporal patterns that are indicative of an ongoing attack.

The final model is deployed in an interactive **Streamlit dashboard** that simulates a live network feed, providing real-time alerts and classifications.

##  Key Features

- **In-Depth EDA:** Comprehensive exploratory data analysis to understand the characteristics of different traffic types.
- **Sequential Modeling:** A Bidirectional LSTM model trained on sequences of network logs to learn temporal attack patterns.
- **Class Imbalance Handling:** Uses `class_weight` to train a robust model on a highly imbalanced dataset.
- **Real-Time Simulation:** An interactive Streamlit dashboard to simulate live log monitoring and visualize the model's predictions as they happen.
- **Dynamic Controls:** The dashboard allows users to adjust the ratio of intrusion events in the simulation for effective demonstration.

##  Technologies Used

- **Python:** Core programming language.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **Seaborn & Matplotlib:** For data visualization during EDA.
- **Scikit-learn:** For data preprocessing and model evaluation.
- **TensorFlow & Keras:** For building and training the Bidirectional LSTM model.
- **Streamlit:** For creating the interactive web-based dashboard.
- **Joblib:** For saving and loading the trained preprocessor and encoder.
