
🚨 #*Anomaly Detection in Sensor Data*🚨
Welcome to the Anomaly on Sensor Data project! This deep learning-based approach leverages unsupervised anomaly detection for time-series sensor data. By utilizing a stacked LSTM (Long Short-Term Memory) network, the model learns normal patterns and identifies deviations in sensor data. This project is ideal for real-world scenarios where anomalies are rare or undefined in advance. No labeled data required – it’s as intuitive as it gets! 🤖

📋 Overview
In this project, we use a deep learning model to predict sensor signals and detect anomalies based on prediction errors. The system is composed of two major parts:

LSTM Model: Trains on synthetically generated sensor signals to learn normal temporal patterns.

DBSCAN Clustering: Used for anomaly detection by clustering prediction errors.

By combining these two techniques, we can robustly detect both subtle and abrupt anomalies in real-time sensor data. 🚀

⚙️ Key Features
Unsupervised anomaly detection — no labeled data required 🏷️

Utilizes LSTM networks to learn time-series patterns ⏳

Anomaly detection based on prediction errors 📉

DBSCAN clustering to identify anomalous regions 🔍

Visualization of actual vs predicted data and error analysis 📊

🏗️ Installation
Clone the repository and install dependencies using:

bash
Copy
Edit
git clone https://github.com/majumdarjoyeeta/Anomaly-on-Sensor-Data.git
cd Anomaly-on-Sensor-Data
pip install -r requirements.txt
Make sure you have the necessary libraries for deep learning and visualization, such as TensorFlow and Matplotlib. 🔥

🛠️ How It Works
1. Generate Synthetic Data 🖥️
First, synthetic sensor data is generated to simulate real-world signals. The data is then split into training and testing sets. 🌊

2. Build and Train the LSTM Model 🧠
We create a stacked LSTM model that learns the temporal patterns from the training data. The model is trained over multiple epochs to minimize the loss function.

python
Copy
Edit
model = build_lstm_model()
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
3. Make Predictions 🧐
After training, the model predicts the test data, and prediction errors are computed:

python
Copy
Edit
predictions = model.predict(X_test).flatten()
mse = (predictions - y_test) ** 2
4. Detect Anomalies 🚨
Prediction errors are visualized, and DBSCAN clustering helps us identify anomalous regions in the data. These regions indicate where the sensor data deviates from the learned patterns.

5. Visualization 📸
The results are visualized in three plots:

Actual Signal with Anomalies

Predicted Signal

Squared Prediction Error (for anomaly detection)

python
Copy
Edit
plt.subplot(311)
plt.plot(y_test, label="Actual")
plt.title("Actual Signal with Anomalies")
plt.legend()

plt.subplot(312)
plt.plot(predictions, label="Predicted", color='green')
plt.title("Predicted Signal")
plt.legend()

plt.subplot(313)
plt.plot(mse, label="Squared Error", color='red')
plt.title("Prediction Error")
plt.legend()
📊 Results
Here's a quick overview of the training process:

Epochs: 5

Training Time: ~218.79 seconds 🕒

Loss: Decreases steadily, indicating learning from data 🎯

📌 Usage
Run the notebook to train the model, make predictions, and visualize the results! 🖥️

bash
Copy
Edit
jupyter notebook Anomaly-on-Sensor-Data.ipynb
🧑‍💻 Contributing
We welcome contributions! If you find a bug or want to add more features, feel free to fork the repository and submit a pull request. 🌟

📑 License
This project is licensed under the MIT License - see the LICENSE file for details. 📄

🙌 Acknowledgments
A big thanks to all the amazing open-source tools and libraries that made this possible:

Keras & TensorFlow for the deep learning magic ✨

Matplotlib for the stunning visualizations 📊

DBSCAN for clustering the anomalies 💡

🔗 Links
GitHub Repository

Jupyter Notebook

Let's detect anomalies together and make sensor data safer! 🚀📉
