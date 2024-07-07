
 Crop Prediction Using AI

 Overview
 
- This project utilizes machine learning and AI techniques to predict the most suitable crops for cultivation based on various environmental and soil parameters. The goal is to aid farmers and agricultural professionals in making informed decisions to optimize crop yield and resource use.

 Features
- Predict the best crop to grow based on input parameters.
- User-friendly web interface for inputting data and viewing predictions.
- Visualizations of data trends and predictions.
- Scalable and easily deployable model.

 Requirements
- Python 3.8+
- Libraries: 
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - Flask (for the web interface)
  - Jupyter Notebook (for development and experimentation)

 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ShivaniMadhav/Crop_prediction_using_AI.git
   cd Crop_prediction_using_AI
   ```

2. Create and activate a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate    On Windows use `venv\Scripts\activate`
   ```

3. Install the required libraries:
   ```sh
   pip install numpy pandas seaborn matplotlib scikit-learn
   ```

 Dataset
- The dataset used for training the model should include features such as soil type, rainfall, temperature, pH level, etc.
- Ensure the dataset is in a CSV format and placed in the `data` directory.

 Usage
1. Data Preprocessing:
   - Preprocess the dataset using the provided 'crop_prediction.py' file for reference.
   - This step includes handling missing values, encoding categorical data, and feature scaling.

2. Training the Model:
   - Train the machine learning model using `crop_prediction.py`.
   - Experiment with different algorithms to find the best-performing model.
   - Save that model as 'model.pkl'

3. Running the Web Application:
   - Start the Flask web application to input parameters and get crop predictions.
   ```sh
   python app.py
   ```
   - Open your web browser and go to `http://127.0.0.1:5000/` to access the application.

 Project Structure
- `data/`: Directory containing the dataset.
- `training/`: Directory for data preprocessing and model training.
- `model.pkl`: Saved models and serialized objects.
- `static/` and `templates/`: Directories for the Flask web application.
- `app.py`: Flask application script.


 Contributing
- Contributions are welcome! Please fork the repository and create a pull request with your changes.

 License
- This project is licensed under the MIT License. See the `LICENSE` file for more details.

 Acknowledgements
- Inspired by various open-source agricultural prediction projects.
- Thanks to the contributors and the open-source community for their valuable inputs and datasets.

 Contact
- For any queries or suggestions, please contact `shivanimadhav0906@gmail.com`.
