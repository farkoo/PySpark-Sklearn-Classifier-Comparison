# SPARK

This repository contains two Python scripts, `ml_with_pyspark.py` and `ml_with_sklearn.py`, that demonstrate image feature vector classification using different machine learning libraries. The `ml_with_pyspark.py` script utilizes PySpark to perform classification tasks and includes implementations of various classification algorithms such as Random Forest, Gradient-Boosted Tree, and Logistic Regression. On the other hand, the `ml_with_sklearn.py` script utilizes scikit-learn and showcases the same classification tasks with algorithms like Random Forest, Decision Tree, and Support Vector Machine. These scripts serve as examples for performing image feature vector classification and can be customized and expanded upon to suit specific project requirements.

## Installation

To use the `ml_with_pyspark.py` script and work with PySpark, you'll need to install a few dependencies and configure Java. Follow the steps below to set up your environment:

### 1. Java Installation

PySpark requires Java Development Kit (JDK) to be installed on your system. Make sure you have Java installed by following these instructions:

- Visit the Oracle Java SE Development Kit downloads page: [Java SE Downloads](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html)
- Select the appropriate JDK version for your operating system and download the installer.
- Run the installer and follow the installation instructions specific to your operating system.

### 2. PySpark Installation

#### PySpark

PySpark can be installed using the `pip` package manager. Open your terminal or command prompt and execute the following command:

```bash
pip install pyspark
```

This will install PySpark and its dependencies.

### 3. Configuration

Once you have Java, Python, and PySpark installed, you'll need to configure the Java environment variable for PySpark to locate the Java installation. 

#### Windows

- Right-click on "This PC" or "My Computer" and select "Properties."
- Click on "Advanced system settings."
- Click on the "Environment Variables" button.
- In the "System Variables" section, click on "New."
- Enter `JAVA_HOME` as the variable name.
- Enter the path to your Java installation directory as the variable value (e.g., `C:\Program Files\Java\jdk1.8.0_281`).
- Click "OK" to save the changes.

#### macOS and Linux

- Open a terminal.
- Run the following command to open the environment variables file:
  ```bash
  nano ~/.bash_profile
  ```
- Add the following line at the end of the file, replacing `/path/to/java` with the path to your Java installation directory:
  ```bash
  export JAVA_HOME=/path/to/java
  ```
- Press `Ctrl + X`, then `Y` to save the changes and exit.

#### Verification

To verify that PySpark is installed correctly, open a terminal or command prompt and run the following command:

```bash
pyspark
```

This should start the PySpark shell, indicating that the installation was successful.

Now you're ready to use the `ml_with_pyspark.py` script and leverage the power of PySpark for image feature vector classification.

If you encounter any issues during the installation process or have further questions, please refer to the official documentation or seek community support.

Remember to update the paths to your actual Java installation directory if they differ from the examples provided.

### 4. scikit-learn Installation

scikit-learn can be installed using the `pip` package manager, which is bundled with Python. Open your terminal or command prompt and execute the following command:

```bash
pip install scikit-learn
```

This will install scikit-learn and its dependencies.

#### Verification

To verify that scikit-learn is installed correctly, open a terminal or command prompt and run the following command:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

This should print the version number of scikit-learn installed on your system, indicating that the installation was successful.

Now you're ready to use the `ml_with_sklearn.py` script and leverage the functionalities of scikit-learn for image feature vector classification.

If you encounter any issues during the installation process or have further questions, please refer to the official scikit-learn documentation or seek community support.

That's it! You've successfully installed scikit-learn. Happy coding!

## **Project Description**

This repository contains two Python scripts, `ml_with_pyspark.py` and `ml_with_sklearn.py`, which perform image feature vector classification using different machine learning libraries.

### **ml_with_pyspark.py:**

The `ml_with_pyspark.py` script uses PySpark to perform image feature vector classification. It utilizes various functionalities from the PySpark library, including data manipulation, feature vector assembly, and classification algorithms. Here are the key steps performed in the script:

1. Initializing SparkSession: The script starts by initializing a SparkSession, which is the entry point for working with Spark functionality.

2. Reading the training data: The script reads the training data from CSV files, consisting of image feature vectors and corresponding labels. It uses Spark's DataFrame API to read the CSV files and assign explicit aliases to the columns.

3. Joining the features and labels: The script joins the features and labels on a common index column to create a unified training dataset.

4. Reading the test data: Similar to the training data, the script reads the test data from CSV files and joins the features and labels to create a unified test dataset.

5. Feature vector preparation: The script prepares the feature vector column by assembling the feature columns from the training and test datasets using the VectorAssembler.

6. Classification using different algorithms: The script applies several classification algorithms available in PySpark, including Random Forest, Gradient-Boosted Tree, Decision Tree, Support Vector Machine, Multilayer Perceptron, Logistic Regression, and Naive Bayes. For each algorithm, it builds a pipeline that incorporates the feature vector assembly and the specific classifier. The trained models are then used to make predictions on the test data.

7. Model evaluation: The script evaluates the performance of each classification algorithm by computing the accuracy metric using a MulticlassClassificationEvaluator.

8. Elapsed time measurement: The script measures the elapsed time for each model, including both data preparation and model training, providing insights into the time complexity of the different algorithms.

### **ml_with_sklearn.py:**

The `ml_with_sklearn.py` script performs image feature vector classification using the scikit-learn library, a popular machine learning toolkit in Python. It employs various classifiers from scikit-learn to train and evaluate the models. Here's an overview of the steps performed in the script:

1. Importing necessary modules: The script imports the required modules from scikit-learn, including feature vector assembly, classification algorithms, and evaluation metrics.

2. Reading the training and test data: The script reads the training and test data from CSV files. The files contain image feature vectors and corresponding labels.

3. Feature vector preparation: The script assembles the feature columns from the training and test datasets using the VectorAssembler from scikit-learn.

4. Classification using different algorithms: The script applies several classification algorithms available in scikit-learn, including Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, Support Vector Machine, Multilayer Perceptron, and Naive Bayes. For each algorithm, it trains a classifier on the training data and makes predictions on the test data.

5. Model evaluation: The script evaluates the performance of each classification algorithm by computing the accuracy metric using the accuracy_score function from scikit-learn.

6. Elapsed time measurement: The script does not include elapsed time measurement.

These scripts provide examples of how to perform image feature vector classification using different machine learning libraries. Users can choose the one that best suits their requirements and adapt the code as needed.

Please note that the paths to the input CSV files in the script need to be adjusted to match the actual file locations on your system.

Feel free to customize the scripts further or explore additional functionalities to enhance the image classification process using PySpark or scikit

-learn.

## Usage

### `ml_with_pyspark.py`

To use the `ml_with_pyspark.py` script for image feature vector classification with PySpark, follow the steps below:

1. Ensure you have installed Java, Python, and PySpark as described in the "Installation" section of this README.

2. Open a terminal or command prompt and navigate to the directory where `ml_with_pyspark.py` is located.

3. Modify the paths to your input CSV files in the script. Look for the following lines of code:

   ```python
   train_features_file = "path/to/train_features.csv"
   train_labels_file = "path/to/train_labels.csv"
   test_features_file = "path/to/test_features.csv"
   test_labels_file = "path/to/test_labels.csv"
   ```

   Replace `"path/to/train_features.csv"`, `"path/to/train_labels.csv"`, `"path/to/test_features.csv"`, and `"path/to/test_labels.csv"` with the actual paths to your respective CSV files containing the image feature vectors and labels.

4. Run the script by executing the following command in the terminal or command prompt:

   ```bash
   python ml_with_pyspark.py
   ```

   The script will execute and perform the image feature vector classification using various classification algorithms available in PySpark.

5. Once the script finishes running, it will display the accuracy of each classification algorithm and the elapsed time for each model.

### `ml_with_sklearn.py`

To use the `ml_with_sklearn.py` script for image feature vector classification with scikit-learn, follow the steps below:

1. Ensure you have installed Python and scikit-learn as described in the "Installation" section of this README.

2. Open a terminal or command prompt and navigate to the directory where `ml_with_sklearn.py` is located.

3. Modify the paths to your input CSV files in the script. Look for the following lines of code:

   ```python
   train_features_file = "path/to/train_features.csv"
   train_labels_file = "path/to/train_labels.csv"
   test_features_file = "path/to/test_features.csv"
   test_labels_file = "path/to/test_labels.csv"
   ```

   Replace `"path/to/train_features.csv"`, `"path/to/train_labels.csv"`, `"path/to/test_features.csv"`, and `"path/to/test_labels.csv"` with the actual paths to your respective CSV files containing the image feature vectors and labels.

4. Run the script by executing the following command in the terminal or command prompt:

   ```bash
   python ml_with_sklearn.py
   ```

   The script will execute and perform the image feature vector classification using various classification algorithms available in scikit-learn.

5. Once the script finishes running, it will display the accuracy of each classification algorithm.

Please note that you may need to adjust other parts of the code, such as feature column names or the number and configuration of hidden layers in the Multilayer Perceptron Classifier, according to your specific requirements.

Feel free to explore and modify the code to suit your needs and experiment with different algorithms or techniques for image feature vector classification.

That's it! You can now use the `ml_with_pyspark.py` and `ml_with_sklearn.py` scripts to perform image feature vector classification. Enjoy!

## Support

**Contact me @:**

e-mail:

* farzanehkoohestani2000@gmail.com

Telegram id:

* [@farzaneh_koohestani](https://t.me/farzaneh_koohestani)

## License
[MIT](https://github.com/farkoo/PySpark-Sklearn-Classifier-Comparison/blob/master/LICENSE)
&#0169; 
[Farzaneh Koohestani](https://github.com/farkoo)

