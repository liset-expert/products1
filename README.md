# Product Categorization using BERT-based Text Classification

Welcome to the Product Categorization Project, where we've harnessed the power of advanced natural language processing techniques to develop a multi-label classification model. This model is based on Logistic Regression, coupled with TF-IDF vectorization, and has undergone rigorous training and refinement.

In this project, we've meticulously fine-tuned our model through an intensive grid search for optimal hyperparameters. The result is a model that not only predicts labels for the test set but also goes the extra mile by calculating metrics like Average Precision and F1 Score for each category. This precision-focused approach ensures that our predictions are not only accurate but also comprehensive.

By training a multi-label classification model using Logistic Regression with TF-IDF vectorization, we've created a powerful tool for categorizing products effectively. With this system in place, you can confidently categorize products with a high degree of accuracy, backed by thorough evaluation and fine-tuning.

## Table of Contents

# Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Data](#data)
4. [Usage](#usage)
    - [Application Usage via Jupyter Notebook](#application-usage-via-jupyter-notebook)
    - [Docker Setup and Usage](#docker-setup-and-usage)
5. [Folder Structure and Explanation](#folder-structure-and-explanation)
6. [Model](#model)
    - [Multiple Training Models in the Project](#multiple-training-models-in-the-project)
    - [Model Selection: TF-IDF with Logistic Regression](#model-selection-tf-idf-with-logistic-regression)
7. [Results](#results)
    - [Project Reflection: Gaining Insightful Perspectives](#project-reflection-gaining-insightful-perspectives)


## Requirements

To effectively use the application, you should meet the following requirements:

1. **Python Environment**:
   - Ensure that you have Python installed on your system. The application and its models are built using Python.

2. **Docker (Optional)**:
   - If you opt to run the application using Docker, Docker should be installed on your system. Docker simplifies the deployment process and creates an isolated environment.

3. **Internet Browser**:
   - An internet browser is essential to interact with the application's user interface and view the predicted results.

4. **Project Files**:
   - You should have access to all files and directories as described in the project's folder structure, particularly those within the `api` folder.

5. **Data Files**:
   - Ensure the necessary data files, like `products.json` and `categories.json`, are available within the `datasets` folder.

6. **Libraries and Dependencies**:
   - The application relies on specific Python packages and libraries. You can install them by executing the following command:
     ```bash
     $ pip install -r api/requirements.txt
     ```
   - If you're using Docker, these dependencies will be automatically installed when you run `$ docker-compose up --build -d`.

7. **Trained Model Files**:
   - For loading and utilizing the prediction models, you should have the corresponding pickle model files, such as `model_tfidf.pkl`, `X_train_vectorized_tfidf.pkl`, etc., inside the `pickle` folder.

8. **Web Interface Access**:
   - Access the web interface of the application. If running locally, you can generally find it at `http://localhost:5000`.

9. **Jupyter Notebook (Optional)**:
   - If you're interested in exploring the data analysis and model training process, Jupyter Notebook is used. Make sure you have Jupyter installed and open the `EDA_End_Proy.ipynb` file.

If you choose to deploy the application with Docker, it significantly streamlines the setup process. Upon executing the command `$ docker-compose up --build -d` in the terminal, Docker will not only initiate the application in containers but also automatically handle the installation of required dependencies listed in the `requirements.txt` file. This means that, when using Docker, you won't need to manually install dependencies in your local environment.

Leveraging Docker ensures consistent, isolated execution of the application and eliminates the need for manual configuration steps. This approach encapsulates the application, its dependencies, and the environment, leading to reproducible and reliable outcomes.

## Folder Structure and Explanation

**Folder Structure and Explanation**

Here's the folder structure with explanations for each folder and key files:

```plaintext
project_folder/
├── api/
|   ├── datasets/
|   |   ├── products.json                # Dataset containing product information
|   |   ├── categories.json              # Dataset containing category information
|   |   └── datos_limpios_tfidf.csv      # Cleaned dataset for TF-IDF vectorization
|   ├── model/
|   |   ├── model_bert.py                # BERT model implementation
|   |   ├── model_tfidf.py               # TF-IDF Logistic Regression model implementation
|   |   ├── model_countVectorizer.py     # Count Vectorization Logistic Regression model implementation
|   |   └── preprocess.py                # Preprocessing utilities
|   ├── pickle/
|   |   ├── model_tfidf.pkl              # Pickled TF-IDF Logistic Regression model
|   |   ├── X_train_vectorized_tfidf.pkl # Pickled TF-IDF vectorized training data
|   |   ├── vectorizer_tfidf.pkl         # Pickled TF-IDF vectorizer
|   |   └── mlb_tfidf.pkl                # Pickled MultiLabelBinarizer for TF-IDF
|   ├── templates/
|   |   └── index.html                   # HTML template for the web interface
|   ├── app.py                           # Main Flask application file
|   ├── Dockerfile                       # Dockerfile for building the application
|   └── requirements.txt                 # List of required Python packages
├── EDA_End_Proy.ipynb                    # Jupyter Notebook for exploratory data analysis
├── README.md                             # Project README file
└── docker-compose.yml                    # Docker Compose configuration file
```

**Explanation of Folders and Files:**

- `api/`: Main application directory.
  - `datasets/`: Contains input datasets and cleaned data for training and predictions.
  - `model/`: Contains model implementations and preprocessing utilities.
  - `pickle/`: Stores pickled model and vectorization artifacts.
  - `templates/`: Holds the HTML template for the web interface.
  - `app.py`: Main Flask application file.
  - `Dockerfile`: Instructions for building the Docker image.
  - `requirements.txt`: List of Python packages required by the application.
  
- `EDA_End_Proy.ipynb`: Jupyter Notebook for exploratory data analysis.
- `README.md`: Project's main README file.
- `docker-compose.yml`: Configuration file for Docker Compose.

This organized folder structure separates different components of our project, making it clear where different types of files are located. It helps maintain clarity and order as you work on different aspects of your project, from data analysis to model development and deployment.

## Dataset

Our project starts with two initial JSON files: 'products.json' and 'categories.json', which provide the foundational data. Following a comprehensive Exploratory Data Analysis (EDA) phase, we merge the 'Name' and 'Description' columns into a consolidated 'Information' column. This combined with the 'extracted_categories' column, which holds the pre-extracted categories in a model-friendly format, constitutes our preprocessed dataset.

Once the merging is complete, we proceed to cleanse the text data by eliminating unnecessary elements. This effort results in the creation of a new, refined dataset that is now ready for training purposes. This dataset encapsulates the core product information and their associated categories in a clean and structured format suitable for model training.

## Usage

This document provides a step-by-step guide on how to use the application and how to set it up using Docker. The project consists of two parts: a usage guide through a Jupyter Notebook and another through the implementation of Docker containers.

This document provides a step-by-step guide on how to use the application and how to set it up using Docker. The project consists of two parts: a usage guide through a Jupyter Notebook and another through the implementation of Docker containers.

### Application Usage via Jupyter Notebook

1. Open the `EDA_End_Proy.ipynb` file in a Jupyter Notebook environment.
2. Follow the detailed steps in the notebook to complete the process of data loading, analysis, text processing, and model training.
3. Within the notebook, three examples of trainings are provided with functioning tests for each of them. This will allow you to understand how the model would classify new products.

### Docker Setup and Usage

Make sure you have Docker installed on your system before proceeding.

1. Open a WSL terminal or an IDE like Visual Studio Code.
2. From the project folder, run the following command to build and launch the Docker containers:

   ```bash
   $ docker-compose up --build -d
   ```

   This command creates and runs two containers: one for the web application and another for the Redis database.

#### Description of the `docker-compose.yml` file

- The `web` service is based on the Dockerfile located in the `./api` folder. It's exposed on port `5000` and depends on the Redis service.
- The `redis` service uses the official Redis image and is exposed on port `6379`. It also uses a volume named `redis_data` to store data persistently.
- The `app-network` network connects both services so they can communicate.

### Accessing the Application

Once the containers are up and running, you can access the web application in your browser through the address `http://localhost:5000`. From here, you can interact with the application and test its functionality.
The results of the model training and evaluation are saved in pickle files, and these files serve as the source for loading predictions within the interface.

**User Interface Guide: Making Predictions on Product Categories**

The user interface of the application allows users to input a product name and description. Upon clicking the "Predict" button, the application will provide predictions for one or multiple categories that the given product might belong to. Additionally, a "Clear Fields" button is available to reset the input fields and perform a new prediction.

Here's how to use the interface:

1. **Product Name and Description Input**:
   - Locate the fields where you can enter the product's name and description.
   - In the "Product Name" field, type a descriptive name for the product.
   - In the "Product Description" field, provide a detailed description of the product.

2. **Making Predictions**:
   - Once you've entered the product information, click the "Predict" button.
   - The application will process the input and generate predictions based on the trained models.
   - The predicted categories will be displayed below, indicating the categories that the product could potentially belong to. These categories could be one or several, depending on the input and the model's predictions.

3. **Clearing Fields and Making New Predictions**:
   - After viewing the predictions, you can click the "Clear Fields" button.
   - This action will reset the input fields, allowing you to enter new information for a fresh prediction.

**Summary of Steps**:
1. Enter the product's name and description.
2. Click the "Predict" button to receive category predictions.
3. Review the predicted categories.
4. Click the "Clear Fields" button to reset and enter new information for another prediction.

This user-friendly interface allows you to quickly input product information, obtain predictions, and explore multiple potential categories that your product could fall into. It also enables you to reset the interface for new predictions without the need to reload the page or restart the application.

## Model

**Multiple Training Models in the Project**

In the course of this project, various training models were developed to achieve the goal of multi-label classification. These models were designed to predict product categories based on their attributes. Here's an overview of the different models and their methodologies:

1. **Logistic Regression with TF-IDF Vectorization**:
   - This model employs Logistic Regression, a powerful classification algorithm, in combination with TF-IDF vectorization to predict multiple categories for a given product.
   - A grid search technique is applied for hyperparameter tuning to optimize the model's performance.
   - Predictions are made on the test set, and metrics such as Average Precision and F1 Score are calculated for each class to assess the model's accuracy.

2. **Logistic Regression with Count Vectorization**:
   - In this model, Logistic Regression is used with Count Vectorization, an alternative method for converting text data into numerical vectors.
   - Predictions are generated for the test set, and the model's performance is evaluated using metrics like Average Precision and F1 Score per class.

3. **BERT for Sequence Classification**:
   - BERT, a state-of-the-art transformer-based model, is employed for sequence classification tasks.
   - The model is trained to predict product categories based on their descriptions.
   - Predictions on the test set are used to calculate metrics such as accuracy, precision, recall, and F1-score.

All three models can be found within the "model" folder of the project directory. Each model represents a different approach to solving the multi-label classification problem, leveraging distinct techniques for text processing, feature extraction, and classification. The various models were developed to provide a comprehensive analysis of different strategies and their performance in predicting product categories accurately.

These trained models offer a range of options for making predictions, catering to different data characteristics and use cases. Users of the application can choose the model that best aligns with their needs and experiment with various prediction methodologies.

**Model Selection: TF-IDF with Logistic Regression**

After thorough evaluation of the multiple training models, the one based on TF-IDF with Logistic Regression was chosen due to its superior performance and results. This model demonstrated its efficacy in predicting product categories accurately and efficiently. Here are the reasons for selecting this particular model:

1. **High Accuracy and Precision**:
   - The TF-IDF model with Logistic Regression achieved high accuracy and precision in predicting product categories based on their attributes.
   - The grid search for hyperparameter tuning further optimized the model's performance, resulting in improved accuracy and precision scores.

2. **Effective Text Representation**:
   - The TF-IDF vectorization method proved to be effective in representing text data, capturing important features within product descriptions.
   - This method enhanced the model's ability to discern relevant information for accurate predictions.

3. **Interpretable Results**:
   - Logistic Regression models are known for their interpretability, providing insight into the factors influencing predictions.
   - This transparency is valuable for understanding how the model arrives at its category predictions.

4. **Fast Inference**:
   - Once trained, the TF-IDF Logistic Regression model offers fast and efficient predictions.
   - The lightweight nature of Logistic Regression makes it suitable for real-time or on-the-fly prediction scenarios.

5. **Consistent Metrics**:
   - The model's evaluation metrics, such as Average Precision and F1 Score per class, consistently demonstrated its robustness across different categories.

Based on the compelling combination of accuracy, precision, interpretability, and efficiency, the TF-IDF Logistic Regression model emerged as the most suitable choice for predicting product categories. Its dependable performance and strong results make it an ideal candidate for deployment in the application's interface, providing users with reliable and accurate category predictions for their products.

## Results

**Project Reflection: Gaining Insightful Perspectives**

Our expedition across the captivating landscape of product categorization has been a profound journey, revealing insights that extend far beyond the boundaries of technology. As we contemplate the culmination of our endeavors, several illuminating lessons have emerged, illuminating the path to a future ripe with potential:

**Empowering Pre-Processing:** Throughout our exploration, we've come to fully grasp the potent influence of text pre-processing. This pivotal step in refining textual data has unveiled its transformative capacity, elevating the quality of our input data and, in turn, bolstering the predictive prowess of our model.

**Harmonious Flask-Redis-Docker Fusion:** The orchestration of Flask, Redis, and Docker has transcended mere technical integration, giving rise to a harmonious ecosystem characterized by accessibility, speed, and scalability. This orchestration has not only expedited seamless interactions, but has also laid the cornerstone for future expansions and innovations.

**Guided by Informed Choices:** The culmination of our classification efforts has underscored the undeniable power of metrics – the compass that steers us through the realm of informed decision making. Metrics such as accuracy, precision, recall, and F1 score have illuminated our path, empowering us to navigate product categorization with unwavering precision.

Our journey is a testament to the multifaceted insights gained along the way, emphasizing the intrinsic fusion of technology and wisdom. As we stand at the precipice of tomorrow, armed with newfound understanding, we look forward to forging ahead, translating these revelations into future strides that transcend the boundaries of innovation.
