# Customer Segmentation Using Agglomerative Clustering
**📅 Date:** Jul 17, 2024

Customer segmentation plays a pivotal role in marketing and business analytics. It allows businesses to categorize customers into specific groups based on shared characteristics, enabling personalized marketing, better customer retention, and more effective product recommendations. In this post, we will walk through how to perform customer segmentation using Principal Component Analysis (PCA) for dimensionality reduction and clustering techniques like KMeans and Agglomerative Clustering.

## Introduction
Understanding customers is key to effective business strategies. By segmenting customers based on demographics and behavior, businesses can tailor their marketing efforts, drive engagement, and increase sales. In this project, we’ll explore how to implement customer segmentation in Python using PCA and clustering methods.

## Project Setup
To follow along, start by cloning the project repository:

```bash
git clone <https://github.com/sairam-penjarla/Customer-segmentation.git>
cd Customer-segmentation
```

### Project Structure
The project is organized into a few essential files:

- **preprocessing.py**: Contains classes responsible for data preprocessing, including feature engineering and scaling.
- **model_training.py**: Includes the clustering model training and visualization logic for KMeans and Agglomerative Clustering.
- **main.py**: The main script that ties everything together – from data preprocessing to model training and visualization.

## Step-by-Step Explanation

### 1. Data Preprocessing (preprocessing.py)
Data preprocessing ensures that the dataset is clean, scaled, and ready for analysis. The **FeatureEngineering** class performs several transformations on the raw data.

#### Feature Engineering
Here’s how the class works:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

class FeatureEngineering:
    def run(self, data):
        data["Age"] = 2021 - data["Year_Birth"]  # Calculate customer age
        data["Spent"] = data[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)  # Total spending
        data["Living_With"] = data["Marital_Status"].replace({"Married": "Partner", "Single": "Alone", "Divorced": "Alone"})  # Determine living situation
        data["Children"] = data["Kidhome"] + data["Teenhome"]  # Total number of children
        data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]  # Calculate family size
        data["Is_Parent"] = (data["Children"] > 0).astype(int)  # Binary indicator for parenthood
        data["Education"] = data["Education"].replace({"Basic": "Undergraduate", "Master": "Postgraduate"})  # Re-categorize education levels
        data = data.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat"})  # Rename columns
        data = data.drop(["Marital_Status", "Year_Birth", "ID"], axis=1)  # Drop unnecessary features
        return data
```

#### Explanation:
- **Age**: Computed based on the customer’s birth year.
- **Total Spending**: Aggregates spending across various product categories.
- **Living Situation**: Maps marital status to living arrangements.
- **Family Size & Parenthood**: Calculates family size and creates a parenthood indicator.
- **Education**: Segments education into broader categories.
- **Column Renaming & Dropping**: Simplifies and drops redundant features.

#### Dimensionality Reduction with PCA
After feature engineering, the **PreprocessingSteps** class applies PCA to reduce dimensionality:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PreprocessingSteps:
    def run(self, data):
        data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")
        data['Customer_For'] = (data['Dt_Customer'].max() - data['Dt_Customer']).dt.days  # Customer tenure
        data = self.fe.run(data)  # Feature engineering step
        data = data[(data["Age"] < 90) & (data["Income"] < 600000)]  # Remove outliers
        data[object_cols] = data[object_cols].apply(lambda col: LE.fit_transform(col))  # Encode categorical variables
        data = data.drop(columns=['AcceptedCmp3', 'AcceptedCmp4'], axis=1)  # Drop unnecessary columns
        scaled_data = StandardScaler().fit_transform(data)  # Scale data
        pca = PCA(n_components=3)  # Reduce to 3 principal components
        PCA_data = pd.DataFrame(pca.transform(scaled_data), columns=["col1", "col2", "col3"])
        return PCA_data
```

#### Explanation:
- **Customer Tenure**: Calculates how long each customer has been with the company.
- **Outlier Removal**: Filters out extreme values for age and income.
- **Scaling & PCA**: Standardizes features and applies PCA to reduce the data to three dimensions.

### 2. Model Training & Visualization (model_training.py)

The **ClusteringModel** class trains clustering models (KMeans and Agglomerative Clustering) and visualizes the results.

#### KMeans Clustering & Elbow Method
We use the **KElbowVisualizer** from Yellowbrick to determine the optimal number of clusters:

```python
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

class ClusteringModel:
    def __init__(self):
        self.PCA_Data = None

    def fit(self, PCA_data):
        self.PCA_Data = PCA_data
        visualizer = KElbowVisualizer(KMeans(), k=(1, 10))  # Visualize the elbow method
        visualizer.fit(PCA_data)
        visualizer.show()
        self.n_clusters = visualizer.elbow_value_
        return self.n_clusters
```

#### Explanation:
The **Elbow Method** helps find the optimal number of clusters by observing where the sum of squared distances begins to stabilize. This guides the selection of **n_clusters** for KMeans.

#### Agglomerative Clustering & 3D Visualization
Agglomerative Clustering is then applied, and the results are visualized in a 3D scatter plot:

```python
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ClusteringModel:
    def predict(self, PCA_data):
        AC = AgglomerativeClustering(n_clusters=self.n_clusters)
        PCA_data['Clusters'] = AC.fit_predict(PCA_data)
        self.plot_predictions_3D(PCA_data)
    
    def plot_predictions_3D(self, PCA_data):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(PCA_data["col1"], PCA_data["col2"], PCA_data["col3"], c=PCA_data["Clusters"])
        plt.show()
```

#### Explanation:
- **Agglomerative Clustering**: Groups customers based on proximity.
- **3D Visualization**: Displays the clustering results in a 3D space using the first three principal components.

### 3. Main Execution (main.py)

The **main.py** script integrates all the steps, from preprocessing to training and visualization.

```python
import pandas as pd
from preprocessing import PreprocessingSteps
from model_training import ClusteringModel

data = pd.read_csv("dataset/marketing_campaign.csv").dropna()
preprocessor = PreprocessingSteps()
PCA_data = preprocessor.run(data)

model = ClusteringModel()
model.fit(PCA_data)
model.predict(PCA_data)
```

#### Explanation:
- **Data Loading**: Loads the marketing campaign dataset.
- **Preprocessing**: Applies feature engineering and PCA transformations.
- **Model Training**: Determines the optimal number of clusters and visualizes the results.

## Conclusion
Customer segmentation enables businesses to tailor marketing strategies, improve customer retention, and optimize product recommendations. By applying PCA for dimensionality reduction and using clustering algorithms like KMeans and Agglomerative Clustering, businesses can unlock valuable insights. This project provides a step-by-step guide to implementing customer segmentation with Python. For more details, refer to the project repository on GitHub.

Happy coding!
