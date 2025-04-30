# Building a Movie Recommendation System from Scratch
**📅 Date:** Jul 17, 2024

### Introduction

In the age of streaming services, recommendation systems are essential for keeping users engaged by suggesting content tailored to their preferences. In this blog post, we'll walk through the process of building a movie recommendation system from scratch using Python and machine learning libraries. We'll cover topics such as data preprocessing, transformation, model training, and generating recommendations based on cosine similarity.

I encourage you to clone the GitHub repository and try the code yourself to better grasp the concepts discussed in this post. Let’s get started!

---

### GitHub Link

You can find the complete code in the GitHub repository. Clone it and follow along to better understand how the recommendation system is built from scratch.

---

### Data Loading and Preprocessing

#### DataLoader Class

The **DataLoader** class is responsible for loading the movie dataset and performing essential preprocessing tasks. This includes renaming columns for clarity, handling missing values, and preparing the data for analysis.

```python
import pandas as pd
import logging
import yaml

class DataLoader:
    def __init__(self, filepath, columns):
        """Initialize the DataLoader with the file path and columns to rename."""
        self.filepath = filepath
        self.columns = columns
        self.df = None

    def preprocess_data(self):
        """Load and preprocess the data."""
        logging.info("Loading data from file: %s", self.filepath)
        self.df = pd.read_csv(self.filepath)
        logging.info("Preprocessing data")

        # Rename columns
        self.df = self.df.rename(columns={
            "listed_in": self.columns['genre'],
            "director": self.columns['director'],
            "cast": self.columns['cast'],
            "description": self.columns['description'],
            "title": self.columns['title'],
            "date_added": self.columns['date_added'],
            "country": self.columns['country']
        })

        # Fill missing values
        self.df[self.columns['country']] = self.df[self.columns['country']].fillna(self.df[self.columns['country']].mode()[0])
        self.df[self.columns['date_added']] = self.df[self.columns['date_added']].fillna(self.df[self.columns['date_added']].mode()[0])
        self.df[self.columns['rating']] = self.df[self.columns['rating']].fillna(self.df[self.columns['country']].mode()[0])
        self.df = self.df.dropna(how='any', subset=[self.columns['cast'], self.columns['director']])

        # Further processing
        self.df['category'] = self.df[self.columns['genre']].apply(lambda x: x.split(",")[0])
        self.df['YearAdded'] = self.df[self.columns['date_added']].apply(lambda x: x.split(" ")[-1])
        self.df['MonthAdded'] = self.df[self.columns['date_added']].apply(lambda x: x.split(" ")[0])
        self.df['country'] = self.df[self.columns['country']].apply(lambda x: x.split(",")[0])

        return self.df
```

---

### Data Transformation

#### DataTransformer Class

Next, we introduce the **DataTransformer** class. This class handles transforming the dataset by cleaning and combining relevant features into a single ‘soup’ of text for each movie. This will be the transformed data used to train the machine learning model.

```python
class DataTransformer:
    def __init__(self, df):
        """Initialize the DataTransformer with the DataFrame."""
        self.df = df
        self.features = ['category', 'director_name', 'cast_members', 'summary', 'movie_title']
        self.filters = self.df[self.features]

    @staticmethod
    def clean_text(text):
        """Clean the text by converting it to lowercase and removing spaces."""
        return str.lower(text.replace(" ", ""))

    def apply_transformations(self):
        """Apply transformations to the data."""
        logging.info("Applying data transformations")
        for feature in self.features:
            self.filters[feature] = self.filters[feature].apply(self.clean_text)

        self.filters['Soup'] = self.filters.apply(self.create_soup, axis=1)
        return self.filters

    @staticmethod
    def create_soup(row):
        """Create a combined 'soup' of features for each movie."""
        return f"{row['director_name']} {row['cast_members']} {row['category']} {row['summary']}"
```

---

### Model Training

#### ModelTrainer Class

The **ModelTrainer** class trains a machine learning model using the **CountVectorizer** to convert the text data into a matrix of token counts. The cosine similarity matrix is then computed to measure similarity between movies.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ModelTrainer:
    def __init__(self, filters):
        """Initialize the ModelTrainer with the filtered DataFrame."""
        self.filters = filters
        self.count_vectorizer = CountVectorizer(stop_words='english')

    def train_model(self):
        """Train the model and compute the cosine similarity matrix."""
        logging.info("Training model")
        self.count_matrix = self.count_vectorizer.fit_transform(self.filters['Soup'])
        self.cosine_sim_matrix = cosine_similarity(self.count_matrix, self.count_matrix)
        return self.cosine_sim_matrix
```

---

### Generating Recommendations

#### Recommender Class

The **Recommender** class generates movie recommendations based on cosine similarity. It finds movies that are most similar to the specified movie title and returns a list of recommended movies.

```python
class Recommender:
    def __init__(self, df, filters, cosine_sim_matrix):
        """Initialize the Recommender with the DataFrame, filters, and cosine similarity matrix."""
        self.df = df
        self.cosine_sim_matrix = cosine_sim_matrix
        filters = filters.reset_index()
        self.indices = pd.Series(filters.index, index=filters['movie_title'])

    def get_recommendations(self, title):
        """Get movie recommendations based on the given title."""
        logging.info("Getting recommendations for title: %s", title)
        title = title.replace(' ', '').lower()
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return self.df['movie_title'].iloc[movie_indices]
```

---

### Putting It All Together

Here’s how we integrate all the components discussed above. We demonstrate how to load the data, preprocess it, transform it, train the model, and finally generate movie recommendations.

```python
data_loader = DataLoader(config['data']['filepath'], config['data']['columns'])
df = data_loader.preprocess_data()

data_transformer = DataTransformer(df)
filters = data_transformer.apply_transformations()

model_trainer = ModelTrainer(filters)
cosine_sim_matrix = model_trainer.train_model()

recommender = Recommender(df, filters, cosine_sim_matrix)
print(recommender.get_recommendations('PK'))
```

---

### Conclusion

In this blog post, we walked through the process of building a movie recommendation system from scratch. Here's a recap of what we covered:

- **Data Loading and Preprocessing**: We introduced the **DataLoader** class to load and clean the movie dataset.
- **Data Transformation**: The **DataTransformer** class combined features into a text 'soup' for model training.
- **Model Training**: We used the **ModelTrainer** class to compute the cosine similarity matrix.
- **Generating Recommendations**: The **Recommender** class provided movie recommendations based on similarity.

By following these steps, you can create your own recommendation system and apply it to various datasets and use cases. Feel free to experiment with the code, try it on different datasets, and modify the model parameters to enhance the recommendations.

Happy coding!