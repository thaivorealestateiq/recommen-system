import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import torch

from recommender.model import MLP

data = pd.read_pickle('dataset/data.pkl')
preprocessor = joblib.load('models/preprocessor.joblib')

# Transform the features
feature_matrix = preprocessor.transform(data)

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Load the saved models
# preprocessor = joblib.load('myApp/models/preprocessor.joblib')
# dbscan = joblib.load('myApp/models/dbscan_model.joblib')
kmeans = joblib.load('models/kmeans_model.joblib')

checkpoint = './checkpoints/MLP.pth'

total_dataframe = pd.read_csv("./content/entire_dataset.csv")

def get_recommendations(search_param, n_recommendations=6):
    # Filter data based on search parameter
    search_results = data[data['Title'].str.contains(search_param, case=False) |
                          (data['Category'] == search_param) |
                          (data['Sub-Category'] == search_param)]

    # If no results found, return an empty DataFrame
    if search_results.empty:
        return pd.DataFrame()

    cluster_label = kmeans.labels_[search_results.index[0]]

    # Get the indices of products in the same cluster
    cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_label]

    # Get the pairwise similarity scores within the cluster
    sim_scores = [(i, cosine_sim[search_results.index[0]][i]) for i in cluster_indices]

    # Sort the products based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N most similar products
    sim_scores = sim_scores[:n_recommendations]
    product_indices = [i[0] for i in sim_scores]
    
    recommendations_kmeans = []

    # Append the recommendations to the list
    for idx in product_indices:
        recommendations_kmeans.append({
            'Title': data.iloc[idx]['Title'],
            'Category': data.iloc[idx]['Category'],
            'Sub_Category': data.iloc[idx]['Sub-Category'],
            'Price': data.iloc[idx]['Price'],
            'Ratings': data.iloc[idx]['Ratings'],
            'Total_Ratings': round(data.iloc[idx]['Total Ratings'], 2)
        })

    return recommendations_kmeans


model = MLP()
model.load_state_dict(torch.load(checkpoint, weights_only=True))
def recommendations(userID,itemID):
    pass