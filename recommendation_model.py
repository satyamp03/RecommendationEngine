import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the preprocessed data
data = pd.read_csv('data/preprocessed_data.csv')

# Ensure the column names are correct
print("\nColumn names in the preprocessed data:")
print(data.columns)

# Drop non-numeric columns
data_numeric = data.drop(columns=['Titile', 'Vote_count', 'Category'])

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(data_numeric)

# Define a function to get recommendations based on a show's index
def get_recommendations(index, cosine_sim, data, num_recommendations=5):
    # Get the pairwise similarity scores of all shows with the given show
    sim_scores = list(enumerate(cosine_sim[index]))

    # Sort the shows based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar shows
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the show indices
    show_indices = [i[0] for i in sim_scores]

    # Return the top most similar shows
    return data.iloc[show_indices]

# Example: Get recommendations for the first show in the dataset
try:
    recommendations = get_recommendations(0, cosine_sim, data)
    print(recommendations)
except Exception as e:
    print(f"An error occurred: {e}")