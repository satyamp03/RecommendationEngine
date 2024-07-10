import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# Load the preprocessed data
data = pd.read_csv('data/preprocessed_data.csv')
print("Data loaded successfully.")
print(data.head())

# Normalize titles for consistent matching
data['Titile'] = data['Titile'].str.lower().str.strip()

# Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.2, random_state=42)
print("Training and testing sets created.")
print(f"Train size: {train.shape[0]}, Test size: {test.shape[0]}")

# Calculate the cosine similarity matrix on the training set
train_numeric = train.drop(columns=['Titile', 'Vote_count', 'Category'])
cosine_sim = cosine_similarity(train_numeric)
print("Cosine similarity matrix calculated.")
print(cosine_sim.shape)


def get_recommendations(index, cosine_sim, data, num_recommendations=5):
    if index >= cosine_sim.shape[0]:
        raise IndexError(f"Index {index} is out of bounds for cosine similarity matrix with size {cosine_sim.shape[0]}")
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    show_indices = [i[0] for i in sim_scores]
    return data.iloc[show_indices]


# Test the recommendation function
try:
    recommendations = get_recommendations(0, cosine_sim, train)
    print("Recommendations for index 0:")
    print(recommendations)
except Exception as e:
    print(f"An error occurred: {e}")


# Implement a function to evaluate the model
def evaluate_model(cosine_sim, train, test):
    precision = []
    recall = []
    for index, row in test.iterrows():
        if index >= cosine_sim.shape[0]:
            continue  # Skip if index is out of bounds

        try:
            recommendations = get_recommendations(index, cosine_sim, train)
            recommendations = recommendations.copy()
            recommendations['Titile'] = recommendations['Titile'].str.lower().str.strip()
            relevant_items = test[test['Titile'].isin(recommendations['Titile'])]

            print(f"Index: {index}")
            print(f"Test title: {row['Titile']}")
            print(f"Recommendations: {recommendations['Titile'].tolist()}")
            print(f"Relevant items: {relevant_items['Titile'].tolist()}")

            if len(relevant_items) > 0:
                precision.append(
                    precision_score([1] * len(relevant_items), [1] * len(recommendations), zero_division=0))
                recall.append(recall_score([1] * len(relevant_items), [1] * len(recommendations), zero_division=0))
        except Exception as e:
            print(f"An error occurred: {e}")

    if precision and recall:
        return {'precision': sum(precision) / len(precision), 'recall': sum(recall) / len(recall)}
    else:
        return {'precision': 0, 'recall': 0}


# Evaluate the model
evaluation_results = evaluate_model(cosine_sim, train, test)
print("Evaluation results:")
print(evaluation_results)