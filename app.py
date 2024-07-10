from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the preprocessed data
data = pd.read_csv('data/preprocessed_data.csv')

# Calculate the cosine similarity matrix
data_numeric = data.drop(columns=['Titile', 'Vote_count', 'Category'])
cosine_sim = cosine_similarity(data_numeric)

def get_recommendations(index, cosine_sim, data, num_recommendations=5):
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    show_indices = [i[0] for i in sim_scores]
    return data.iloc[show_indices]

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        show_index = int(request.args.get('index'))
        num_recommendations = int(request.args.get('num', 5))
        recommendations = get_recommendations(show_index, cosine_sim, data, num_recommendations)
        return recommendations.to_json(orient='records')
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)