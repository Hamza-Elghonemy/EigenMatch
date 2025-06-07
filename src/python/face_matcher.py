import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class FaceMatcher:
    def __init__(self, pca_processor, similarity_metric='cosine'):
        self.pca_processor = pca_processor
        self.similarity_metric = similarity_metric

    def match_face(self, input_image, top_k=5):
        """Find the closest matching faces for the input image"""
        # Preprocess and transform the input image
        input_pca = self.pca_processor.transform(input_image)
        
        # Get dataset features and labels
        dataset_features, dataset_labels = self.pca_processor.get_dataset_features()
        
        # Find closest matches
        matches = self._find_closest_matches(input_pca, dataset_features, dataset_labels, top_k)
        
        return matches

    def _find_closest_matches(self, input_pca, dataset_features, dataset_labels, top_k=5):
        """Find and return the closest matching images from the dataset"""
        if self.similarity_metric == 'cosine':
            # Calculate cosine similarity
            similarities = cosine_similarity(input_pca, dataset_features)[0]
            # Higher similarity is better for cosine
            closest_indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[closest_indices]
        else:
            # Calculate Euclidean distance
            distances = euclidean_distances(input_pca, dataset_features)[0]
            # Lower distance is better for Euclidean
            closest_indices = np.argsort(distances)[:top_k]
            scores = distances[closest_indices]
        
        # Prepare results
        matches = []
        for i, idx in enumerate(closest_indices):
            match = {
                'person': dataset_labels[idx],
                'score': float(scores[i]),
                'rank': i + 1,
                'index': int(idx)
            }
            matches.append(match)
        
        return matches

    def get_person_matches(self, input_image, top_k=5):
        """Get matches grouped by person (best match per person)"""
        all_matches = self.match_face(input_image, top_k=len(self.pca_processor.dataset_labels))
        
        # Group by person and take best match for each
        person_matches = {}
        for match in all_matches:
            person = match['person']
            if person not in person_matches or match['score'] > person_matches[person]['score']:
                person_matches[person] = match
        
        # Sort by score and return top_k
        sorted_matches = sorted(person_matches.values(), 
                              key=lambda x: x['score'], 
                              reverse=(self.similarity_metric == 'cosine'))
        
        return sorted_matches[:top_k]