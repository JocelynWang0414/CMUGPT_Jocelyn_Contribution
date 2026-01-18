import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.embeddings import get_distilbert_embedding
from config import KNOWLEDGE_AREA_DESCRIPTIONS

def add_knowledge_area(df):
    """Assign knowledge areas to courses based on description similarity"""
    # Check if df is empty
    if df.empty:
        return df
    
    # Handle missing descriptions
    df['Description'] = df['Description'].fillna('')
    
    # Combine course descriptions and knowledge area descriptions
    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = list(df["Description"]) + list(KNOWLEDGE_AREA_DESCRIPTIONS.values())
    all_texts = [str(text).lower() for text in all_texts]
    all_texts = [text.replace("\n", " ") for text in all_texts]
    all_texts = [text.replace("\t", " ") for text in all_texts]

    # Compute TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Compute similarity between course descriptions and knowledge areas
    course_vectors = tfidf_matrix[:len(df)]
    knowledge_vectors = tfidf_matrix[len(df):]
    similarity_matrix = cosine_similarity(course_vectors, knowledge_vectors)
    
    # Assign each course to the most similar knowledge area
    knowledge_areas = list(KNOWLEDGE_AREA_DESCRIPTIONS.keys())
    df["Knowledge Area"] = [knowledge_areas[i] for i in similarity_matrix.argmax(axis=1)]
    
    return df

def add_professor(df):
    """Add dummy professor data to courses"""
    # Check if df is empty
    if df.empty:
        return df
        
    num_profs = 100
    sparsity = 0.9
    df['Professors'] = [np.random.choice([0, 1], size=num_profs, p=[sparsity, 1-sparsity]).tolist() 
                       for _ in range(len(df))]
    return df

def add_TFIDF_keywords(df):
    """Extract keywords from course descriptions using TF-IDF"""
    # Check if df is empty
    if df.empty:
        return df
    
    # Handle missing descriptions
    df['Description'] = df['Description'].fillna('')
    
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_df=0.95, 
        min_df=0.05, 
        sublinear_tf=True, 
        token_pattern=r"(?u)\b[a-zA-Z]+\b"
    )
    
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])
        
        def get_top_keywords_tfidf(row_index, top_n=5):
            feature_names = tfidf_vectorizer.get_feature_names_out()
            coo_matrix = tfidf_matrix[row_index].tocoo()
            if coo_matrix.data.size == 0:
                return []
            top_indices = np.argsort(coo_matrix.data)[::-1][:min(top_n, coo_matrix.data.size)]
            keywords = [feature_names[coo_matrix.col[i]] for i in top_indices]
            return keywords
        
        df['Keywords_TFIDF'] = [get_top_keywords_tfidf(i) for i in range(len(df))]
    except ValueError as e:
        print(f"Error in TF-IDF processing: {e}")
        # If vectorization fails, add empty keywords
        df['Keywords_TFIDF'] = [[] for _ in range(len(df))]
    
    return df

def add_keywords_knowledgeArea_embeddings(df):
    """Add embeddings for keywords and knowledge areas"""
    # Check if df is empty
    if df.empty:
        return df
    
    def get_distilbert_embedding_list(keywords):
        if not keywords:
            return None
        try:
            embeddings = []
            for keyword in keywords:
                if keyword:
                    embeddings.append(get_distilbert_embedding([keyword]))
            if not embeddings:
                return None
            return np.array(embeddings)
        except Exception as e:
            print(f"Error getting keyword embedding: {e}")
            return None
    
    # Apply embedding functions with error handling
    df['Keywords_TFIDF_Embeddings'] = df['Keywords_TFIDF'].apply(get_distilbert_embedding_list)
    
    # Ensure all knowledge areas have values
    df['Knowledge Area'] = df['Knowledge Area'].fillna('Others')
    
    # Get knowledge area embeddings
    df['Knowledge_Area_Embedding'] = df['Knowledge Area'].apply(
        lambda x: get_distilbert_embedding([x]) if x else np.zeros((1, 768)))
    
    return df

def return_final_embedded_df(df):
    """Return final dataframe with only needed columns"""
    # Check if df is empty
    if df.empty:
        return df
        
    # Select required columns
    final_df = df[['Course Number', 'Professors', 'Keywords_TFIDF_Embeddings', 'Knowledge_Area_Embedding']]
    
    # Drop rows with missing embeddings
    final_df = final_df.dropna(subset=['Keywords_TFIDF_Embeddings', 'Knowledge_Area_Embedding'])
    
    return final_df

def calculate_professor_similarity(df):
    """Calculate similarity between courses based on professors"""
    # Check if df is empty
    if df.empty:
        return np.zeros((0, 0))
        
    def jaccard_similarity(list1, list2):
        if not list1 or not list2:
            return 0
        intersection = len(set(list1) & set(list2))
        union = len(set(list1) | set(list2))
        return intersection / union if union > 0 else 0

    num_courses = len(df)
    similarity_matrix = np.zeros((num_courses, num_courses))
    
    for i in range(num_courses):
        for j in range(i + 1, num_courses):
            professors1 = df['Professors'].iloc[i]
            professors2 = df['Professors'].iloc[j]

            # Convert professor lists to sets of indices for Jaccard calculation
            prof_set1 = set(i for i, val in enumerate(professors1) if val == 1)
            prof_set2 = set(i for i, val in enumerate(professors2) if val == 1)

            similarity_matrix[i, j] = jaccard_similarity(list(prof_set1), list(prof_set2))
            similarity_matrix[j, i] = similarity_matrix[i, j]  # Symmetry

    return similarity_matrix

def calculate_keyword_similarity(df):
    """Calculate similarity between courses based on keywords"""
    # Check if df is empty
    if df.empty:
        return np.zeros((0, 0))
        
    num_courses = len(df)
    similarity_matrix = np.zeros((num_courses, num_courses))
    
    for i in range(num_courses):
        for j in range(i + 1, num_courses):
            keywords1 = df['Keywords_TFIDF_Embeddings'].iloc[i]
            keywords2 = df['Keywords_TFIDF_Embeddings'].iloc[j]

            if keywords1 is None or keywords2 is None or len(keywords1) == 0 or len(keywords2) == 0:
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
                continue

            # Average embeddings if multiple keywords
            embedding1 = np.mean(keywords1, axis=0)
            embedding2 = np.mean(keywords2, axis=0)

            # Calculate cosine similarity
            sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetry

    return similarity_matrix

def calculate_knowledge_area_similarity(df):
    """Calculate similarity between courses based on knowledge areas"""
    # Check if df is empty
    if df.empty:
        return np.zeros((0, 0))
        
    num_courses = len(df)
    similarity_matrix = np.zeros((num_courses, num_courses))
    
    for i in range(num_courses):
        for j in range(i + 1, num_courses):
            area1 = df['Knowledge_Area_Embedding'].iloc[i]
            area2 = df['Knowledge_Area_Embedding'].iloc[j]
            
            # Calculate cosine similarity
            sim = cosine_similarity(area1, area2)[0][0]
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetry
            
    return similarity_matrix

def calculate_aggregated_similarity(df):
    """Calculate aggregated similarity based on multiple factors"""
    # Check if df is empty
    if df.empty:
        return pd.DataFrame()
    
    # Calculate individual similarity matrices
    prof_mx = calculate_professor_similarity(df)
    keyw_mx = calculate_keyword_similarity(df)
    knowlarea_mx = calculate_knowledge_area_similarity(df)
    
    # Weights for different similarity measures
    w_prof = 0.33
    w_key = 0.33
    w_area = 0.34
    
    num_courses = len(df)
    aggregated_similarity_matrix = np.zeros((num_courses, num_courses))

    for i in range(num_courses):
        for j in range(i + 1, num_courses):
            prof_sim = prof_mx[i, j]
            key_sim = keyw_mx[i, j]
            area_sim = knowlarea_mx[i, j]

            aggregated_similarity_matrix[i, j] = (
                w_prof * prof_sim + w_key * key_sim + w_area * area_sim
            )
            aggregated_similarity_matrix[j, i] = aggregated_similarity_matrix[i, j]  # Symmetry

    # Normalize similarity to [1, 5] range
    if num_courses > 0:
        min_sim = aggregated_similarity_matrix.min()
        max_sim = aggregated_similarity_matrix.max()
        
        # Avoid division by zero if all similarities are the same
        if max_sim > min_sim:
            normalized_similarity_matrix = 5 - (aggregated_similarity_matrix - min_sim) * 4 / (max_sim - min_sim)
        else:
            normalized_similarity_matrix = np.ones((num_courses, num_courses)) * 3  # Default mid-range value
    else:
        normalized_similarity_matrix = np.array([])

    # Create DataFrame with course numbers as indices
    course_numbers = df['Course Number'].values
    normalized_similarity_df = pd.DataFrame(
        normalized_similarity_matrix, index=course_numbers, columns=course_numbers
    )
    
    return normalized_similarity_df

def get_cbf_recommended_courses(student, normalized_similarity_df):
    """Get content-based filtering recommendations for a student"""
    # Check if normalized_similarity_df is empty
    if normalized_similarity_df.empty:
        return []
    
    cbf_similarity_ranking = dict()
    top_n_similar_courses_considered_per_liked_course = 5
    
    for course_number in normalized_similarity_df.columns:
        satisfaction_col = f"Course_{course_number}_Satisfaction"
        grade_col = f"Course_{course_number}_Grade"
        
        # Check if student has taken this course
        if (satisfaction_col in student.columns and grade_col in student.columns and 
            student[satisfaction_col].astype(str).tolist()[0] != 'None' and 
            student[grade_col].astype(str).tolist()[0] != 'None'):
            
            row_of_this_course = normalized_similarity_df[course_number]
            predicted_ratings = row_of_this_course.values
            sorted_indices = np.argsort(predicted_ratings)[::-1]
            top_n_indices = sorted_indices[:min(top_n_similar_courses_considered_per_liked_course, len(sorted_indices))]
            top_n_courses = [normalized_similarity_df.index[i] for i in top_n_indices]
            
            for course in top_n_courses:
                if course not in cbf_similarity_ranking:
                    cbf_similarity_ranking[course] = row_of_this_course[course]
                else:
                    cbf_similarity_ranking[course] += row_of_this_course[course]
    
    # If no rankings, return empty list
    if not cbf_similarity_ranking:
        return []
    
    # Exclude previously taken courses
    course_numbers = list(cbf_similarity_ranking.keys())
    for course_number in course_numbers:
        satisfaction_col = f"Course_{course_number}_Satisfaction"
        grade_col = f"Course_{course_number}_Grade"
        
        if (satisfaction_col in student.columns and grade_col in student.columns and 
            student[satisfaction_col].iloc[0] is not None and 
            student[grade_col].iloc[0] is not None):
            cbf_similarity_ranking[course_number] = -1  # Penalize previously taken courses
    
    # Get top recommendations
    cbf_max_num_courses = 10
    cbf_recommended_courses = sorted(
        cbf_similarity_ranking, key=cbf_similarity_ranking.get, reverse=True
    )[:min(cbf_max_num_courses, len(cbf_similarity_ranking))]

    return cbf_recommended_courses

def cbf_recommendations(df, new_user):
    """Generate content-based filtering recommendations"""
    # Check if df is empty
    if df.empty:
        return [], pd.DataFrame()
        
    # Process data
    df = add_knowledge_area(df)
    df = add_professor(df)
    df = add_TFIDF_keywords(df)
    df = add_keywords_knowledgeArea_embeddings(df)
    df = return_final_embedded_df(df)
    
    # Calculate similarity and recommendations
    normalized_similarity_df = calculate_aggregated_similarity(df)
    cbf_recommended_courses = get_cbf_recommended_courses(new_user, normalized_similarity_df)
    
    return cbf_recommended_courses, normalized_similarity_df