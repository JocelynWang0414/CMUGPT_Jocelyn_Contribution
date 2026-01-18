import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from utils.embeddings import get_text_embedding, get_embeddings_for_list
from config import POSSIBLE_VALUES
from recommenders.data_loader import get_all_courses_from_db, get_knowledge_areas

def generate_dummy_user_data_pd(num_students, num_courses, max_courses_per_student, 
                               min_courses_per_student, num_concentrations_per_major, 
                               max_knowledge_area_of_interest):
    """Generate dummy student data for collaborative filtering"""
    students = []
    all_courses = get_all_courses_from_db()
    knowledge_areas = get_knowledge_areas()
    
    for _ in range(num_students):
        student = {}
        concentrations = {major: [f"Concentration {i+1}" for i in range(random.randint(1, num_concentrations_per_major))] 
                        for major in POSSIBLE_VALUES['majors']}
        student["Major"] = major = random.choice(POSSIBLE_VALUES['majors'])
        student["Concentration"] = random.choice([None] + concentrations[major])
        student["Minor"] = random.choice([None] + [m for m in POSSIBLE_VALUES['majors'] if m != major])
        student["Level"] = random.choice(POSSIBLE_VALUES['levels_of_study'])
        student["Career Direction"] = random.choice(POSSIBLE_VALUES['career_directions'])
        student["Knowledge Areas"] = random.sample(
            knowledge_areas, 
            k=random.randint(1, min(max_knowledge_area_of_interest, len(knowledge_areas)))
        )
        
        taken_courses = random.sample(
            all_courses, 
            k=random.randint(min_courses_per_student, min(max_courses_per_student, len(all_courses)))
        )
        
        for course in all_courses:
            if course in taken_courses:
                student[f"Course_{course}_Satisfaction"] = random.randint(1, 5)
                student[f"Course_{course}_Grade"] = random.choice(POSSIBLE_VALUES['ALL_POSSIBLE_GRADES'])
            else:
                student[f"Course_{course}_Satisfaction"] = None
                student[f"Course_{course}_Grade"] = None
        
        students.append(student)
    
    students_df = pd.DataFrame(students)
    return students_df

def convert_to_embeddings_students_df(students_df):
    # Get embeddings for all textual data
    students_df['Major_Embedding'] = students_df['Major'].apply(get_text_embedding)
    students_df['Concentration_Embedding'] = students_df['Concentration'].apply(lambda x: get_text_embedding(x) if x else np.zeros(768))
    students_df['Minor_Embedding'] = students_df['Minor'].apply(lambda x: get_text_embedding(x) if x else np.zeros(768))
    students_df['Level_Embedding'] = students_df['Level'].apply(get_text_embedding)
    students_df['Career_Direction_Embedding'] = students_df['Career Direction'].apply(get_text_embedding)
    students_df['Knowledge_Areas_Embedding'] = students_df['Knowledge Areas'].apply(get_embeddings_for_list)

    return students_df

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def compute_similarity(students_df):
    """Compute similarity matrix between students based on courses and features"""
    # Compute course similarity using Jaccard similarity
    course_columns = [col for col in students_df.columns if "Course_" in col and "_Satisfaction" not in col and "_Grade" not in col]
    if not course_columns:
        course_columns = [col for col in students_df.columns if "Course_" in col and "_Grade" in col]
    
    course_matrix = students_df[course_columns].notnull().astype(int).values
    
    # Avoid division by zero
    denominators = np.dot(course_matrix.sum(axis=1, keepdims=True), course_matrix.sum(axis=1, keepdims=True).T)
    denominators = np.maximum(denominators, 1e-8)  # Avoid division by zero
    
    course_similarity = np.dot(course_matrix, course_matrix.T) / denominators

    # Compute textual feature similarity using cosine similarity
    embedding_columns = ['Major_Embedding', 'Concentration_Embedding', 'Minor_Embedding',
                        'Level_Embedding', 'Career_Direction_Embedding', 'Knowledge_Areas_Embedding']
    
    # Make sure all embedding columns exist
    for col in embedding_columns:
        if col not in students_df.columns:
            students_df[col] = [np.zeros(768) for _ in range(len(students_df))]
    
    # Concatenate embeddings
    embeddings = np.array([
        np.concatenate([
            np.array(row[col]) for col in embedding_columns
        ]) for _, row in students_df.iterrows()
    ])
    
    # Calculate cosine similarity
    feature_similarity = cosine_similarity(embeddings)
    
    # Combine similarities
    similarity_matrix = 0.5 * course_similarity + 0.5 * feature_similarity
    
    # Set self-similarity to 1
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return similarity_matrix

def cf_recommendations(student_profile, students_df):
    course_columns = [col for col in students_df.columns if "Course_" in col and "_Satisfaction" not in col]
    num_courses = len(course_columns)
    predicted_ratings = np.zeros((num_courses))
    num_students = len(students_df)
    students_df.loc[num_students] = student_profile.iloc[0]
    similarity_matrix = compute_similarity(students_df)
    i = num_students
    for j, course in enumerate(course_columns):
          if pd.notnull(students_df.at[i,course]): #students_df.iloc[i][course]
              predicted_ratings[j] = students_df.at[i,course]
          else:
              weighted_sum = 0
              similarity_sum = 0
              for k in range(num_students):
                  if i != k and pd.notnull(students_df.at[k,course]):
                      weighted_sum += similarity_matrix[i, k] * students_df.at[k,course]
                      similarity_sum += similarity_matrix[i, k]
              if similarity_sum > 0:
                  predicted_ratings[j] = weighted_sum / similarity_sum
              else:
                  predicted_ratings[j] = 0
    
    predicted_df = pd.DataFrame([dict(zip(course_columns, predicted_ratings))])

    available_courses = [col for col in course_columns if student_profile[col].astype(str).tolist()[0] == 'None']

    # sorting available courses by predicted rating
    course_ratings = [(course, predicted_ratings[j]) for j, course in enumerate(course_columns)
                      if course in available_courses]
    top_courses = sorted(course_ratings, key=lambda x: x[1], reverse=True)


    return [course.split("_")[1] for course, _ in top_courses[:10]], predicted_df

def get_dummy_user_input_with_embeddings():
    """Generate a dummy user profile with embeddings for testing"""
    MyUser = {}
    MyUser["Major"] = major = random.choice(POSSIBLE_VALUES['majors'])
    
    concentrations = POSSIBLE_VALUES.get('concentrations', {}).get(major, [])
    if concentrations:
        MyUser["Concentration"] = random.choice([None] + concentrations)
    else:
        MyUser["Concentration"] = None
    
    MyUser["Minor"] = random.choice([None] + [m for m in POSSIBLE_VALUES['majors'] if m != major])
    MyUser["Level"] = random.choice(POSSIBLE_VALUES['levels_of_study'])
    MyUser["Career Direction"] = random.choice(POSSIBLE_VALUES['career_directions'])
    
    knowledge_areas = get_knowledge_areas()
    max_areas = min(5, len(knowledge_areas))
    MyUser["Knowledge Areas"] = random.sample(knowledge_areas, 
                                            k=random.randint(1, max_areas))

    all_courses = get_all_courses_from_db()
    max_courses = min(POSSIBLE_VALUES.get('max_courses_per_student', 5), len(all_courses))
    min_courses = min(POSSIBLE_VALUES.get('min_courses_per_student', 3), max_courses)
    
    taken_courses = random.sample(all_courses, 
                                k=random.randint(min_courses, max_courses))
    
    for course in all_courses:
        if course in taken_courses:
            MyUser[f"Course_{course}_Satisfaction"] = random.randint(1, 5)
            MyUser[f"Course_{course}_Grade"] = random.choice(POSSIBLE_VALUES['ALL_POSSIBLE_GRADES'])
        else: 
            MyUser[f"Course_{course}_Satisfaction"] = None
            MyUser[f"Course_{course}_Grade"] = None
    
    MyUser = pd.DataFrame([MyUser])

    # Add embeddings
    MyUser['Major_Embedding'] = MyUser['Major'].apply(get_text_embedding)
    MyUser['Concentration_Embedding'] = MyUser['Concentration'].apply(
        lambda x: get_text_embedding(x) if x else np.zeros(768))
    MyUser['Minor_Embedding'] = MyUser['Minor'].apply(
        lambda x: get_text_embedding(x) if x else np.zeros(768))
    MyUser['Level_Embedding'] = MyUser['Level'].apply(get_text_embedding)
    MyUser['Career_Direction_Embedding'] = MyUser['Career Direction'].apply(get_text_embedding)
    MyUser['Knowledge_Areas_Embedding'] = MyUser['Knowledge Areas'].apply(get_embeddings_for_list)

    return MyUser