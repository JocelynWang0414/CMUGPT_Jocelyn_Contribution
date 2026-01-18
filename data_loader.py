import os
import pandas as pd
from urllib.parse import urlparse
import pg8000
from config import KNOWLEDGE_AREA_DESCRIPTIONS
import random

# Store cached course data
_course_data = None
_course_numbers = None

def get_ALL_COURSES(num_courses=200, use_cache=True):
    """
    Get course data from database
    
    Args:
        num_courses: Number of courses to retrieve
        use_cache: Whether to use cached data if available
        
    Returns:
        DataFrame with course data and list of course numbers
    """
    global _course_data, _course_numbers
    
    # Return cached data if available and requested
    if use_cache and _course_data is not None and _course_numbers is not None:
        return _course_data, _course_numbers
    
    try:
        # Parse the DATABASE_URL
        db_url = os.getenv('DATABASE_URL')
        
        # Parse the URL to get components
        parsed = urlparse(db_url)
        
        # Extract connection parameters
        user = parsed.username
        password = parsed.password
        host = parsed.hostname
        port = parsed.port or 5432
        database = parsed.path[1:]
        
        # Connect with pg8000
        conn = pg8000.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database
        )
        
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM courses LIMIT {num_courses}")
        data_query_output = cursor.fetchall()
        data_query_output = list(data_query_output)

        course_number_list, course_description_list = [], []
        for course in data_query_output:
            index, description, course_number, course_title, course_semester, course_unit = course
            course_number_list.append(course_number)
            course_description_list.append(description)
        cursor.close()
        conn.close()

        data = {'Course Number': course_number_list, 'Description': course_description_list}
        df = pd.DataFrame(data)
        
        # Cache the data
        _course_data = df
        _course_numbers = course_number_list
        
        return df, course_number_list
    except Exception as e:
        print(f"Connection error: {type(e).__name__}: {e}")
        
        # If error occurs and no cached data, create dummy data
        if _course_data is None or _course_numbers is None:
            # Generate dummy course data
            dummy_courses = []
            dummy_course_numbers = []
            
            for i in range(num_courses):
                dept = random.choice(["15", "10", "17", "18", "21", "36", "70", "80", "85"])
                num = random.randint(100, 999)
                course_number = f"{dept}-{num}"
                description = f"This is a course about {random.choice(['computer science', 'statistics', 'design', 'business', 'engineering', 'mathematics', 'science', 'art'])}"
                
                dummy_courses.append({
                    'Course Number': course_number,
                    'Description': description
                })
                dummy_course_numbers.append(course_number)
            
            dummy_df = pd.DataFrame(dummy_courses)
            
            # Cache the dummy data
            _course_data = dummy_df
            _course_numbers = dummy_course_numbers
            
            return dummy_df, dummy_course_numbers
        
        # Return cached data if available
        return _course_data, _course_numbers

def get_all_courses_from_db():
    """Get all course numbers from database"""
    _, course_numbers = get_ALL_COURSES()
    return course_numbers

def get_knowledge_areas():
    """Get list of knowledge areas"""
    return list(KNOWLEDGE_AREA_DESCRIPTIONS.keys())