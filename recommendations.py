from flask import request, jsonify
from api import api_blueprint
from recommenders.collaborative import (
    cf_recommendations, 
    get_dummy_user_input_with_embeddings, 
    generate_dummy_user_data_pd, 
    convert_to_embeddings_students_df
)
from recommenders.content_based import cbf_recommendations
from recommenders.trend_analysis import TrendAnalysis
from recommenders.data_loader import get_ALL_COURSES
from utils.data_processing import transform_user_profile, get_course_details

@api_blueprint.route('/recommend', methods=['POST'])
def recommend():
    """
    Generate course recommendations based on the user's profile.
    If no user profile is provided, a dummy profile will be used.
    """
    try:
        data = request.get_json() or {}
        
        # Use provided user profile if available; otherwise, generate dummy data.
        if 'user_profile' in data:
            user_profile = transform_user_profile(data['user_profile'])
        else:
            user_profile = get_dummy_user_input_with_embeddings()
        
        # Load course data from the database (or dummy fallback)
        df, course_number_list = get_ALL_COURSES(num_courses=200)
        
        # Generate dummy historical student data for collaborative filtering
        num_students = 100
        num_courses = 20  # This can be adjusted or read from your configuration
        max_courses_per_student = 5
        min_courses_per_student = 3
        max_knowledge_area = 2
        num_concentrations_per_major = 2  # Or use your config values
        
        students_df = generate_dummy_user_data_pd(
            num_students, num_courses, 
            max_courses_per_student, min_courses_per_student, 
            num_concentrations_per_major, max_knowledge_area
        )
        students_df = convert_to_embeddings_students_df(students_df)
        
        # Generate recommendations from the collaborative filtering module
        cf_recommended_courses, cf_predictions = cf_recommendations(user_profile, students_df)
        
        # Generate content-based filtering recommendations
        cbf_recommended_courses, normalized_similarity_df = cbf_recommendations(df, user_profile)
        
        # Perform trend analysis based on historical data
        TA = TrendAnalysis(students_df, user_profile, cf_recommended_courses)
        trend_analysis_major = TA.check_course_popularity('Major', True)
        trend_analysis_career = TA.check_course_popularity('Career Direction', True)
        
        # Optionally, include additional course details for the frontend
        #course_details = get_course_details(df)
        
        return jsonify({
            'cf_recommendations': cf_recommended_courses,
            'cbf_recommendations': cbf_recommended_courses,
            'trend_analysis_major': trend_analysis_major,
            'trend_analysis_career': trend_analysis_career
            #'course_details': course_details
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
