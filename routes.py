from flask import jsonify, request
from db_models import Course
from config import POSSIBLE_VALUES, KNOWLEDGE_AREA_DESCRIPTIONS
from . import api_blueprint
from recommenders.data_loader import get_ALL_COURSES

@api_blueprint.route('/options', methods=['GET'])
def get_options():
    """Returns all possible values for form options"""
    return jsonify({
        'majors': POSSIBLE_VALUES.get('majors', []),
        'levels_of_study': POSSIBLE_VALUES.get('levels_of_study', []),
        'career_directions': POSSIBLE_VALUES.get('career_directions', []),
        'knowledge_areas': list(KNOWLEDGE_AREA_DESCRIPTIONS.keys()),
        'concentrations': POSSIBLE_VALUES.get('concentrations', {})
    })

# @api_blueprint.route('/courses', methods=['GET'])
# def get_courses():
#     """Returns a list of courses for the course history selector"""
#     try:
#         limit = int(request.args.get('limit', 50))
        
#         courses = Course.query.limit(limit).all()
#         course_list = [course.to_dict() for course in courses]
        
#         return jsonify({
#             'courses': course_list
#         })
#     except Exception as e:
#         print(f"Error getting courses: {str(e)}")
#         return jsonify({'error': str(e)}), 500

@api_blueprint.route('/courses', methods=['GET'])
def get_courses():
    # getting available courses 
    try:
        df, course_number_list = get_ALL_COURSES(num_courses=200)
        
        # Format courses for the frontend
        courses = []
        for i, course_number in enumerate(course_number_list):
            if i < len(df):
                course_name = df.iloc[i].get('Description', '')[:50]  # Get first 50 chars as preview
                courses.append({
                    'course_number': course_number,
                    'course_name': course_name
                })
        
        return jsonify({'courses': courses})
    except Exception as e:
        #logger.error(f"Error fetching courses: {str(e)}")
        print(f"Error fetching courses: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_blueprint.route('/course/<course_number>', methods=['GET'])
def get_course(course_number):
    """Returns details for a specific course"""
    try:
        course = Course.query.filter_by(course_number=course_number).first()
        
        if course:
            return jsonify(course.to_dict())
        else:
            return jsonify({'error': 'Course not found'}), 404
    except Exception as e:
        print(f"Error getting course: {str(e)}")
        return jsonify({'error': str(e)}), 500