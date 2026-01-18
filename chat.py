from flask import request, jsonify
import re
from db_models import Course
from api import api_blueprint

@api_blueprint.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests from frontend"""
    try:
        # Extract data from request
        data = request.get_json() or {}
        query = data.get("query", "")
        user_profile = data.get("userProfile", {})
        recommendations = data.get("recommendations", {})
        
        # Process the query to extract relevant information
        course_match = extract_course_from_query(query)
        
        if course_match:
            # If query is about a specific course, get detailed information
            course_info = get_course_info(course_match)
            if course_info:
                response = generate_course_response(course_info, user_profile)
            else:
                response = f"I don't have information about course {course_match}. Would you like me to recommend similar courses instead?"
        else:
            # Otherwise, generate a more general response
            response = generate_general_response(query, user_profile, recommendations)
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing chat query: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_course_from_query(query):
    """Extract course number from query using regex"""
    course_pattern = r'\b(\d{1,2}-\d{3})\b'
    match = re.search(course_pattern, query)
    if match:
        return match.group(1)
    return None

def get_course_info(course_number):
    """Get detailed information about a course"""
    course = Course.query.filter_by(course_number=course_number).first()
    if course:
        return {
            'course_number': course.course_number,
            'title': course.title,
            'description': course.description,
            'semester': course.semester,
            'units': course.units
        }
    return None

def generate_course_response(course_info, user_profile):
    """Generate a detailed response about a specific course"""
    response = f"{course_info['course_number']}: {course_info['title']} ({course_info['units']} units, {course_info['semester']})\n\n"
    response += f"{course_info['description']}\n\n"
    
    # Add personalized note if user has a major
    if user_profile and 'major' in user_profile:
        response += f"This course could be particularly valuable for your {user_profile['major']} major."
    
    return response

def generate_general_response(query, user_profile, recommendations):
    """Generate a response for general queries"""
    query_lower = query.lower()
    
    # Check for recommendation-related queries
    if any(word in query_lower for word in ['recommend', 'suggest', 'course', 'class']):
        if recommendations and (recommendations.get('cf') or recommendations.get('cbf')):
            courses = recommendations.get('cf', [])[:3]
            return f"Based on your profile, I recommend checking out these courses: {', '.join(courses)}. Would you like more details about any of them?"
        else:
            return "I can recommend courses based on your academic profile. What are your major and interests?"
    
    # Check for major-related queries
    if any(word in query_lower for word in ['major', 'concentration', 'program']):
        if user_profile and user_profile.get('major'):
            return f"Your major is {user_profile.get('major')}. I can help you find courses that are required for your major or complement your interests."
        else:
            return "I can provide information about different majors at CMU. Which major are you interested in learning more about?"
    
    # Check for career-related queries
    if any(word in query_lower for word in ['career', 'job', 'industry', 'work']):
        if user_profile and user_profile.get('careerDirection'):
            return f"For your interest in {user_profile.get('careerDirection')}, I recommend courses that develop both technical skills and domain knowledge. Would you like specific recommendations?"
        else:
            return "I can suggest courses that align with different career paths. What career are you interested in?"
    
    # Generic response for other queries
    return "I'm your CMU course assistant. I can help you find courses, learn about requirements, or get personalized recommendations. What would you like to know?"