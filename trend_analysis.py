import pandas as pd
import numpy as np

class TrendAnalysis:
    """Analyze trends in course selection based on student profiles"""
    
    def __init__(self, students_df, test_student, recommended):
        """
        Initialize trend analysis
        
        Args:
            students_df: DataFrame with historical student data
            test_student: DataFrame with new student data
            recommended: List of recommended course IDs
        """
        self.data = students_df
        self.user = test_student
        self.recommendations = [f'Course_{course_number}_Grade' for course_number in recommended]

    def conditioned_analysis_recommendation_based(self, feature, same_feature_students, electives_only):
        """Analyze trends in recommended courses for students with similar feature"""
        statistics_recommendation_based = {}
        s = []
    
        for course in self.recommendations:
            if course not in same_feature_students.columns:
                continue
                
            count = 0
            for score in same_feature_students[course]:
                if pd.notnull(score) and score > 0:
                    count += 1
                    
            course_number = course.split('_')[1]
            statistics_recommendation_based[course_number] = count
                
        if not statistics_recommendation_based:
            return ["No trend data available for your recommendations."]
                
        statistics_recommendation_based = sorted(
            statistics_recommendation_based.items(), key=lambda item: item[1], reverse=True
        )[:min(10, len(statistics_recommendation_based))]

        recommendations = [rec.split('_')[1] for rec in self.recommendations if rec in same_feature_students.columns]
        
        if not recommendations:
            s.append(f"No specific course recommendations available based on your {feature}.")
        else:
            s.append(f'Based on your {feature} in {self.user.iloc[0][feature]}, we recommend these courses: {", ".join(recommendations)}')
        
        s.append(f'Particularly worthy of noting:  In {self.user.iloc[0][feature]} {feature.lower()}, ')
        
        has_popular_courses = False
        for course_name, number_of_people_taken in statistics_recommendation_based:
            proportion_of_students_taking_course = number_of_people_taken/len(same_feature_students)
            if proportion_of_students_taking_course > 0.2:
                has_popular_courses = True
                s.append(f" {proportion_of_students_taking_course* 100:.2f}% have taken course {course_name}")
                
        if not has_popular_courses:
            s.append(" no specific courses stand out as particularly common among your peers.")
                
        return s
    
    def conditioned_analysis_profile_based(self, feature, same_feature_students, electives_only):
        """Analyze trends in all courses for students with similar feature"""
        statistics_profile_based = {}
        s = []
        
        for i in range(len(same_feature_students)):
            student_data = same_feature_students.iloc[i]
            for column_name in self.data.columns:
                if "Course_" in column_name and '_Grade' in column_name:
                    if column_name in student_data and pd.notnull(student_data[column_name]) and student_data[column_name] > 0:
                        course_number = column_name.split('_')[1]
                        statistics_profile_based[course_number] = 1 if (course_number not in statistics_profile_based) else statistics_profile_based[course_number] + 1
                        
        if not statistics_profile_based:
            return ["No trend data available for courses taken by students with similar profiles."]
                        
        statistics_profile_based = sorted(
            statistics_profile_based.items(), key=lambda item: item[1], reverse=True
        )[:min(10, len(statistics_profile_based))]
        
        s.append(f" These courses are also popular in {self.user.iloc[0][feature]} {feature}:")
        
        has_popular_courses = False
        for course_name, number_of_people_taken in statistics_profile_based:
            proportion_of_students_taking_course = number_of_people_taken/len(same_feature_students)
            if proportion_of_students_taking_course > 0.2:
                has_popular_courses = True
                s.append(f" {proportion_of_students_taking_course* 100:.2f}% have taken {course_name}")
        
        if not has_popular_courses:
            s.append(" no specific courses stand out as particularly common among students with your profile.")
        
        return s

    def check_course_popularity(self, feature, electives_only):
        """Check popularity of courses for students with similar feature"""
        if feature not in self.data.columns:
            return [f"Feature '{feature}' not found in the dataset. Cannot analyze trends."]
            
        if feature not in self.user.columns or pd.isnull(self.user.iloc[0][feature]):
            return [f"You haven't specified your {feature}. Cannot analyze trends."]
            
        user_feature = self.user.iloc[0][feature]
        
        mask = self.data[feature] == user_feature
        same_feature_students = self.data[mask].copy()
        
        if len(same_feature_students) == 0:
            return [f"No students found with {feature}: {user_feature}. Cannot analyze trends."]
        
        s1 = self.conditioned_analysis_recommendation_based(feature, same_feature_students, electives_only)
        s2 = self.conditioned_analysis_profile_based(feature, same_feature_students, electives_only)
        
        return s1 + s2