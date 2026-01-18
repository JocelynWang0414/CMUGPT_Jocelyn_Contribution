from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class UserProfile(db.Model):
    __tablename__ = 'user_profiles'
    id = db.Column(db.Integer, primary_key=True)
    major = db.Column(db.String(30), nullable=False)
    study_level = db.Column(db.String(10), nullable=False)
    concentration = db.Column(db.String(30), nullable=False)
    career_direction = db.Column(db.String(30), nullable=False)
    hobbies = db.Column(db.String(50), nullable=False)
    difficulty_pref = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
                "id": self.id,
                "major": self.major,
                "study_level": self.study_level,
                "concentration": self.concentration,
                "career_direction": self.career_direction,
                "hobbies": self.hobbies,
                "difficulty_pref": self.difficulty_pref
                }

class Course(db.Model):
    __tablename__ = 'courses'
    id = db.Column(db.Integer, primary_key=True)
    course_number = db.Column(db.String(10), unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    semester = db.Column(db.String(20), default="Unknown")
    units = db.Column(db.Integer, default=0)
    description = db.Column(db.Text, default="")

    def to_dict(self):
        return {
                "id": self.id,
                "Course Number": self.course_number,
                "Title": self.title,
                "Semester": self.semester,
                "Units": self.units,
                "Description": self.description
                }
