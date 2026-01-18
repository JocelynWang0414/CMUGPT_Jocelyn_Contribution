import csv
import os

from app import app, db
from db_models import Course

CSV_FILE = os.path.expanduser('~/Downloads/CMU_courses.csv')  # path to the data .csv file

def load_courses_from_csv(csv_file):
    with app.app_context():
        Course.query.delete()
        db.session.commit()
        with open(csv_file, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            # row_count = 0
            for row in reader:
                # print("Processing row:", row)
                # row_count += 1
                '''
                example)
                course_number: "39-109"
                title:         "Grand Challenge Freshman Seminar: Climate Change"
                semester:      "Fall and Spring"
                units_str:     "9"
                description:   "Climate change is considered by many..."
                '''
                course_number = row.get('Course Number', '')      
                title = row.get('Title', '')                      
                semester = row.get('Semester', 'Unknown')         
                units_str = row.get('Units', '0')                 
                description = row.get('Description', '')          
                # might want to convert units to int for computing total units
                try:
                    units = int(units_str)
                except ValueError:
                    units = 0
                existing_course = Course.query.filter_by(course_number=course_number).first()
                if existing_course:
                    continue  # Skip inserting this duplicate
                new_course = Course(
                    course_number=course_number,
                    title=title,
                    semester=semester,
                    units=units,
                    description=description
                )
                db.session.add(new_course)
            # print(f"Processed {row_count} rows.")
            db.session.commit()
        print("Courses loaded successfully!")

if __name__ == '__main__':
    load_courses_from_csv(CSV_FILE)
