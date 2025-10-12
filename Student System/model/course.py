# from system_manager import *
#
class Course:
    _id_counter = 1
    def __init__(self, name):
        self.course_id = Course._id_counter
        Course._id_counter += 1
        self.name = name
        self.enrolled_students = []
        
        
    def __str__(self):
        return f"Course ID: {self.course_id}, Name: {self.name}, Enrolled: {len(self.enrolled_students)}"

    def enroll_student(self, student):
        if student not in self.enrolled_students:
            self.enrolled_students.append(student)
            print("Student enrolled successfully.")
        else:
            print("Student already enrolled.")

    def remove_student(self, student):
        if student in self.enrolled_students:
            self.enrolled_students.remove(student)

