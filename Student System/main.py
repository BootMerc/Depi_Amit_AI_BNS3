from .core.system_manager import SystemManager  # Import the main manager class to handle students and courses

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))


# Function to display the main menu
def show_menu():
    print("1. Add student")
    print("2. Remove student")
    print("3. Add course")
    print("4. Remove course")
    print("5. Search courses")
    print("6. Record grade")
    print("7. Get all students")
    print("8. Get all courses")
    print("9. Enroll course")
    print("10. Exit")
    
# Add new student
def add_student_1(manager):
    name = input("Enter student name: ")  # Ask for student name
    student_id = manager.add_student(name)  # Add student and get ID
    print("Student ID:", student_id)  # Display the generated ID
    print("\n" + "=" * 40)  # Visual separator

# Remove existing student by ID
def remove_student(manager):
    student_id = int(input("Enter student ID: "))  # Ask for student ID
    manager.remove_student(student_id)  # Remove student from system
    print("\n" + "=" * 40)

# Add new course
def add_course(manager):
    name = input("Enter course name: ")  # Ask for course name
    course_id = manager.add_course(name)  # Add course and get ID
    print("Course ID:", course_id)
    print("\n" + "=" * 40)

# Remove existing course
def remove_course(manager):
    course_id = int(input("Enter course ID: "))  # Ask for course ID
    manager.remove_course(course_id)  # Remove course from system
    print("\n" + "=" * 40)

# Search for courses by name
def search_courses(manager):
    search_name = input("Enter course name to search: ")
    courses = manager.search_courses(search_name)  # Search and return matching courses
    for course in courses:
        print(course)  # Print each matched course
    print("\n" + "=" * 40)

# Record a grade for a student in a course
def record_grade(manager):
    student_id = int(input("Enter student ID: "))
    course_id = int(input("Enter course ID: "))
    grade = input("Enter grade: ")
    manager.record_grade(student_id, course_id, grade)  # Save grade in system
    print("\n" + "=" * 40)

# Display all students in the system
def get_all_students(manager):
    students = manager.get_all_students()
    for student in students:
        print(student)
    print("\n" + "=" * 40)

# Display all courses in the system
def get_all_courses(manager):
    courses = manager.get_all_courses()
    for course in courses:
        print(course)
    print("\n" + "=" * 40)

# Enroll a student in a specific course
def enroll_course(manager):
    student_id = int(input("Enter student ID: "))
    course_id = int(input("Enter course ID: "))
    manager.enroll_course(student_id, course_id)
    print("\n" + "=" * 40)

# Main function: entry point of the application
def core():
    manager = SystemManager()  # Initialize the system manager instance
    while True:
        show_menu()  # Display menu options
        choice = input("Enter choice: ")  # Get user choice

        # Call the corresponding function based on choice
        if choice == '1':
            add_student_1(manager)
        elif choice == '2':
            remove_student(manager)
        elif choice == '3':
            add_course(manager)
        elif choice == '4':
            remove_course(manager)
        elif choice == '5':
            search_courses(manager)
        elif choice == '6':
            record_grade(manager)
        elif choice == '7':
            get_all_students(manager)
        elif choice == '8':
            get_all_courses(manager)
        elif choice == '9':
            enroll_course(manager)
        elif choice == '10':
            print("Exiting...")  # Exit message
            break  # Terminate the loop
        else:
            print("Invalid choice.")  # Handle wrong input
            print("\n" + "=" * 40)

# This makes sure main() runs only when the file is run directly
if __name__ == "__main__":
    core()
