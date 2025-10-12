CREATE TABLE Groups(
	group_id SERIAL PRIMARY KEY NOT NULL,
	name VARCHAR(50) NOT NULL
);
CREATE TABLE Students(
	Student_id SERIAL PRIMARY KEY NOT NULL,
	First_name VARCHAR(10) NOT NULL,
	Last_name VARCHAR(10) NOT NULL,
	group_id SERIAL REFERENCES Groups(group_id)
);
CREATE TABLE Teachers(
	Teacher_id SERIAL PRIMARY KEY NOT NULL,
	First_name VARCHAR(10) NOT NULL,
	Last_name VARCHAR(10) NOT NULL
);
CREATE TABLE Subjects(
	Subject_id SERIAL PRIMARY KEY NOT NULL,
	Title VARCHAR(20) NOT NULL
);
CREATE TABLE Subject_teacher(
	Subject_id SERIAL REFERENCES Subjects(Subject_id),
	Teacher_id SERIAL REFERENCES Teachers(Teacher_id),
	group_id SERIAL REFERENCES Groups(group_id),
	PRIMARY KEY (Subject_id,Teacher_id,group_id)
);
CREATE TABLE Marks(
	Mark_id SERIAL PRIMARY KEY NOT NULL,
	Student_id SERIAL REFERENCES Students(Student_id),
	Subject_id SERIAL REFERENCES Subjects(Subject_id),
	Date TIMESTAMP,
	Mark INT
);

INSERT INTO Groups(name) VALUES
('Group A'),
('Group B'),
('Group C');
SELECT * FROM Groups

INSERT INTO Students(First_name, Last_name, group_id) VALUES
('Alice', 'Smith', 1),
('Bob', 'Johnson', 1),
('Charlie', 'Brown', 2),
('David', 'Lee', 3),
('Eva', 'Clark', 3),
('Liam', 'Williams', 1),
('Noah', 'Davis', 2),
('Oliver', 'Martin', 2),
('Emma', 'Wilson', 3),
('Ava', 'Taylor', 1),
('Sophia', 'Anderson', 2),
('Mia', 'Thomas', 3),
('Lucas', 'Jackson', 1),
('Ethan', 'White', 3),
('Amelia', 'Harris', 2);
SELECT * FROM Students

INSERT INTO Teachers(First_name, Last_name) VALUES
('John', 'Doe'),
('Jane', 'Miller'),
('Emily', 'Stone');
SELECT * FROM Teachers

INSERT INTO Subjects(Title) VALUES
('Math'),
('Physics'),
('Biology'),
('Chemistry'),
('History'),
('English'),
('Computer Science');
SELECT * FROM Subjects

INSERT INTO Subject_teacher(Subject_id, Teacher_id, group_id) VALUES
(1, 1, 1),-- John teaches Math to Group A
(2, 2, 2),-- Jane teaches Physics to Group B
(3, 3, 3),-- Emily teaches Biology to Group C
(1, 1, 2),-- John teaches Math to Group B too
(2, 2, 3);-- Jane teaches Physics to Group C
SELECT * FROM Subject_teacher

INSERT INTO Marks(Student_id, Subject_id, Date, Mark) VALUES
(1, 1, '2025-08-01 09:00:00', 85), -- Alice - Math
(2, 1, '2025-08-02 09:00:00', 78), -- Bob - Math
(3, 2, '2025-08-03 09:00:00', 90), -- Charlie - Physics
(4, 3, '2025-08-04 09:00:00', 88), -- David - Biology
(5, 3, '2025-08-05 09:00:00', 92), -- Eva - Biology
(6, 1, '2025-07-10 10:00:00', 88),
(6, 5, '2025-07-20 10:00:00', 76),
(7, 1, '2025-07-15 11:00:00', 81),
(7, 2, '2025-07-22 11:00:00', 69),
(8, 1, '2025-07-12 12:00:00', 72),
(8, 2, '2025-07-28 12:00:00', 90),
(9, 3, '2025-07-18 09:00:00', 95),
(9, 2, '2025-07-30 09:30:00', 85),
(10, 1, '2025-07-09 10:30:00', 92),
(10, 5, '2025-07-25 11:00:00', 87),
(11, 1, '2025-08-01 09:15:00', 84),
(11, 2, '2025-08-05 09:45:00', 91),
(12, 3, '2025-08-03 10:00:00', 89),
(12, 2, '2025-08-06 10:30:00', 77),
(13, 1, '2025-08-01 11:00:00', 90),
(13, 5, '2025-08-07 11:30:00', 78),
(14, 3, '2025-07-31 10:00:00', 73),
(14, 2, '2025-08-08 10:30:00', 80),
(15, 1, '2025-07-19 09:00:00', 91),
(15, 2, '2025-07-27 09:30:00', 88);
SELECT * FROM Marks

--makes u list all stdents with a group name
SELECT s.Student_id, s.First_name, s.Last_name, g.name AS group_name
FROM Students s
JOIN Groups g ON s.group_id = g.group_id;

--shows u all the teachers and thier subjects
SELECT t.First_name || ' ' || t.Last_name AS Teacher, sub.Title AS Subject, g.name AS Group
FROM Subject_teacher st
JOIN Teachers t ON st.Teacher_id = t.Teacher_id
JOIN Subjects sub ON st.Subject_id = sub.Subject_id
JOIN Groups g ON st.group_id = g.group_id;

-- marks of the students in their subjects
SELECT s.First_name || ' ' || s.Last_name AS Student, sub.Title AS Subject, m.Mark, m.Date
FROM Marks m
JOIN Students s ON m.Student_id = s.Student_id
JOIN Subjects sub ON m.Subject_id = sub.Subject_id
ORDER BY m.Date;

--this will calculate the average mark of students
SELECT sub.Title AS Subject, AVG(m.Mark) AS Average_Mark
FROM Marks m
JOIN Subjects sub ON m.Subject_id = sub.Subject_id
GROUP BY sub.Title;

--to students who got more than 90 (A)
SELECT s.First_name || ' ' || s.Last_name AS Student, sub.Title, m.Mark
FROM Marks m
JOIN Students s ON m.Student_id = s.Student_id
JOIN Subjects sub ON m.Subject_id = sub.Subject_id
WHERE m.Mark > 90;

--a query to search or a specific student
SELECT DISTINCT sub.Title AS Subject
FROM Marks m
JOIN Subjects sub ON m.Subject_id = sub.Subject_id
JOIN Students s ON m.Student_id = s.Student_id
WHERE s.First_name = 'Ava' AND s.Last_name = 'Taylor';

