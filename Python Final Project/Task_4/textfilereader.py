class TextFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_file(self):
        try:
            with open(self.file_path, 'r') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return f"Error: The file '{self.file_path}' does not exist."
    
    def count_lines(self):
        try:
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
            return len(lines)
        except FileNotFoundError:
            return f"Error: The file '{self.file_path}' does not exist."
    
    def count_words(self):
        try:
            with open(self.file_path, 'r') as file:
                content = file.read()
            words = content.split()
            return len(words)
        except FileNotFoundError:
            return f"Error: The file '{self.file_path}' does not exist."
    
    def count_characters(self):
        try:
            with open(self.file_path, 'r') as file:
                content = file.read()
                content = content.replace(' ', '')
            return len(content)
        except FileNotFoundError:
            return f"Error: The file '{self.file_path}' does not exist."
    def display_content(self):
        content = self.read_file()
        print(content)

file = TextFileReader('task_4/test.text')
print(file.count_characters())
    
