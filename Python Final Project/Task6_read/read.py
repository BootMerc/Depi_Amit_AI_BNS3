def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except IOError:
        return "Error: An error occurred while reading the file."

class UserExtractor:
    def __init__(self, file_path):
        """
        Initialize UserExtractor with file path and empty dictionary for usernames.
        """
        self.file_path = file_path
        self.usernames = {}

    def extract_usernames(self):
        """
        Reads file content using read_txt_file and extracts usernames into a dictionary.
        Each line should be in 'username:password' format.
        """
        content = read_txt_file(self.file_path)

        if content.startswith("Error:"):
            return content

        for line in content.strip().splitlines():
            if ":" in line:
                username, password = line.split(":", 1)
                username = username.strip()
                password = password.strip()
                if username:  
                    self.usernames[username] = password 

        return self.usernames

if __name__ == "__main__":
    file_path = r"C:\Users\Zyad Diab\OneDrive\Desktop\final\TASK6_read\test.txt" # adjust the path as necessary
    extractor = UserExtractor(r"C:\Users\Zyad Diab\OneDrive\Desktop\final\TASK6_read\test.txt") # adjust the path as necessary
    result = extractor.extract_usernames()

    if isinstance(result, dict):
        print("Extracted Usernames:", result)
    else:
        print(result) 
