class Calculator:
    def __init__(self):
        self.decimal_places = 1
        self.operations = {
            '1': ('+', self.add),
            '2': ('-', self.subtract),
            '3': ('*', self.multiply),
            '4': ('/', self.divide)
        }
    def display_menu(self):
        """Display the calculator menu options."""
        print("Simple Calculator")
        print("1-Addition (+)")
        print("2-Subtraction (-)")
        print("3-Multiplication (*)")
        print("4-Division (/)")
        print("5-Exit")

    def get_numbers(self):
        """Get two numbers from user input."""
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
        return num1, num2

    def add(self, x, y):
        """Perform addition."""
        return x + y
    def subtract(self, x, y):
        """Perform subtraction."""
        return x - y
    def multiply(self, x, y):
        """Perform multiplication."""
        return x * y
    def divide(self, x, y):
        """Perform division with zero check."""
        if y == 0:
            raise ValueError("Division by zero is not allowed.")
        return x / y

    def display_result(self, num1, num2, operation, result):
        """Display the calculation result."""
        print(f"{num1} {operation} {num2} = {round(result, self.decimal_places)}\n")

    def run(self):
        """Run the calculator program."""
        while True:
            self.display_menu()
            choice = input("Enter your choice (1-5): ")

            if choice == '5':
                print("Exiting calculator. Goodbye!")
                break

            if choice not in self.operations:
                print("Invalid choice. Please select a valid option.")
                continue

            try:
                num1, num2 = self.get_numbers()
                operation_symbol, operation_func = self.operations[choice]
                result = operation_func(num1, num2)
                self.display_result(num1, num2, operation_symbol, result)
            except ValueError as e:
                print(f"Error: {e}")
                continue

# Create calculator instance and run it
calculator = Calculator()
calculator.run()

#my final answers will be rounded to nearest tenth in display