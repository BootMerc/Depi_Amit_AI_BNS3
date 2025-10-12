class Calculator:
    def add(self, a, b):
        return a + b
    def subtract(self, a, b):
        return a - b
    def multiply(self, a, b):
        return a * b
    def divide(self, a, b):
        if b != 0:
            return a / b
        else:
            return "Error! Division by zero."
    def run(self):
        print("Welcome to the calculator!")
        print("Choose an operation:")
        print("1: Add")
        print("2: Subtract")
        print("3: Multiply")
        print("4: Divide")
        choice = input("Enter the number of the operation you want to perform (1/2/3/4): ")

        if choice in ['1', '2', '3', '4']:
            try:
                num1 = float(input("Enter the first number: "))
                num2 = float(input("Enter the second number: "))
            except ValueError:
                print("Invalid input! Please enter numbers only.")
                return
            if choice == '1':
                result = self.add(num1, num2)
                print(f"{num1} + {num2} = {round(result, 1)}")
            elif choice == '2':
                result = self.subtract(num1, num2)
                print(f"{num1} - {num2} = {round(result, 1)}")
            elif choice == '3':
                result = self.multiply(num1, num2)
                print(f"{num1} * {num2} = {round(result, 1)}")
            elif choice == '4':
                result = self.divide(num1, num2)
                if isinstance(result, str):
                    print(result)  # Error message
                else:
                    print(f"{num1} / {num2} = {round(result, 1)}")
        else:
            print("Invalid choice! Please select a valid operation (1/2/3/4).")

if __name__ == "__main__":
    calc = Calculator()
    calc.run()
