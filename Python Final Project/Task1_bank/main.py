from acc import BankAccount
from customers import Customer

def main():
    # Create a customer
    customer1 = Customer("zoz", "001")

    # Create two accounts
    account1 = BankAccount("1001", 500)
    account2 = BankAccount("1002", 300)

    # Add accounts to the customer
    customer1.add_account(account1)
    customer1.add_account(account2)

    # Operate on an account
    acc = customer1.get_account("1001")
    if acc:
        acc.deposit(200)
        acc.withdraw(100)
        acc.check_balance()

    # List all accounts
    customer1.list_accounts()

if __name__ == "__main__":
    main()
