class Customer:
    def __init__(self, name, customer_id):
        self.name = name
        self.customer_id = customer_id
        self.accounts = {}

    def add_account(self, account):
        if account.account_number in self.accounts:
            print("Account already exists.")
        else:
            self.accounts[account.account_number] = account
            print(f"Account {account.account_number} added for {self.name}.")

    def get_account(self, account_number):
        return self.accounts.get(account_number)

    def list_accounts(self):
        print(f"{self.name}'s Accounts:")
        for acc_num, account in self.accounts.items():
            print(f"  - Account {acc_num}: Balance = {account.balance}")
