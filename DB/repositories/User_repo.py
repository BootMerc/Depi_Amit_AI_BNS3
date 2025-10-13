from domain.User import user
import psycopg2

class user_repo:
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()

    def create(self, user : user):
        self.cur.execute(
            "INSERT INTO users (name, age,gender,user_type) VALUES (%s,%s,%s,%s);",
            (user.name, user.age, user.gender,user.user_type)
        )
        self.conn.commit()
    