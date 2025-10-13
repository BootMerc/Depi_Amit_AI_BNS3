from domain.Time import time
import psycopg2

class time_repo:
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()

    def create(self, time : time):
        self.cur.execute(
            "INSERT INTO station (date, hour,day,month,year,day_of_week) VALUES (%s,%s,%s,%s,%s,%s);",
            (time.date, time.hour, time.day, time.month, time.year, time.day_of_week)
        )
        self.conn.commit()
    