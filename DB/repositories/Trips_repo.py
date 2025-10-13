from domain.Trips import trips
import psycopg2

class trips_repo:
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()

    def create(self, trips : trips):
        self.cur.execute(
            "INSERT INTO trips (start_time,end_time, duration_sec, start_station_id, end_station_id,user_id, time_id) VALUES (%s,%s,%s,%s,%s,%s,%s);",
            (trips.start_time, trips.end_time, trips.duration_sec, trips.start_station_id, trips.end_station_id, trips.user_id, trips.time_id)
        )
        self.conn.commit()
    