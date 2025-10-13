from domain.Station import station
import psycopg2

class station_repo:
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()

    def create(self, station : station):
        self.cur.execute(
            "INSERT INTO station (station_name, latitude,longitude) VALUES (%s,%s,%s);",
            (station.station_name, station.latitude, station.longitude)
        )
        self.conn.commit()