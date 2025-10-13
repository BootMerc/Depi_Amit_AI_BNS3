class trips:
    def __init__(self, start_time: str,end_time: str, duration_sec: int, start_station_id: int, end_station_id: int,user_id: int, time_id: int):
        self.start_time = start_time
        self.end_time = end_time 
        self.duration_sec = duration_sec
        self.start_station_id = start_station_id
        self.end_station_id = end_station_id
        self.user_id = user_id
        self.time_id = time_id