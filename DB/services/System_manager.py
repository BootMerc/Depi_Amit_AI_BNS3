from domain.Station import station
from domain.Time import time
from domain.Trips import trips
from domain.User import user
from repositories.Station_repo import station_repo
from repositories.Time_repo import time_repo
from repositories.Trips_repo import trips_repo
from repositories.User_repo import user_repo
class system_manager:
    def __init__(self, station_repo,time_repo, trips_repo, user_repo):
        self.station_repo = station_repo
        self.time_repo = time_repo
        self.trips_repo = trips_repo
        self.user_repo = user_repo
    def add_station(self, station : station):
        self.station_repo.create(station)
    def add_time(self, time : time):
        self.time_repo.create(time)
    def add_trips(self, trips : trips):
        self.trips_repo.create(trips)
    def add_user(self, user : user):
        self.user_repo.create(user)
    