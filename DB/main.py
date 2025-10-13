import psycopg2
from config import get_db_config
from repositories.Station_repo import station_repo
from repositories.Time_repo import time_repo
from repositories.User_repo import user_repo
from repositories.Trips_repo import trips_repo
from services.System_manager import system_manager
from ui.Main_app import main_app
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys



if __name__ == '__main__':
    conn = psycopg2.connect(**get_db_config())
    stn_repo = station_repo(conn)
    tim_repo = time_repo(conn)
    trp_repo = trips_repo(conn)
    use_repo = user_repo(conn)
    manager = system_manager(stn_repo,tim_repo,trp_repo,use_repo)

    app = QApplication(sys.argv)
    window = main_app(manager)
    window.show()
    sys.exit(app.exec_())