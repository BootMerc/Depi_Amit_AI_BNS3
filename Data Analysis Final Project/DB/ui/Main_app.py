from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from PyQt5.QtCore import QTimer
import sys
import time
from domain.Station import station
from domain.Time import time
from domain.Trips import trips
from domain.User import user
from services.System_manager import system_manager

class main_app(QMainWindow):
    def __init__(self,manger : system_manager):
        super(main_app, self).__init__()
        uic.loadUi('C:/Users/6y6yy/Desktop/Depi_Team/data_analysis_final_project/DB/ui/GoBike.ui', self)
        self.manger = manger
        self.InitUI()
        self.handle_btn()
    
    def InitUI(self):
        self.setWindowTitle("GoBike Management System")
        self.statusBar().showMessage("Welcome to the GoBike Management System")
    
    def handle_btn(self):
        self.stn_add_btn.clicked.connect(self.add_stn_info)
        self.tme_add_btn.clicked.connect(self.add_tme_info)
        self.trp_add_btn.clicked.connect(self.add_trp_info)
        self.usr_add_btn.clicked.connect(self.add_usr_info)
    
    def add_stn_info(self):
        stn_name = self.stn_name_text.text()
        stn_latitude = self.stn_ltd_text.text()
        stn_longitude = self.stn_lng_text.text()
        if stn_name and stn_latitude and stn_longitude:
            stn = station(stn_name, stn_latitude, stn_longitude)
            self.manger.add_station(stn)
            self.stn_name_text.setText("")
            self.stn_ltd_text.setText("")
            self.stn_lng_text.setText("")
            self.statusBar().showMessage("Station added successfully")
        else:
            self.statusBar().showMessage("Please fill all fields for Station")

    def add_tme_info(self):
        time_date = self.tme_date_text.text()
        tme_hour = self.tme_hour_text.text()
        tme_day = self.tme_day_text.text()
        tme_month = self.tme_month_text.text()
        tme_year = self.tme_year_text.text()
        time_day_of_week = self.tme_day_of_week_text.text()
        if tme_hour and time_date and tme_day and tme_month and tme_year and time_day_of_week:
            tme = time(time_date, tme_hour, tme_day, tme_month, tme_year, time_day_of_week)
            self.manger.add_time(tme)
            self.tme_date_text.setText("")
            self.tme_hour_text.setText("")
            self.tme_day_text.setText("")
            self.tme_month_text.setText("")
            self.tme_year_text.setText("")
            self.tme_day_of_week_text.setText("")
            self.statusBar().showMessage("Time added successfully")
        else:
            self.statusBar().showMessage("Please fill all fields for Time")
    
    def add_trp_info(self):
        trp_start_time = self.trp_start_time_text.text()
        trp_end_time = self.trp_end_time_text.text()
        trp_duration_sec = self.trp_duration_sec_text.text()
        trp_start_stn_id = self.trp_start_stn_id_text.text()
        trp_end_stn_id = self.trp_end_stn_id_text.text()
        trp_user_id = self.trp_user_id_text.text()
        trp_time_id = self.trp_time_id_text.text()
        if trp_start_time and trp_end_time and trp_duration_sec and trp_start_stn_id and trp_end_stn_id and trp_user_id and trp_time_id:
            trp = trips(trp_start_time, trp_end_time, trp_duration_sec, trp_start_stn_id, trp_end_stn_id, trp_user_id, trp_time_id)
            self.manger.add_trip(trp)
            self.trp_start_time_text.setText("")
            self.trp_end_time_text.setText("")
            self.trp_duration_sec_text.setText("")
            self.trp_start_stn_id_text.setText("")
            self.trp_end_stn_id_text.setText("")
            self.trp_user_id_text.setText("")
            self.trp_time_id_text.setText("")
            self.statusBar().showMessage("Trip added successfully")
        else:
            self.statusBar().showMessage("Please fill all fields for Trip")
    def add_usr_info(self):
        usr_name = self.usr_name_text.text()
        usr_age = self.usr_age_text.text()
        usr_gender = self.usr_gender_text.text()
        usr_type = self.usr_type_text.text()
        if usr_name and usr_age and usr_gender and usr_type:
            usr = user(usr_name, usr_age,usr_gender, usr_type)
            self.manger.add_user(usr)
            self.usr_name_text.setText("")
            self.usr_age_text.setText("")
            self.usr_gender_text.setText("")
            self.usr_type_text.setText("")
            self.statusBar().showMessage("User added successfully")
        else:
            self.statusBar().showMessage("Please fill all fields for User")
        
        

