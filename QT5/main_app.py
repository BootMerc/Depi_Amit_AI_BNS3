from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
import sys
import time
import psycopg2
 
class Main_App(QMainWindow):
    def __init__(self):
        super(Main_App,self).__init__()
        uic.loadUi(r"C:\Users\Zyad Diab\OneDrive\Desktop\QT\std.ui",self)
        self.tabWidget.tabBar().setVisible(False)
        self.InitUI()
        self.handle_btn()
        self.handle_db_conn()
    def InitUI(self):
        self.setWindowTitle("Student System")
   
    def handle_db_conn(self):
        self.db = psycopg2.connect( host="localhost",
                               database="students",
                               user="postgres",
                               password="Lenovo123!")
        self.curr = self.db.cursor()
        print("Connection is Done!")
   
       
    def handle_btn(self):
        ############################student_btn#######################
        self.std_add_btn.clicked.connect(self.add_std_info)
        self.std_update_btn.clicked.connect(self.update_std_info)
        self.std_del_btn.clicked.connect(self.del_std_info)
       
        ######################## Enrollment btn  #################################
       
        # self.std_add_btn.clicked.connect(self.add_std_info)
        # self.std_update_btn.clicked.connect(self.update_std_info)
        # self.std_del_btn.clicked.connect(self.del_std_info)
       
       
 ###################################################################      
       
       
    def add_std_info(self):
        std_id = self.std_Id_txt.text()
        std_name = self.std_name_txt.text()
        std_email = self.std_email_txt.text()
        std_phone = self.std_phone_txt.text()
        # comb_box = self.cmb_type.currentIndex()
        self.curr.execute('INSERT INTO students (student_id, name, email, phone) VALUES(%s,%s,%s,%s)',
                         ( std_id, std_name,std_email,std_phone))
        self.db.commit()
    def update_std_info(self):
        print('welcome to my app')
    def del_std_info(self):
        print('welcome to my app')    
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Main_App()
    window.show()
    sys.exit(app.exec_())