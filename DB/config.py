import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "hospital_management_system")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))

def get_db_config():
    return {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT
    }
