import psycopg2
from .config import DB

def get_conn():
    return psycopg2.connect(
        host=DB["host"], port=DB["port"], dbname=DB["dbname"],
        user=DB["user"], password=DB["password"],
        sslmode=DB["sslmode"], connect_timeout=DB["connect_timeout"],
    )
