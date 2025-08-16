# change_db_password.py
import os
import getpass
import psycopg2
from dotenv import load_dotenv

load_dotenv()  # reads .env if present

host = os.getenv("DB_HOST", "localhost")
port = int(os.getenv("DB_PORT", 5432))
user = os.getenv("POSTGRES_USER", "postgres")
db = os.getenv("POSTGRES_DB", "postgres")

print("You will be prompted for the current password, then the new password.")
old = getpass.getpass("Current DB password: ")
new = getpass.getpass("New DB password (will also update .env): ")

conn = psycopg2.connect(host=host, port=port, dbname=db, user=user, password=old)
conn.autocommit = True
cur = conn.cursor()
cur.execute("ALTER USER %s WITH PASSWORD %s;", (psycopg2.extensions.AsIs(user), new))
cur.close()
conn.close()
print("Password changed successfully. Now update your .env with the new password.")
