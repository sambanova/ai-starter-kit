import os
import sqlite3
from datetime import datetime

_DB_FILE_NAME = 'query_logs.db'


def initialize_database() -> None:
    """Create database and table if they don't exist"""
    conn = sqlite3.connect(_DB_FILE_NAME)
    c = conn.cursor()

    # Create table
    c.execute("""CREATE TABLE IF NOT EXISTS query_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME NOT NULL,
                  origin TEXT,
                  destination TEXT NOT NULL,
                  age INTEGER NOT NULL,
                  trip_duration INTEGER NOT NULL,
                  budget INTEGER)""")
    conn.commit()
    conn.close()


def log_query(origin: str, destination: str, age: int, trip_duration: int, budget: int) -> None:
    """Log a query to the database"""

    # Check if file exists, if not initialize the db
    if not os.path.isfile(_DB_FILE_NAME):
        initialize_database()

    conn = sqlite3.connect(_DB_FILE_NAME)
    c = conn.cursor()

    # Insert log record
    c.execute(
        """INSERT INTO query_logs 
                 (timestamp, origin, destination, age, trip_duration, budget)
                 VALUES (?, ?, ?, ?, ?, ?)""",
        (datetime.now(), origin, destination, age, trip_duration, budget),
    )
    conn.commit()
    conn.close()
