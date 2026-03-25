import psycopg2
from psycopg2 import sql
import hashlib
import os

# Configuration
DB_HOST = "localhost"
DB_USER = "postgres"
DB_PASS = "Cokkies" # Updated based on user input
TARGET_DB_NAME = "vision_rag_db" # dedicated DB for this app

def get_connection(dbname=None):
    """Establish a connection to the database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        dbname=dbname if dbname else "postgres" # Connect to default 'postgres' db to create new db
    )
    conn.autocommit = True
    return conn

def create_database():
    """Create the target database if it doesn't exist."""
    print(f"Connecting to Postgres to check for '{TARGET_DB_NAME}' database...")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if DB exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (TARGET_DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Database '{TARGET_DB_NAME}' not found. Creating...")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TARGET_DB_NAME)))
            print(f"Database '{TARGET_DB_NAME}' created successfully.")
        else:
            print(f"Database '{TARGET_DB_NAME}' already exists.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")

def create_users_table():
    """Create the users table in the target database."""
    print(f"Connecting to '{TARGET_DB_NAME}' to create tables...")
    try:
        conn = get_connection(dbname=TARGET_DB_NAME)
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            username VARCHAR(50) PRIMARY KEY,
            password_hash VARCHAR(256) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        print("Table 'users' verified/created.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating table: {e}")

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_admin_user():
    """Create a default admin user if it doesn't exist."""
    try:
        conn = get_connection(dbname=TARGET_DB_NAME)
        cursor = conn.cursor()
        
        username = "admin"
        password = "admin_password" # Change this!
        
        cursor.execute("SELECT 1 FROM users WHERE username = %s", (username,))
        if not cursor.fetchone():
            hashed = hash_password(password)
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed))
            print(f"Created default user: '{username}' (Password: '{password}')")
        else:
            print(f"User '{username}' already exists.")
            
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating admin user: {e}")

if __name__ == "__main__":
    print("--- Setting up Auth Database ---")
    create_database()
    create_users_table()
    create_admin_user()
    print("--- Setup Complete ---")
