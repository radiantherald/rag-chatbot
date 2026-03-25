import psycopg2
import hashlib
import os

# Configuration (Ensure these match setup_auth_db.py or use Environment Variables)
DB_HOST = "localhost"
DB_USER = "postgres"
DB_PASS = "Cokkies"
DB_NAME = "vision_rag_db"

def get_db_connection():
    """Get a connection to the application database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME
        )
        return conn
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return None

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    """Verify a stored password against one provided by user."""
    return stored_hash == hash_password(provided_password)

def login_user(username, password):
    """
    Verify user credentials.
    Returns True if valid, False otherwise.
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        
        if result:
            stored_hash = result[0]
            if verify_password(stored_hash, password):
                return True
    except Exception as e:
        print(f"Login Error: {e}")
    finally:
        if conn:
            conn.close()
    
    return False

def create_user(username, password):
    """
    Register a new user.
    Returns:
        (True, "Success Message")
        (False, "Error Message")
    """
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed."
    
    try:
        cursor = conn.cursor()
        # Check if exists
        cursor.execute("SELECT 1 FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return False, "Username already exists."
        
        # Create
        hashed = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed))
        conn.commit()
        return True, "User created successfully."
    except Exception as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()
