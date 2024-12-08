import sqlite3
import bcrypt

def create_user_db():
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            is_active INTEGER DEFAULT 0,
            verification_code TEXT DEFAULT NULL,       
            permissions TEXT DEFAULT NULL
        )
    ''')
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN permissions TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        # Column already exists, skip adding it
        pass
    conn.commit()
    conn.close()

def create_activity_log_db():
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def populate_activity_log():
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()

    # Sample data for user_activity
    sample_data = [
        ("user1", "Logged in"),
        ("user2", "Accessed System Performance Dashboard"),
        ("user3", "Accessed Admin Management"),
        ("user1", "Logged out"),
        ("user2", "Logged in"),
        ("user1", "Accessed Course Recommender")
    ]

    # Insert sample data
    for username, action in sample_data:
        cursor.execute('''
            INSERT INTO user_activity (username, action)
            VALUES (?, ?)
        ''', (username, action))
    
    conn.commit()
    conn.close()
    print("Sample activity log data inserted.")

def create_course_and_student_tables():
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()
    
    # Create courses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_name TEXT NOT NULL,
            description TEXT,
            credits INTEGER NOT NULL
        )
    ''')
    
    # Create student records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            advisor TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def add_admin_user():
    username = "admin"
    email = "admin@example.com"
    password = "password123"
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()

    # Check if the admin user already exists
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone() is None:
        cursor.execute('''
            INSERT INTO users (username, email, password, role, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, hashed_password, "Admin", 1))
        conn.commit()
        print("Admin user created successfully.")
    else:
        print("Admin user already exists, skipping insertion.")
    conn.close()

if __name__ == "__main__":
    # Create database tables
    create_user_db()
    create_activity_log_db()
    create_course_and_student_tables()
    
    # Add default admin user
    add_admin_user()
    
    # Populate user_activity table with sample data
    populate_activity_log()
    
    print("Database setup complete.")
