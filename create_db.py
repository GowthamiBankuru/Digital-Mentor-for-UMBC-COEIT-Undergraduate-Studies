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
            is_active INTEGER DEFAULT 0
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

    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if cursor.fetchone() is None:
        cursor.execute('''
            INSERT INTO users (username, email, password, role, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, hashed_password, "Admin", 1))
        conn.commit()
    conn.close()


def add_admin_user():
    username = "admin"
    password = "password123"  
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())  
    
    # Connection to SQLite database
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()

    # Check if the admin already exists in the users table
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        print("Admin user already exists, skipping insertion.")
    else:
        cursor.execute("INSERT INTO users (username, email, password, role, is_active) VALUES (?, ?, ?, ?, ?)", ("admin", "admin@example.com", hashed_password, "Admin", 1))
        conn.commit()
        print("Admin user created successfully.")
    conn.close()

# Run the functions to create the database and add the admin user (if not already present)
if __name__ == "__main__":
    create_user_db()
    add_admin_user()
    print("Database setup complete.")
