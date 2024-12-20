import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
from scipy.stats import skew
import sqlite3
import bcrypt
from dotenv import load_dotenv
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variables
feedback_dataframe = pd.DataFrame(columns=['Name', 'Mail', 'course_suggested', 'Feedback', 'Transcript'])
admission_confidence_dataframe = pd.DataFrame(columns=['Name', 'Confidence'])
courses_confidence_dataframe = pd.DataFrame(columns=['Name', 'Confidence'])

import sqlite3
import bcrypt
import streamlit as st

if "users_db" not in st.session_state:
    st.session_state.users_db = []  

# Initialize Session State
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Intro Page"  

# Function to logout
def logout():
    st.session_state['is_logged_in'] = False
    st.session_state['role'] = None
    st.session_state['current_page'] = "Login"

# Sidebar menu for navigation
def sidebar_menu():
    if 'is_logged_in' in st.session_state and st.session_state['is_logged_in']:
        # Logout button for logged-in users
        st.sidebar.button("Logout", on_click=logout)
        
        # Define accessible pages based on role
        if st.session_state['role'] in ["Admin", "System Administrator", "Program Director", "Academic Advisor"]:
            accessible_pages = {
                "My Home": admin,
                "System Performance Dashboard": system_performance_dashboard,
                "Admin Management": admin_management,
                "Admission Checker": admission_selection,
                "Course Recommender": course_recommendation,
                "Academic Progress": academic_progress,
                "Career Accelerator": career_opportunities,
            }
            if st.session_state['role'] == "Admin":
                accessible_pages["Create User"] = create_user  # Only for Admins
        else:
            st.error("Access denied. Your role does not have permissions to access any features.")
            return None
    else:
        # Pages accessible to non-logged-in users
        accessible_pages = {
            "Intro Page": intro,
            "Login": login,
            "Sign Up": signup,
            "Admission Checker": admission_selection,
            "Course Recommender": course_recommendation,
            "Career Accelerator": career_opportunities,
        }

    # Filtering out None pages
    accessible_pages = {k: v for k, v in accessible_pages.items() if v is not None}

   # Initialize session state for current page
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = list(accessible_pages.keys())[0]

    # Sidebar styling and navigation
    CSS = """
    <style>
    .sidebar .sidebar-content {
        background-color: #00457c; /* UMBC dark blue */
        color: #ffffff;
    }
    .sidebar .block-container {
        padding: 0;
    }
    .sidebar .menu-item {
        margin: 10px 0;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #ffffff;
        background-color: #00457c; /* UMBC dark blue */
        transition: background-color 0.3s ease, color 0.3s ease;
        cursor: pointer;
    }
    .sidebar .menu-item:hover {
        background-color: #ffc904; /* UMBC gold */
        color: #00457c;
    }
    .sidebar .menu-item.selected {
        background-color: #ffc904; /* UMBC gold */
        color: #00457c;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    </style>
    """
    st.markdown(CSS, unsafe_allow_html=True)

    # Display navigation options
    selected_page = None
    st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    for option, func in accessible_pages.items():
        is_selected = st.session_state["current_page"] == option

        # Update session state when a menu item is clicked
        if st.sidebar.button(option, key=option, use_container_width=True):
            st.session_state["current_page"] = option
            selected_page = func

        # Highlight selected page
        if is_selected:
            selected_page = func

    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    return selected_page

# Function for the intro Page
def intro():
    # Load Images
    umbc_logo = Image.open("/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/Icons/umbc.png")  
    digital_mentor_icon = Image.open("/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/Icons/digital_mentor.jpg")  
    admission_prediction_icon = Image.open("/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/Icons/admission_prediction.png")  
    course_recommendation_icon = Image.open("/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/Icons/course_recommendation.png")  
    academic_progress_icon = Image.open("/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/Icons/academic_progress.jpg") 
    activity_icon =  Image.open("/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/Icons/activity.png")
    admin_icon = Image.open("/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/Icons/admin.png")  

    # Set up layout
    st.image(umbc_logo, width=200)
    st.title("UMBC COEIT Undergraduate Studies Course Recommender")
    st.markdown("""
        Welcome to the UMBC COEIT Undergraduate Studies Course Recommender, your personalized guide to navigating the academic programs and opportunities offered at UMBC's College of Engineering and Information Technology.
        
        **With Smart Program Advisor - A Digital Mentor, you will be able to:**
        - Explore recommended courses based on your interests and career goals
        - Check your likelihood of admission to UMBC
        - Track academic progress
        - Receive personalized guidance from our digital mentor

        **Select a tool from the sidebar to get started!
    """)

    # Tool summary
    col1, col2 = st.columns(2)
    with col1:
        st.image(digital_mentor_icon, width=100)
        st.markdown("### Digital Mentor")
        st.write("Personalized course recommendations based on your interests.")
        
        st.image(admission_prediction_icon, width=100)
        st.markdown("### Admission Predictor")
        st.write("Estimate your chances of being admitted to UMBC based on academic data.")

        st.image(course_recommendation_icon, width=100)
        st.markdown("### Course Recommender")
        st.write("Explore courses aligned with your academic goals.")
    
    with col2:
        st.image(academic_progress_icon, width=100)
        st.markdown("### Academic Progress")
        st.write("Monitor student's academic journey and track progress.")

        st.image(activity_icon, width=100)
        st.markdown("### Career Accelerator")
        st.write("Discover relevant internships, workshops aligns with your career goals.")
        
        st.image(admin_icon, width=100)
        st.markdown("### Admin Dashboard")
        st.write("Analyze user feedback and system performance for continuous improvement.")

    st.sidebar.success("Select a tool above.")

# Function for the Login Page
def login():
    if st.session_state['is_logged_in']:
        st.warning("You are already logged in.")
        st.session_state['current_page'] = "My Home"
        return
    st.title("Login Page-Admins")
    username = st.text_input(label="Username", placeholder="Enter your username", label_visibility="collapsed")
    password = st.text_input(label="Password", placeholder="Enter your password", type="password", label_visibility="collapsed")

    if st.button("Login"):
        role = check_user_credentials(username, password)
        if role:
            log_user_action(username, "Logged in")
            st.session_state['role'] = role
            st.session_state['is_logged_in'] = True
            if role in ["Admin", "System Administrator", "Program Director", "Academic Advisor"]:  
                st.session_state['is_admin'] = True
            else:
                st.session_state['is_admin'] = False

            st.success(f"Login successful as {role}!")
            st.session_state['current_page'] = "My Home"
        else:
            st.error("Invalid credentials or account inactive.")

# Function to log user activity
def log_user_action(username, action):
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_activity (username, action)
        VALUES (?, ?)
    ''', (username, action))
    conn.commit()
    conn.close()

# Function to check user credentials from the database
def check_user_credentials(username, password):
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT password, role, is_active
        FROM users
        WHERE username = ?
    ''', (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        stored_password, role, is_active = result
        if is_active == 1 and bcrypt.checkpw(password.encode('utf-8'), stored_password):
            return role
    return None

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import bcrypt
import plotly.express as px

# Define functions for new user stories
def system_performance_dashboard():
    st.title("System Performance Dashboard")
    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()

    # Active users count
    cursor.execute("SELECT COUNT(DISTINCT username) FROM user_activity")
    active_users = cursor.fetchone()[0]

    # System access frequency
    cursor.execute("SELECT COUNT(*) FROM user_activity")
    total_actions = cursor.fetchone()[0]

    # Display metrics
    st.metric("Number of Active Users", active_users)
    st.metric("System Access Frequency", total_actions)

    # Activity trends
    cursor.execute("""
        SELECT DATE(timestamp) as access_date, COUNT(*) as access_count
        FROM user_activity
        GROUP BY access_date
        ORDER BY access_date
    """)
    activity_data = pd.DataFrame(cursor.fetchall(), columns=["Access Date", "Access Count"])
    if not activity_data.empty:
        fig = px.line(activity_data, x="Access Date", y="Access Count", title="System Access Trends")
        st.plotly_chart(fig)
    else:
        st.write("No activity data available.")

    conn.close()

def admin_management():
    st.title("Admin Management")
    tabs = st.tabs(["Manage Courses", "Manage Student Records", "Manage Advisor Permissions"])

    conn = sqlite3.connect('user_db.db')
    cursor = conn.cursor()

    # Manage Courses
    with tabs[0]:
        st.subheader("Manage Courses")
        with st.form("add_course"):
            course_name = st.text_input("Course Name")
            description = st.text_area("Description")
            credits = st.number_input("Credits", min_value=1, max_value=10, step=1)
            if st.form_submit_button("Add Course"):
                cursor.execute("INSERT INTO courses (course_name, description, credits) VALUES (?, ?, ?)",
                               (course_name, description, credits))
                conn.commit()
                st.success("Course added successfully.")
        cursor.execute("SELECT * FROM courses")
        courses = pd.DataFrame(cursor.fetchall(), columns=["ID", "Course Name", "Description", "Credits"])
        st.dataframe(courses)

    # Manage Student Records
    with tabs[1]:
        st.subheader("Manage Student Records")
        with st.form("add_student"):
            student_name = st.text_input("Student Name")
            email = st.text_input("Email")
            advisor = st.text_input("Advisor")
            if st.form_submit_button("Add Student"):
                cursor.execute("INSERT INTO student_records (student_name, email, advisor) VALUES (?, ?, ?)",
                               (student_name, email, advisor))
                conn.commit()
                st.success("Student record added successfully.")
        cursor.execute("SELECT * FROM student_records")
        students = pd.DataFrame(cursor.fetchall(), columns=["ID", "Student Name", "Email", "Advisor"])
        st.dataframe(students)

    # Manage Advisor Permissions
    with tabs[2]:
        st.subheader("Manage Advisor Permissions")
        cursor.execute("SELECT id, username, permissions FROM users WHERE role='Academic Advisor'")
        advisors = pd.DataFrame(cursor.fetchall(), columns=["ID", "Username", "Permissions"])
        st.dataframe(advisors)
        advisor_id = st.number_input("Enter Advisor ID to update permissions", step=1, format="%d")
        new_permissions = st.text_input("Enter New Permissions")
        if st.button("Update Permissions"):
            cursor.execute("UPDATE users SET permissions=? WHERE id=?", (new_permissions, advisor_id))
            conn.commit()
            st.success("Permissions updated successfully.")

    conn.close()

import os
import random
import string
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

def sanitize_string(input_string):
    """Remove non-ASCII characters from a string."""
    return ''.join(char for char in input_string if ord(char) < 128)

# Retrieve and sanitize the environment variable
raw_password = os.getenv('GMAIL_APP_PASSWORD')
sanitized_password = sanitize_string(raw_password)

def generate_verification_code():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

def send_verification_email(email, code):
    sender_email = "bankurugowthami5@gmail.com"
    sender_password = os.getenv("GMAIL_APP_PASSWORD")  # Fetch password securely  
    subject = "Complete Your Digital Mentor Account Setup"
    message = f"""
    <html>
    <body>
        <h2 style="color:#00457c;">Welcome to Digital Mentor!</h2>
        <p>
            You're just one step away from completing your account setup. Use the verification code below to activate your account:
        </p>
        <h3 style="color:#ffc904;">{code}</h3>
        <p>
            If you didn't sign up for Digital Mentor, please ignore this email.
        </p>
        <p>
            Best regards,<br>
            <strong>The Digital Mentor Team</strong><br>
        </p>
    </body>
    </html>
    """

    msg = MIMEText(message, "html") 
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
        print(f"Verification email sent to {email}.")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Function for Sign-Up Page
def signup():
    st.title("Sign Up")

    with st.form("signup_form"):
        email = st.text_input(label="Email", placeholder="Enter your email", label_visibility="collapsed")
        username = st.text_input(label="Username", placeholder="Enter your username", label_visibility="collapsed")
        verification_code_input = st.text_input(label="Verification Code", placeholder="Enter your verification code", label_visibility="collapsed")
        password = st.text_input(label="Password", placeholder="Create a password", type="password", label_visibility="collapsed")
        confirm_password = st.text_input(label="Confirm Password", placeholder="Confirm your password", type="password", label_visibility="collapsed")
        submitted = st.form_submit_button("Sign Up")

    if submitted:
        if not username:
            st.error("Username is required.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            conn = sqlite3.connect('user_db.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, verification_code, is_active FROM users WHERE email = ? OR username = ?
            ''', (email, username))
            user = cursor.fetchone()

            if user:
                user_id, stored_code, is_active = user
                if is_active == 1:
                    st.error("Account already active. Please log in.")
                elif verification_code_input != stored_code:
                    st.error("Invalid verification code.")
                else:
                    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                    cursor.execute('''
                        UPDATE users SET username = ?, password = ?, is_active = 1, verification_code = NULL WHERE id = ?
                    ''', (username, hashed_password, user_id))
                    conn.commit()
                    st.success("Account activated! You can now log in.")
            else:
                st.error("No account found for this email or username. Contact admin.")
            conn.close()

# Function for Admin Create User Page
def create_user():
    if 'role' not in st.session_state or st.session_state['role'] != "Admin":
        st.error("Access denied. Only Admins can create users.")
        return

    with st.form("create_user_form"):
        email = st.text_input(label="Email", placeholder="Enter the user's email", label_visibility="collapsed")
        username = st.text_input(label="Username", placeholder="Enter the user's username", label_visibility="collapsed")
        role = st.selectbox(label="Select Role", options=["Program Director", "Academic Advisor", "System Administrator"], label_visibility="collapsed")
        submitted = st.form_submit_button("Create User")

    if submitted:
        conn = sqlite3.connect('user_db.db')
        cursor = conn.cursor()
        verification_code = generate_verification_code()
        hashed_password = bcrypt.hashpw("defaultpassword".encode('utf-8'), bcrypt.gensalt())
        try:
            cursor.execute('''
                INSERT INTO users (email, username, password, role, is_active, verification_code)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (email, username, hashed_password, role, 0, verification_code))
            conn.commit()
            if send_verification_email(email, verification_code):
                st.success(f"User {username} created successfully. Verification code sent to {email}.")
            else:
                st.error("Failed to send verification email. Please try again.")
        except sqlite3.IntegrityError:
            st.error("User with this email or username already exists.")
        conn.close()

import streamlit as st
from PIL import Image

# Function for admission prediction
def admission_selection():
    import streamlit as st
    import pandas as pd
    from joblib import load
    import numpy as np
    from scipy.stats import skew
    import time
    import os

    st.markdown(f"# Admission Prediction Tool")
    st.write("""
        Digital Mentor tool helps estimate your likelihood of being admitted to UMBC based on your standardized test scores for undergraduate studies in the (COEIT).
    """)

    # Utility function to make predictions
    def make_prediction(model, scaler, program, department, high_school_gpa, sat_score, act_score, advanced_course_work, stem_extracurriculars, leadership_experience, letters_of_recommendation, personal_statement_quality, diversity_contribution, transfer_Student_gpa):
        # Create a DataFrame from the inputs
        data = pd.DataFrame({
            'Program': [program],
            'Department': [department],
            'High School GPA': [high_school_gpa],
            'SAT Score': [sat_score],
            'ACT Score': [act_score],
            'Advanced Coursework (0-10)': [advanced_course_work],
            'STEM Extracurriculars (0-10)': [stem_extracurriculars],  
            'Leadership Experience (0-10)': [leadership_experience],
            'Letters of Recommendation (0-10)': [letters_of_recommendation],
            'Personal Statement Quality (0-10)': [personal_statement_quality],
            'Diversity/Inclusion Contribution (0-10)': [diversity_contribution],
            'Transfer Student GPA': [transfer_Student_gpa],
        })

        # Same preprocessing implemented during training 
        categorical_features = ['Program', 'Department']
        numerical_features = ['High School GPA', 'SAT Score', 'ACT Score', 
                              'Advanced Coursework (0-10)', 'STEM Extracurriculars (0-10)', 
                              'Leadership Experience (0-10)', 'Letters of Recommendation (0-10)', 
                              'Personal Statement Quality (0-10)', 'Diversity/Inclusion Contribution (0-10)', 
                              'Transfer Student GPA']
        
        for col in numerical_features:
            if skew(data[col]) != 0:  # Avoid log(0)
                data[col] = np.log1p(data[col])

        # This is categorical data with OneHotEncoder
        data_encoded = pd.get_dummies(data, columns=categorical_features)
        training_columns = [
            'High School GPA', 'SAT Score', 'ACT Score',
            'Advanced Coursework (0-10)', 'STEM Extracurriculars (0-10)',
            'Leadership Experience (0-10)', 'Letters of Recommendation (0-10)',
            'Personal Statement Quality (0-10)',
            'Diversity/Inclusion Contribution (0-10)', 'Transfer Student GPA',  
            'Program_Bioinformatics and Computational Biology',
            'Program_Business TechNology Administration',
            'Program_Chemical Engineering', 'Program_Computer Engineering',
            'Program_Computer Science', 'Program_Cybersecurity Informatics',
            'Program_Electrical Engineering', 'Program_Environmental Engineering',
            'Program_Information Systems', 'Program_Mechanical Engineering',
            'Department_Biological Sciences (in collaboration with COEIT)',
            'Department_Chemical, Biochemical & Environmental Engineering',
            'Department_Computer Science and Electrical Engineering',
            'Department_Information Systems', 'Department_Mechanical Engineering'
        ]

        data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
        X_test_scaled = scaler.transform(data_encoded)

        prediction = model.predict(X_test_scaled)
        confidences = model.predict_proba(X_test_scaled)
        return prediction, confidences[0]

    # Loading the model and scaler
    def load_artifacts(model_path, scaler_path):
        model = load(model_path)
        scaler = load(scaler_path)
        return model, scaler

    @st.cache_data
    def load_data(grades_path):
        grades_dataset = pd.read_csv(grades_path)
        program_names = grades_dataset['Program'].unique().tolist()
        department_names = grades_dataset['Department'].unique().tolist()
        return program_names, department_names

    # Paths to datasets and models
    grades_path = "/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/data/Admission_Dataset.csv"
    model_path = "/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/Models/admission_pipeline.joblib"
    scaler_path = "/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/models/scaler_pipeline.joblib"

    model, scaler = load_artifacts(model_path, scaler_path)
    program_names, department_names = load_data(grades_path)

    # These are the example User inputs
    name = st.text_input('Your Name')
    program = st.selectbox("Select your Program", options=program_names)
    department = st.selectbox("Select your Department", options=department_names)
    high_school_gpa = st.number_input("Enter your High School GPA", min_value=0.0, max_value=4.0, value=3.5)
    sat_score = st.number_input("Enter your SAT Score", min_value=400, max_value=1600, value=1190)
    act_score = st.number_input("Enter your ACT Score", min_value=0, max_value=36, value=27)
    advanced_course_work = st.number_input("Advanced Coursework (0-10)", min_value=0, max_value=10, value=5)
    stem_extracurriculars = st.number_input("STEM Extracurriculars (0-10)", min_value=0, max_value=10, value=5)
    leadership_experience = st.number_input("Leadership Experience (0-10)", min_value=0, max_value=10, value=5)
    letters_of_recommendation = st.number_input("Letters of Recommendation (0-10)", min_value=0, max_value=10, value=5)
    personal_statement_quality = st.number_input("Personal Statement Quality (0-10)", min_value=0, max_value=10, value=5)
    diversity_contribution = st.number_input("Diversity/Inclusion Contribution (0-10)", min_value=0, max_value=10, value=5)
    transfer_Student_gpa = st.number_input("Transfer Student GPA", min_value=0.0, max_value=4.0, value=3.5)

    # Initialize prediction variables
    pred, conf = None, None
    admission_accepted = 0 

    # Main Button to make prediction
    if st.button('Admission Prediction'):
        with st.spinner('Prediction result'):
            time.sleep(3)
            pred, conf = make_prediction(model, scaler, program, department, high_school_gpa, sat_score, act_score, advanced_course_work, stem_extracurriculars, leadership_experience, letters_of_recommendation, personal_statement_quality, diversity_contribution, transfer_Student_gpa)
            rejection_reasons = []
            if sat_score < 400 or high_school_gpa < 2.5 or act_score < 20:
                if sat_score < 400:
                    rejection_reasons.append("SAT score below the minimum threshold of 400.")
                if high_school_gpa < 2.5:
                    rejection_reasons.append("High School GPA below the minimum threshold of 2.5.")
                if act_score < 20:
                    rejection_reasons.append("ACT score below the minimum threshold of 20.")
                rejection_html = "<ul>" + "".join(f"<li>{reason}</li>" for reason in rejection_reasons) + "</ul>"
                st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Not Admitted: {rejection_html}</div>", unsafe_allow_html=True)
            else:   
                # Check the prediction outcome only if pred has been assigned
                if pred is not None and pred[0] == 1:
                    st.markdown(f"<div style='background-color:#4CAF50; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Admitted</div>", unsafe_allow_html=True)
                    admission_accepted = 1
                else:
                    rejection_reasons.append("Meets all individual score thresholds but does not fit the overall profile.")
                    rejection_html = "<ul>" + "".join(f"<li>{reason}</li>" for reason in rejection_reasons) + "</ul>"
                    st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Not Admitted: {rejection_html}</div>", unsafe_allow_html=True)

        # Saving admission prediction and confidence score if `conf` is available
        if conf is not None:
            confidence_row = pd.DataFrame([[name, conf[1], admission_accepted]], columns=['Name', 'Confidence', 'Admitted'])
            header = not os.path.exists('admission_confidence.csv')
            confidence_row.to_csv('admission_confidence.csv', mode='a', header=header, index=False)

# Function for course recommendations
def course_recommendation():
    import streamlit as st
    import pandas as pd
    import pdfplumber
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity    
    from sentence_transformers import SentenceTransformer
    import time
    import os
    
    st.markdown(f'# Course Recommendation Tool')
    st.write("""
        This tool assists you in selecting and comparing courses that align with your interests and career aspirations. Have fun exploring!
    """)

    # Utility function for course recommendation
    def recommend_courses(interests, career_goals, course_df):
        profile_text = interests + " " + career_goals

        # Load the pre-trained sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        profile_vector = model.encode([profile_text])
        course_vectors = model.encode(course_df['Matched Courses'].tolist())
        similarity_scores = cosine_similarity(profile_vector, course_vectors)
        top_indices = similarity_scores.argsort()[0][::-1]

        # Retrieve and display the top two distinct course recommendations
        unique_courses = []
        seen_descriptions = []
        similarity_values = []

        for index in top_indices:
            if len(unique_courses) == 2:
                break
            course_description = course_df['Matched Courses'].iloc[index]
            current_similarity = similarity_scores[0][index]
            if not any(cosine_similarity([model.encode(course_description)], [model.encode(seen)]) > 0.9 for seen in seen_descriptions):
                unique_courses.append(course_description)
                seen_descriptions.append(course_description)
                similarity_values.append(current_similarity)

        return unique_courses, similarity_values

    # Load course data
    @st.cache_data
    def load_data(courses_path):
        courses_dataset = pd.read_csv(courses_path)
        unique_interests = courses_dataset['Interests'].unique().tolist()
        unique_career_goals = courses_dataset['Career Goals'].unique().tolist()
        return courses_dataset, unique_interests, unique_career_goals

    # Paths to datasets
    courses_path = "/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/data/UMBC_COEIT_courses_dataset.csv"
    comparison_path = "/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/data/UMBC_COEIT_course_comparision.csv"
    
    courses_dataset, unique_interests, unique_career_goals = load_data(courses_path)
    comparison_dataset = pd.read_csv(comparison_path)

    # User input for name, interests, and career goals
    name = st.text_input('Your Name')
    interests = st.selectbox("Select your interests", options=unique_interests)
    career_goals = st.selectbox("Select your career goals", options=unique_career_goals)
    transcript_file = st.file_uploader("Upload your transcript (PDF, CSV, or Excel)", type=['pdf', 'csv', 'xlsx'])

    # Initialize feedback visibility state
    if 'feedback_visible' not in st.session_state:
        st.session_state['feedback_visible'] = False

    # Main Button to make prediction
    if st.button('Recommend Courses'):
        if not transcript_file:
            st.error("Please upload the transcript to proceed!")
            time.sleep(2)
            st.rerun()
        with st.spinner('Finding the right courses for you...'):
            recommendations, cosine_values = recommend_courses(interests, career_goals, courses_dataset)
            st.write("Recommended Courses:")
            for i, course in enumerate(recommendations):
                cosine_row = pd.DataFrame([[name, cosine_values[i], course]], columns=['Name', 'Confidence', 'Course'])
                header = not os.path.exists('course_confidence.csv')
                cosine_row.to_csv('course_confidence.csv', mode='a', header=header, index=False)
                st.markdown(
                    f"<div style='background-color:#C0C0C0; font-size:20px; font-weight:bold; border-radius:5px; padding:10px;'>{course}</div><br>",
                    unsafe_allow_html=True
                )

            # Display the comparison dataset for the recommended courses
            st.dataframe(
                comparison_dataset.loc[
                    (comparison_dataset['Core Courses'].str.contains(recommendations[0], case=False)) | 
                    (comparison_dataset['Core Courses'].str.contains(recommendations[1], case=False))
                ]
            )

            # Show feedback form after recommendation
            st.session_state['feedback_visible'] = True

    # Conditional display of feedback form
    if st.session_state['feedback_visible']:
        st.markdown(f'## Feedback Form')
        with st.form('feedback_form'):
            country = st.text_input('Your Country')
            mail = st.text_input('Enter your mail ID')
            course_suggested = st.text_input('Enter the course you were suggested')
            feedback = st.text_area('Please provide your feedback so we can improve!')
            submitted_feedback = st.form_submit_button('Submit feedback')

            if submitted_feedback:
                transcript_filename = transcript_file.name if transcript_file else None
                feedback_row = pd.DataFrame([[name, country, mail, course_suggested, feedback, transcript_filename]],
                                            columns=['Name', 'Country', 'Mail', 'course_suggested', 'Feedback', 'Transcript'])
                header = not os.path.exists('feedback.csv')
                feedback_row.to_csv('feedback.csv', mode='a', header=header, index=False)
                if transcript_file:
                    with open(os.path.join('uploaders', transcript_file.name), "wb") as f:
                        f.write(transcript_file.getbuffer())
                else:
                    st.error("Please upload the transcript to proceed.")
                st.success("Thank you for your feedback!")

# Function for Career Accelerator
def career_opportunities():
    import streamlit as st
    import pandas as pd
    import time
    import os
    
    st.markdown(f'# Career Accelerator')
    st.write("""
        This tool helps you find internships, workshops, or extracurricular activities relevant to your career interests.
    """)

    # Utility function to recommend activities based on interests
    def recommend_extracurriculars(career_interest, activities_df):
        # Filter the activities based on career interests
        relevant_activities = activities_df[activities_df['Career Interests'].str.contains(career_interest, case=False)]
        return relevant_activities

    # Load extracurricular activities data
    @st.cache_data
    def load_activities_data(activities_path):
        activities_dataset = pd.read_csv(activities_path)
        career_interests_list = activities_dataset['Career Interests'].unique().tolist()
        return activities_dataset, career_interests_list

    # Paths to datasets
    activities_path = "/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/data/UMBC_COEIT_activities_dataset.csv"
    
    activities_dataset, career_interests_list = load_activities_data(activities_path)

    # User input for name and career interest
    name = st.text_input('Your Name')
    career_interest = st.selectbox("Select your Career Interest", options=career_interests_list)

    # Button to recommend activities
    if st.button('Suggest Activities'):
        with st.spinner('Finding relevant activities for you...'):
            time.sleep(2)
            relevant_activities = recommend_extracurriculars(career_interest, activities_dataset)
            
            if not relevant_activities.empty:
                st.write("Recommended Internships, Workshops, or Extracurricular Activities:")
                st.dataframe(relevant_activities[['Activity Name', 'Description', 'Location', 'Date']])
                
                # Optionally, you can log this interaction for future analysis or feedback
                activity_log = pd.DataFrame([[name, career_interest]], columns=['Name', 'Career Interest'])
                header = not os.path.exists('activities_confidence.csv')
                activity_log.to_csv('activities_confidence.csv', mode='a', header=header, index=False)
            else:
                st.write("No relevant activities found. Please try another interest.")

# Function for Academic Progress Tracker
def academic_progress():
    import os
    
    st.markdown(f'# Academic Progress Dashboard')
    st.write("""This tool helps to track academic progress of a student, including GPA, total credits, and courses completed or in progress.""")

    # Utility function to load the academic progress data
    @st.cache_data
    def load_academic_data(progress_path):
        progress_dataset = pd.read_csv(progress_path)
        unique_names = progress_dataset['Name'].unique().tolist()
        return progress_dataset, unique_names
    
    # Path to dataset
    progress_path = "/Users/gowthami/Documents/pythonProjects/DigitalMentorUS/data/UMBC_COEIT_academic_progress_dataset.csv"
    
    progress_dataset, unique_names = load_academic_data(progress_path)

    # Select the student's name from a dropdown
    student_name = st.selectbox("Select Student name", options=unique_names)
    student_data = progress_dataset[progress_dataset['Name'] == student_name].iloc[0]

    # Main Button to display academic progress
    if st.button('View Academic Progress'):
        with st.spinner('Loading your academic progress...'):
            st.markdown(f"<div style='background-color:#F0F0F0; padding:10px; border-radius:10px;'>", unsafe_allow_html=True)
            st.markdown(f"### {student_name}'s Academic Progress", unsafe_allow_html=True)
            st.markdown(f"**Program**: `{student_data['Program']}`", unsafe_allow_html=True)
            st.markdown(f"**Department**: `{student_data['Department']}`", unsafe_allow_html=True)
            st.markdown(f"### GPA: `{student_data['GPA']}`", unsafe_allow_html=True)
            st.markdown(f"### Credits Earned: `{student_data['TotalCredits']}` / `{student_data['RequiredCredits']}`", unsafe_allow_html=True)
            st.markdown("### Completed Courses", unsafe_allow_html=True)
            st.markdown(f"- {', '.join(student_data['CompletedCourses'].split(', '))}", unsafe_allow_html=True)
            st.markdown("### In-Progress Courses", unsafe_allow_html=True)
            st.markdown(f"- {', '.join(student_data['InProgressCourses'].split(', '))}", unsafe_allow_html=True)           
            st.markdown("</div>", unsafe_allow_html=True)   

# Function for Admin
def admin():
    import streamlit as st
    import pandas as pd
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px

    if 'is_logged_in' not in st.session_state or not st.session_state['is_logged_in']:
        st.error("You need to log in to access this page.")
        return

    if 'role' not in st.session_state or st.session_state['role'] not in ["Admin", "System Administrator", "Program Director", "Academic Advisor"]:
        st.error("Access denied. This page is restricted to administrators.")
        return

    st.title(f"My Home - {st.session_state['role']}")
    st.write(f"Welcome to the Admin Dashboard, {st.session_state['role']}!")

    tabs = st.tabs(["Transcript Review", "Model Performance Dashboard"])

    # try:
    with tabs[0]:
        st.subheader("Transcript Review")
        try:
            feedback_df = pd.read_csv('feedback.csv')
            if not feedback_df.empty:
                # Check and update session state for checkboxes
                if 'checkbox_state' not in st.session_state or len(st.session_state.checkbox_state) != len(feedback_df):
                    st.session_state.checkbox_state = [False] * len(feedback_df)
                # Display data in a table format with interactive checkboxes for user selection
                for index, row in feedback_df.iterrows():
                    cols = st.columns([1, 3, 1, 5, 4])
                    with cols[0]:
                        checkbox = st.checkbox("Select item", key=f"check{index}", value=st.session_state.checkbox_state[index], label_visibility="collapsed")
                        st.session_state.checkbox_state[index] = checkbox
                    with cols[1]:
                        st.write(f"{row['Name']}")
                    with cols[2]:
                        st.write(f"{row['Country']}")
                    with cols[3]:
                        st.write(f"{row['Mail']}")
                    with cols[4]:
                        st.write(f"{row['course_suggested']}")
                        if row['Transcript']:
                            file_path = os.path.join('uploaders', row['Transcript'])
                            with open(file_path, "rb") as file:
                                st.download_button(label="Download Transcript", data=file.read(), file_name=row['Transcript'], mime='application/pdf', key=f"download_button_{index}")
                        else:
                            st.write("No transcript uploaded")
                # Button to process checked items
                if st.button('Process Checked Items'):
                    checked_feedback = feedback_df[st.session_state.checkbox_state]
                    st.write("Checked items processed:", checked_feedback)
            else:
                st.write("No feedback yet.")
        except FileNotFoundError:
            st.error("Feedback file not found.")

    # Model Performance Dashboard Tab
    with tabs[1]:
        st.subheader("Model Performance Dashboard")

        # Function to plot admission confidence
        def plot_admission_confidence(df):
            fig = px.line(df, x='Name', y='Confidence', markers=True, title='Admission Confidence by Candidate')
            fig.update_xaxes(title_text='Candidate Name')
            fig.update_yaxes(title_text='Confidence Level')
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

        # Function to plot course confidence
        def plot_course_confidence(df):
            fig = px.line(df, x='Name', y='Confidence', color='Course', markers=True, title='Course Recommendation Confidence by Candidate')
            fig.update_xaxes(title_text='Candidate Name')
            fig.update_yaxes(title_text='Confidence Level')
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

        # Load and plot admission confidence scores
        try:
            admission_df = pd.read_csv('admission_confidence.csv')
            if not admission_df.empty:
                plot_admission_confidence(admission_df)
            else:
                st.write("No admission data yet.")
        except FileNotFoundError:
            st.write("No admission data found.")

        # Load and plot course confidence scores
        try:
            course_df = pd.read_csv('course_confidence.csv')
            if not course_df.empty:
                plot_course_confidence(course_df)
            else:
                st.write("No course data yet.")
        except FileNotFoundError:
            st.write("No course data found.")

# Main function to map pages   
def main():        
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Intro Page" 
    selected_page_function = sidebar_menu()
    if selected_page_function:
        selected_page_function()
    else:
        intro()

# Main function to execute the selected page
def main():
    selected_page_function = sidebar_menu()
    if selected_page_function:
        selected_page_function()
if __name__ == "__main__":
    main()

