Digital-Mentor-for-UMBC-COEIT-Undergraduate-Studies

Project Overview
The Digital Mentor is a personalized tool designed to assist undergraduate students at UMBC, particularly in the College of Engineering and Information Technology (COEIT). It provides personalized academic recommendations, admission predictions, course suggestions, and progress tracking, all in one platform. The project is built using Streamlit for an interactive user interface and leverages machine learning models to provide data-driven recommendations.
Features
•	Admission Checker: Estimate the likelihood of admission to UMBC based on academic performance and standardized test scores.
•	Course Recommender: Get personalized course recommendations based on your interests and career goals.
•	Academic Progress Tracker: Monitor GPA, completed credits, and in-progress courses.
•	Career Accelerator: Discover internships, workshops, and extracurricular activities aligned with your career goals.
•	Admin Dashboard: Review user feedback and analyze system performance for iterative improvements.
 
System Requirements

You can install all the necessary libraries from req.txt using the command:

pip install -r req.txt
 
File Structure

Technical Stack: for the Digital Mentor for UMBC Undergraduate Studies application consists of the following technologies:

I am using Git, to manage and collaborate my source code.

Front-end Framework:

•	Streamlit: I have used it to build the user interface, which allows users to interact with the application, upload data, receive predictions, and provide feedback.
•	PIL (Python Imaging Library): Used for handling image files and displaying icons for better user experience.

Back-end and Logic:
•	Python: Core language I have used is python for backend development in order to implement logic of the application, all machine learning algorithms, and handling of the data.
•	Joblib: For loading pre-trained machine learning models that predict admissions.
•	Pandas: Used for data handling and manipulation, especially for processing user inputs, datasets, and feedback.
•	NumPy: For numerical operations and handling mathematical calculations (e.g., skew for data transformations).
•	Scikit-learn: For machine learning models, data transformation (e.g., OneHotEncoding), and scaling data.
•	SentenceTransformers: A pre-trained NLP model (e.g., all-MiniLM-L6-v2) for generating course recommendations based on user input.
•	Cosine Similarity: From Scikit-learn to find similarity between user profiles and courses.

Data Storage and Processing:
•	CSV Files: Used for storing datasets, including course data, admission data, academic progress data, extracurricular activities, and user feedback.
•	MySQL: User data, such as admission confidence scores and recommended courses, is saved as a backup for my application.

Visualization:
•	Matplotlib and Seaborn: For creating plots and data visualizations for the admin dashboard.
•	Plotly Express: Interactive data visualization library for plotting model performance and feedback data.
 
How to Run the Project
1.	Clone the repository:
bash
git clone https://github.com/GowthamiBankuru/Digital-Mentor-for-UMBC-COEIT-Undergraduate-Studies
2.	Navigate to the project directory:
cd DigitalMentorUS
3.	Install dependencies: Ensure you have Python installed, then install the required packages using:
pip install -r req.txt
4.	Run the Streamlit app: Start the Streamlit application by running:
streamlit run app.py
5.	Interact with the application: Open your browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501). From the sidebar, you can navigate through different tools such as the Admission Checker, Course Recommender, Academic Progress Tracker, Career Accelerator, and the Admin Dashboard.
 
How to Use Each Feature
1.	Admission Checker:
o	Enter your academic information (e.g., high school GPA, SAT/ACT scores).
o	Receive a prediction of your likelihood of being admitted to a UMBC COEIT program.
o	Review any areas where your scores may need improvement.
2.	Course Recommender:
o	Input your academic interests and career goals.
o	Upload your transcript to get personalized course suggestions.
o	View a comparison of the recommended courses.
3.	Academic Progress Tracker:
o	Select your name from the list.
o	View your GPA, completed credits, and courses in progress.
o	Optionally update your academic progress manually.
4.	Career Accelerator:
o	Select your career interests.
o	Receive suggestions for relevant extracurricular activities, internships, and workshops.
5.	Admin Dashboard:
o	Review user feedback and transcript uploads.
o	Visualize the confidence scores of the admission predictions and course recommendations.
 
Project Contributions
This project was developed to provide UMBC students with personalized academic and career guidance. It integrates multiple datasets and machine learning techniques to deliver accurate recommendations. Any feedback or contributions are welcome to improve the system further.
For any issues or contributions, please contact [your-email@example.com].
 
Future Enhancements
•	Integrating additional datasets to refine recommendations and predictions.
•	Improving the machine learning models for greater accuracy.
•	Expanding the Career Accelerator tool to include more career-focused opportunities.
•	Adding authentication and user account management for a more personalized experience.
 
Acknowledgments
Special thanks to the UMBC COEIT for providing the datasets and support throughout the project development.
 
This README.txt serves as a comprehensive guide for running, understanding, and contributing to the Digital Mentor for UMBC Undergraduate Studies project.
