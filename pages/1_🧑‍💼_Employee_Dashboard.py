import streamlit as st
import pandas as pd
import cv2
from deepface import DeepFace
from streamlit_extras.stateful_button import button
import matplotlib.pyplot as plt
import csv

st.set_page_config(
    layout="wide",
)
# Initialize session state
if 'stop_camera' not in st.session_state:
    st.session_state['stop_camera'] = False
if 'emotions_list' not in st.session_state:
    st.session_state['emotions_list'] = []

def save_to_csv():
    data = {
        'Empid':[employee_id],
        'Empusername':[username],
        'Age': [age],
        'Daily_Rate': [daily_rate],
        'Distance_From_Home': [distance_from_home],
        'Hourly_Rate': [hourly_rate],
        'Monthly_Income': [monthly_income],
        'Monthly_Rate': [monthly_rate],
        'Num_Companies_Worked': [num_companies_worked],
        'Percent_Salary_Hike': [percent_salary_hike],
        'Total_Working_Years': [total_working_years],
        'Training_Times_Last_Year': [training_times_last_year],
        'Years_at_Company': [years_at_company],
        'Years_in_Current_Role': [years_in_current_role],
        'Years_Since_Last_Promotion': [years_since_last_promotion],
        'Years_with_Current_Manager': [years_with_curr_manager],
        'Environment_Satisfaction': [environment_satisfaction],
        'Job_Involvement': [job_involvement],
        'Job_Level': [job_level],
        'Job_Satisfaction': [job_satisfaction],
        'Performance_Rating': [performance_rating],
        'Relationship_Satisfaction': [relationship_satisfaction],
        'Stock_Option_Level': [stock_option_level],
        'Work_Life_Balance': [work_life_balance],
        'Business_Travel': [business_travel],
        'Department': [department],
        'Education_Field': [education_field],
        'Gender': [gender],
        'Job_Role': [job_role],
        'Marital_Status': [marital_status],
        'Over_Time': [over_time],
        'Over_18': [over_18],
        'Emotion_labels':[labels],
        'Emotion_count':[sizes],
        'Emotions':[emotion_counts]
    }

    df = pd.DataFrame(data)
    # with open('input_data.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(df.values.tolist()[0])
    df.to_csv('input_data.csv', index=False)

st.title("Employee Dashboard")
# Load user data
user_data = pd.read_csv('users.csv')

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Login", "Signup"])
login_var= False
if page == "Login":
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    employee_id = st.sidebar.text_input("Employee ID")
    login_var = False
    with st.sidebar:
        if button("Login", key="button22"):
            login_var = True
            user_data = pd.read_csv('users.csv')
            if username in user_data['username'].values:
                # Check if the password and employee ID match
                stored_password = user_data.loc[user_data['username'] == username, 'password'].values[0]
                stored_employee_id = user_data.loc[user_data['username'] == username, 'employeeid'].values[0]

                if password == stored_password and employee_id == stored_employee_id:
                    st.sidebar.success("Logged in as {}".format(username))
elif page == "Signup":
    st.sidebar.subheader("Signup")
    new_username = st.sidebar.text_input("Choose a username")
    new_password = st.sidebar.text_input("Choose a password", type="password")
    new_employee_id = st.sidebar.text_input("Enter Employee ID")

    if st.sidebar.button("Signup"):
        if new_username in user_data['username'].values:
            st.sidebar.warning("Username already exists. Please choose a different one.")
        else:
            # Add new user to the CSV
            new_user = pd.DataFrame({'username': [new_username], 'password': [new_password], 'employeeid': [new_employee_id]})
        
            # Concatenate the new user DataFrame with the existing user_data
            user_data = pd.concat([user_data, new_user], ignore_index=True)
            # user_data.to_csv('pages/users.csv', index=False)
            user_data.to_csv('users.csv', index=False)
            st.sidebar.success("Signup successful! You can now login.")
            user_data = pd.read_csv('users.csv')

if login_var is True:
    if button("Employee Dashboard", key="button33"):
        # Create a VideoCapture object
        cap = cv2.VideoCapture(0)
        stframe = st.image([])  # Placeholder for the webcam feed
        # Add a stop button to end the camera loop
        if button("Finish", key="stop_button"):
            st.session_state['stop_camera'] = True

        while not st.session_state['stop_camera']:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Analyze the frame using DeepFace
            result = DeepFace.analyze(img_path=frame, actions=['emotion'],
                                        enforce_detection=False,
                                        detector_backend="opencv",
                                        align=True,
                                        silent=True)

            # Extract the face coordinates
            dominant_emotion = result[0]['dominant_emotion']
            st.session_state['emotions_list'].append(dominant_emotion)

            face_coordinates = result[0]["region"]
            x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Convert the BGR frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            stframe.image(frame_rgb, channels="RGB")
            

        # Release the camera when stopped
        cap.release()

        # Reset session state to allow restarting the camera
        st.session_state['stop_camera'] = False
        emotion_counts = {emotion: st.session_state['emotions_list'].count(emotion) for emotion in set(st.session_state['emotions_list'])}
        # st.text(emotion_counts)
        st.header('Survey Form')

        # Numerical input controls
        # Numerical input controls
        age = st.slider('Age', min_value=18, max_value=65, value=30)
        daily_rate = st.slider('Daily_Rate', min_value=0, max_value=2000, value=1000)
        distance_from_home = st.slider('Distance_From_Home', min_value=1, max_value=30, value=10)
        hourly_rate = st.slider('Hourly_Rate', min_value=0, max_value=100, value=50)
        monthly_income = st.slider('Monthly_Income', min_value=1000, max_value=20000, value=5000)
        monthly_rate = st.slider('Monthly_Rate', min_value=0, max_value=25000, value=12000)
        num_companies_worked = st.slider('Num_Companies_Worked', min_value=0, max_value=10, value=5)
        percent_salary_hike = st.slider('Percent_Salary_Hike', min_value=0, max_value=30, value=15)
        total_working_years = st.slider('Total_Working_Years', min_value=0, max_value=40, value=10)
        training_times_last_year = st.slider('Training_Times_Last_Year', min_value=0, max_value=6, value=2)
        years_at_company = st.slider('Years_at_Company', min_value=0, max_value=40, value=5)
        years_in_current_role = st.slider('Years_in_Current_Role', min_value=0, max_value=20, value=3)
        years_since_last_promotion = st.slider('Years_Since_Last_Promotion', min_value=0, max_value=15, value=2)
        years_with_curr_manager = st.slider('Years_with_Current_Manager', min_value=0, max_value=15, value=3)
        environment_satisfaction = st.slider('Environment_Satisfaction', min_value=1, max_value=4, value=3)
        job_involvement = st.slider('Job_Involvement', min_value=1, max_value=4, value=3)
        job_level = st.slider('Job_Level', min_value=1, max_value=5, value=3)
        job_satisfaction = st.slider('Job_Satisfaction', min_value=1, max_value=4, value=3)
        performance_rating = st.slider('Performance_Rating', min_value=1, max_value=4, value=3)
        relationship_satisfaction = st.slider('Relationship_Satisfaction', min_value=1, max_value=4, value=3)
        stock_option_level = st.slider('Stock_Option_Level', min_value=0, max_value=3, value=1)
        work_life_balance = st.slider('Work_Life_Balance', min_value=1, max_value=4, value=3)

        # Categorical input controls
        business_travel = st.selectbox('Business_Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
        department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
        education_field = st.selectbox('Education_Field', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        job_role = st.selectbox('Job_Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
        marital_status = st.selectbox('Marital_Status', ['Single', 'Married', 'Divorced'])
        over_time = st.selectbox('Over_Time', ['Yes', 'No'])
        over_18 = st.selectbox('Over_18', ['Y','N'])  # Assuming 'Over 18' always 'Y'
        if st.button('Submit Form'):
            labels = list(emotion_counts.keys())
            sizes = list(emotion_counts.values())
            save_to_csv()
            st.success('Form Submitted!')
            # Plot pie chart

            # fig, ax = plt.subplots()
            # _, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,shadow=True)
            # ax.axis('equal')
            # for text in texts + autotexts:
            #     text.set_color('white')
            # fig.patch.set_facecolor('#121212')
            # st.pyplot(fig)
            # st.text("AlreadyStopped")
else:
    st.error("Please Login First")

    # # Add a stop button to end the camera loop
    # if button("Stop Camera", key="stop_button"):
    #     st.session_state['stop_camera'] = True


            #     # Check if the "Stop" button is clicked
            # if button("Button 4", key="button44"):
            #     stop_camera=1
            # else:
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     st.write("All 3 buttons are pressed")

            # # Release the webcam and close all windows
            

# # Function to analyze facial attributes using Deep

# elif page == "Signup":
#     st.sidebar.subheader("Signup")
#     new_username = st.sidebar.text_input("Choose a username")
#     new_password = st.sidebar.text_input("Choose a password", type="password")
#     new_employee_id = st.sidebar.text_input("Enter Employee ID")

#     if st.sidebar.button("Signup"):
#         if new_username in user_data['username'].values:
#             st.sidebar.warning("Username already exists. Please choose a different one.")
#         else:
#             # Create a new DataFrame for the new user
#             new_user = pd.DataFrame({'username': [new_username], 'password': [new_password], 'employeeid': [new_employee_id]})
            
#             # Concatenate the new user DataFrame with the existing user_data
#             user_data = pd.concat([user_data, new_user], ignore_index=True)
            
#             # Save the updated user_data DataFrame back to CSV
#             user_data.to_csv('users.csv', index=False)
#             st.sidebar.success("Signup successful! You can now login.")
