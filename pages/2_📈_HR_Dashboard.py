import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import ast
import seaborn as sns
from streamlit_extras.stateful_button import button

# Sample DataFrame with categorical columns
st.set_page_config(
    layout="wide",
)
model = joblib.load('lgbm_model.pkl')

# Load the saved preprocessing pipeline
preprocessor = joblib.load('preprocessing_pipelineclassi.pkl')
user_data = pd.read_csv('userhr.csv')

# Create a new DataFrame with 1 for 'Travel_Rarely' and 0 for other columns
def display_column_data(dataframe, column_name):
    """
    Display the column name and its respective data using st.metric.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column to display.

    """
    # Get the column data
    col_data = dataframe[column_name]

    # Compute a summary statistic based on the data type of the column
    if pd.api.types.is_numeric_dtype(col_data):
        value = col_data.mean()  # Compute mean for numeric columns
    else:
        value = col_data.mode().iloc[0]  # Compute mode for categorical columns
    
    st.metric(label=column_name, value=value)

# Load the pre-trained RandomForestRegressor model

def preprocess_input(input_data):
    transformed_input = preprocessor.transform(input_data)  # Apply preprocessing pipeline
    return transformed_input

def extract_emotions(data):
    # Extract 'Emotions' column and convert it to a list of dictionaries
    emotions_list = data['Emotions'].apply(ast.literal_eval).tolist()
    return emotions_list

def plot_emotions(emotions_list):
    # st.text(emotions_list)
    emotion_counts = {}
    
    for emotion_dict in emotions_list:
        for emotion, count in emotion_dict.items():
            if emotion in emotion_counts:
                emotion_counts[emotion] += count
            else:
                emotion_counts[emotion] = count

    # Convert dictionary into lists for plotting
    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    # st.text(labels)
    # st.text(sizes)
    # Plot pie chart
    fig, ax = plt.subplots()
    _, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,shadow=True)
    ax.axis('equal')
    for text in texts + autotexts:
        text.set_color('white')
    fig.patch.set_facecolor('#121212')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # st.pyplot(fig)  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

def plot_column_distribution(data, column_name):
    plt.figure(figsize=(10, 6))
    
    # Check if the column is categorical or numerical
    if pd.api.types.is_numeric_dtype(data[column_name]):
        # Plot histogram for numerical columns
        sns.histplot(data[column_name], kde=True)
        plt.xlabel(column_name)
        plt.title(f'{column_name} Distribution')
    else:
        # Plot bar chart for categorical columns
        data[column_name].value_counts().plot(kind='bar')
        plt.xlabel(column_name)
        plt.ylabel('Count')
        plt.title(f'{column_name} Distribution')
        plt.xticks(rotation=45)
    
    # Return the current figure
    return plt.gcf()


# Sidebar for navigation
st.title('HR Dashboard')
st.subheader('Employee Attrition')
st.write('This app predicts employee attrition')
page = st.sidebar.selectbox("Choose a page", ["Login", "Signup"])
login_varia= False
if page == "Login":
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    employee_id = st.sidebar.text_input("HR ID")
    login_varia = False
    if st.sidebar.button("Login", key="button22"):
        login_varia = True
        user_data = pd.read_csv('userhr.csv')
        if username in user_data['username'].values:
            # Check if the password and employee ID match
            stored_password = user_data.loc[user_data['username'] == username, 'password'].values[0]
            stored_employee_id = user_data.loc[user_data['username'] == username, 'employeeid'].values[0]

            if password == stored_password and employee_id == stored_employee_id:
                st.sidebar.success("Logged in as {}".format(username))
                login_varia= True
                if login_varia is True:
                    # Load CSV file
                    # uploaded_file = pd.read_csv('input_data.csv')
                    input_data = pd.read_csv('input_data.csv')
                    employee_ids = input_data['Empid'].unique()

                    # Create a dropdown menu to select an employee
                    selected_employee_id = st.selectbox("Select an employee ID", employee_ids)

                    # Filter the data for the selected employee
                    employee_data = input_data[input_data['Empid'] == selected_employee_id]
                    st.write(employee_data)
                    emotions_list = extract_emotions(employee_data)
                    st.subheader('Emotions Distribution')
                    plot_emotions(emotions_list)

                    # Drop the 'Empid' and 'Empusername' columns
                    employee_data = employee_data.drop(['Empid', 'Empusername','Emotions','Emotion_labels','Emotion_count'], axis=1)

                    for column in employee_data.columns:
                        display_column_data(employee_data, column)

                    # if button("Predict Attrition",key='predict button'):
                    employee_data.columns = ['DailyRate','Age', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
                                        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 
                                        'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                                            'YearsWithCurrManager', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
                                            'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 
                                            'WorkLifeBalance',
                                            'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                                            'OverTime', 'Over18',]

                    # Preprocess user input
                    # st.write(employee_data)
                    processed_inputs = preprocess_input(employee_data)

                    # Make prediction using the pre-trained model
                    predictions = model.predict(processed_inputs)

                    # Display prediction result
                    st.header('Predictions')
                    if predictions[0]==0:
                        st.metric(label='Predicted Employee Attrition:', value='No')
                    if predictions[0]==1:
                        st.metric(label='Predicted Employee Attrition:', value='Yes')        
                else:
                    st.error("Please Login First")
            else:
                st.sidebar.error("Login First")
elif page == "Signup":
    st.sidebar.subheader("Signup")
    new_username = st.sidebar.text_input("Choose a username")
    new_password = st.sidebar.text_input("Choose a password", type="password")
    new_employee_id = st.sidebar.text_input("Enter HR ID")

    if st.sidebar.button("Signup"):
        if new_username in user_data['username'].values:
            st.sidebar.warning("Username already exists. Please choose a different one.")
        else:
            # Add new user to the CSV
            new_user = pd.DataFrame({'username': [new_username], 'password': [new_password], 'employeeid': [new_employee_id]})
        
            # Concatenate the new user DataFrame with the existing user_data
            user_data = pd.concat([user_data, new_user], ignore_index=True)
            user_data.to_csv('userhr.csv', index=False)
            st.sidebar.success("Signup successful! You can now login.")
            user_data = pd.read_csv('userhr.csv')
            login_varia= False



