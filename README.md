**EMPLOYEE ATTRITION AND EMOTION DETECTION**

You can run this project in your desktop by installing all the libraries and packages that are required. The requirements are listed in req.txt file. 
Make sure that you set up a virtual environment and run this project in that virtual environment. 
You are all set to run the project locally in your machine!

You can run the project by entering the command "streamlit run 0_🏠_Home.py" in your terminal.

`HOME PAGE`
![Screenshot (86)](https://github.com/FaaizMohamed/Employee_Attrition/assets/125682181/526e9b73-f50c-45e6-bc2a-0cc150455462)

`SIGNUP AND LOGIN PAGE`
![Screenshot (64)](https://github.com/FaaizMohamed/Employee_Attrition/assets/125682181/7fbf7eee-bd4d-4f6d-9c40-726c6638a4e1)

`EMPLOYEE DASHBOARD (EMOTION DETECTION)`
![Screenshot (65)](https://github.com/FaaizMohamed/Employee_Attrition/assets/125682181/88d54424-5ce8-402b-a219-bb4460d29d1b)

`EMPLOYEE DASHBOARD (SURVEY FORM)`
![Screenshot (78)](https://github.com/FaaizMohamed/Employee_Attrition/assets/125682181/2451a948-5a87-4056-9e4a-98988783b116)

`HR DASHBOARD (EMOTION ANALYTICS)`
![Screenshot (80)](https://github.com/FaaizMohamed/Employee_Attrition/assets/125682181/e9cd543f-ce92-486b-a772-d3fbbffa06fc)

`HR DASHBOARD (ATTRITION PREDICTION RESULT)`
![Screenshot (85)](https://github.com/FaaizMohamed/Employee_Attrition/assets/125682181/2147abec-d646-4697-8bcc-80537899505a)

### Project : Employee Attrition Prediction and Emotion Detection

**Objective:**
The primary goal of this final year project is to predict employee attrition and monitor employee emotions to enhance employee satisfaction and reduce attrition rates, thereby maintaining productivity in an organization. The project employs machine learning models for attrition prediction and uses computer vision techniques to detect and analyze employee emotions, providing valuable insights to the Human Resources (HR) department.

**Technologies Used:**
- **Machine Learning**: For predicting employee attrition.
- **Python**: The programming language used for implementing various parts of the project.
- **Flask**: A micro web framework used to build the web application.
- **Streamlit**: An open-source app framework used for creating interactive web applications for the project.
- **OpenCV**: A library used for real-time computer vision.
- **DeepFace**: A facial recognition and facial attribute analysis framework used for emotion detection.

**Components:**

1. **Employee Attrition Prediction:**
   - **Data Collection and Preprocessing**: Gathered historical employee data including features such as job satisfaction, work environment, salary, years at company, and more. The data was cleaned and preprocessed for training machine learning models.
   - **Model Training**: Various machine learning models (e.g., logistic regression, decision trees, random forests, gradient boosting) were trained to predict the likelihood of an employee leaving the company. The best-performing model was selected based on accuracy, precision, recall, and other relevant metrics.
   - **Prediction and Analysis**: The final model was used to predict whether an employee is likely to leave. The predictions were analyzed to identify key factors contributing to attrition, providing actionable insights to HR.

2. **Emotion Detection:**
   - **Real-time Emotion Monitoring**: Using OpenCV and DeepFace, a real-time emotion detection system was developed. The system captures facial images of employees and analyzes their emotions (e.g., happy, sad, angry, neutral).
   - **Integration with HR Systems**: The emotion data is regularly reported to the HR department. This helps HR understand the emotional well-being of employees and take necessary actions to improve job satisfaction and overall morale.

3. **Web Application:**
   - **Flask Backend**: The backend of the application was developed using Flask. It handles data processing, model predictions, and integration with the database.
   - **Streamlit Frontend**: Streamlit was used to create an interactive and user-friendly web interface for HR to visualize attrition predictions and emotion analysis. The dashboard provides detailed reports, trends, and real-time monitoring capabilities.

**Benefits:**
- **Proactive HR Management**: Enables HR to identify at-risk employees and take proactive measures to retain them.
- **Employee Satisfaction**: By monitoring emotions and addressing concerns promptly, the system helps in maintaining a positive work environment.
- **Data-Driven Decisions**: Provides HR with data-driven insights for better decision-making regarding employee management and retention strategies.

**Conclusion:**
This project integrates advanced machine learning techniques with real-time emotion detection to provide a comprehensive solution for predicting employee attrition and enhancing employee welfare. By leveraging Python, Flask, Streamlit, OpenCV and DeepFace, the project delivers a robust tool for HR to improve employee satisfaction and reduce turnover, ultimately contributing to the organization's success.

