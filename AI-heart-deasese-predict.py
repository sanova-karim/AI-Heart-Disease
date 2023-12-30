import pandas as pd
from sklearn.model_selection import train_test_split # utility function that splits dataset into training and testing subsets
from sklearn.tree import DecisionTreeClassifier # Machine learning algorithm that builds the decision tree

# Function to convert shorthand responses to full words
def standardize_user_responses(response, category):
    if category == "High_Med_Low":
         return {"L": "Low", "M": "Moderate", "H": "High"}.get(response.upper(), response.capitalize())
    else:
        return "Yes" if response.upper() == "Y" else "No"

# Function to validate non-numeric inputs 
def validate_user_input(prompt, valid_choices):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in valid_choices:
            return user_input
        else:
            print("Invalid input. Please enter a valid choice.")

# Read the dataset from a CSV file
# In order to use this tool effectively the data base should come from the locality/region
# but the name of the file must be heart_disease_data.csv

df = pd.read_csv('heart_disease_data.csv')

# Convert categorical variables to numeric including all categories
df_encoded = pd.get_dummies(df, columns=['Gender','PhysicalActivity', 'Smoking', 'AlcoholConsumption', 'Diabetes', 'HighFatDiet', 'HighCarbDiet', 'FamilyHistory'], drop_first=False)

# Separate features and target, including 'Diastolic', 'Systolic', 'HighFatDiet', 'HighCarbDiet'
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# Include Cholestorol, Systolic, Diastolic and resting heart rate columns in X
X['Cholesterol'] = df_encoded['Cholesterol']
X['Systolic'] = df_encoded['Systolic']
X['Diastolic'] = df_encoded['Diastolic']
X['RestingHeartRate'] = df_encoded['RestingHeartRate']


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Get new patient data from user input with conversion

# Personal data
age = int(input("Enter age: "))
gender = validate_user_input("Enter gender (Male/Female): ", ["male", "female"])
family_history = validate_user_input("Family history of heart disease? (Yes/Y/No/N): ", ["yes", "y", "no", "n"])

# Life style related data
physical_activity = standardize_user_responses(validate_user_input("Physical activity (Low/L/Moderate/M/High/H): ", ["low", "l", "moderate", "m", "high", "h"]), "High_Med_Low")
smoking = standardize_user_responses(validate_user_input("Smoking (Yes/Y/No/N): ", ["yes", "y", "no", "n"]), "yes_no")
alcohol_consumption = standardize_user_responses(validate_user_input("Alcohol consumption (Yes/Y/No/N): ", ["yes", "y", "no", "n"]), "yes_no")
high_fat_diet = standardize_user_responses(validate_user_input("High fat diet (Yes/Y/No/N): ", ["yes", "y", "no", "n"]), "yes_no")
high_carb_diet = standardize_user_responses(validate_user_input("High carb diet (Yes/Y/No/N): ", ["yes", "y", "no", "n"]), "yes_no")


#Medical data
diabetes = standardize_user_responses(validate_user_input("Diabetes (Yes/Y/No/N): ", ["yes", "y", "no", "n"]), "yes_no")
cholesterol = int(input("Enter cholesterol level: "))
systolic = int(input("Enter systolic blood pressure: "))  
diastolic = int(input("Enter diastolic blood pressure: "))  
resting_heart_rate = int(input("Enter resting heart rate: "))


# Creating a new patient DataFrame
new_patient = pd.DataFrame({
    'Age': [age],
    'Gender_Male': [1 if gender == 'Male' else 0],
    'FamilyHistory': [family_history],       
    'PhysicalActivity_Low': [1 if physical_activity == 'Low' else 0],
    'PhysicalActivity_Moderate': [1 if physical_activity == 'Moderate' else 0],
    'PhysicalActivity_High': [1 if physical_activity == 'High' else 0],
    'Smoking_Yes': [1 if smoking == 'Yes' else 0],
    'Smoking_No': [1 if smoking == 'No' else 0],
    'AlcoholConsumption_Yes': [1 if alcohol_consumption == 'Yes' else 0],
    'AlcoholConsumption_No': [1 if alcohol_consumption == 'No' else 0],   
    'HighFatDiet_Yes': [1 if high_fat_diet == 'Yes' else 0],
    'HighFatDiet_No': [1 if high_fat_diet == 'No' else 0],
    'HighCarbDiet_Yes': [1 if high_carb_diet == 'Yes' else 0],
    'HighCarbDiet_No': [1 if high_carb_diet == 'No' else 0],
    'Diabetes_Yes': [1 if diabetes == 'Yes' else 0],
    'Diabetes_No': [1 if diabetes == 'No' else 0],
    'Cholesterol': [cholesterol],
    'Systolic': [systolic],
    'Diastolic': [diastolic],    
    'RestingHeartRate': [resting_heart_rate]
})

# Ensure new_patient has the same column order as X_train
new_patient = new_patient.reindex(columns=X_train.columns, fill_value=0)


# Predict the presence of heart disease for the new patient
heart_disease_probability = clf.predict(new_patient)

# Map predicted data from "Yes" to "High" and "No" to "Low"
heart_disease_probability_str = "High" if heart_disease_probability[0] == 'Yes' else "Low"

print("Pobability of heart disease for the new patient:", heart_disease_probability_str)


