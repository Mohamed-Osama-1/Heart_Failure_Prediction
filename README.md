Heart Disease Prediction Project
The primary objective of this project is to predict the likelihood of heart failure based on 11 clinical features. These features include:

[1]-Age: Age of the patient in years.
[2]-Sex: Gender of the patient [M: Male, F: Female].
[3]-Chest Pain Type:
  -TA: Typical Angina.
  -ATA: Atypical Angina.
  -NAP: Non-Anginal Pain.
  -ASY: Asymptomatic.
[4]-Resting Blood Pressure (RestingBP): Resting blood pressure measured in mm Hg.
[5]-Cholesterol: Serum cholesterol level measured in mg/dl (normal range: 200-240).
[6]-Fasting Blood Sugar (FastingBS): Indicates prediabetes or diabetes status [1: FastingBS > 120 mg/dl, 0: otherwise].
[7]Resting Electrocardiogram (RestingECG): Results of resting ECG.
[8]-Maximum Heart Rate Achieved (MaxHR): Maximum heart rate achieved during exercise.
[9]-Exercise-Induced Angina (ExerciseAngina): Chest pain that occurs during exercise [Y: Yes, N: No].
[10]-Oldpeak: ST depression value, indicating potential health conditions such as hypokalemia (low potassium levels) or myocardial ischemia (reduced blood flow to the heart).
[11]-ST_Slope: Slope of the peak exercise ST segment [Up: Upsloping, Flat: Flat, Down: Downsloping].

After conducting data preprocessing steps such as cleaning, visualization, handling overfitting, scaling, and encoding (using label encoding), we applied 10 different machine learning models:

[1]-AdaBoost
[2]-Decision Tree
[3]-Gradient Boosting
[4]-K-Nearest Neighbors (KNN)
[5]-Logistic Regression
[6]-Naive Bayes
[7]-Random Forest
[8]-Stacking
[9]-Support Vector Classifier (SVC)
[10]-XGBoost

The evaluation results showed that most of these models achieved satisfactory performance,
with an average accuracy of around 90%. 
The project concluded with the deployment of the model using Streamlit.

Tools & Libraries Used:
Pandas, 
NumPy, 
Scikit-learn,
Matplotlib,
Seaborn, 
Pickle, 
Streamlit.


