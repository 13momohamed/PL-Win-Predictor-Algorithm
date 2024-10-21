import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

# Load and preprocess the training and test data
TRAIN_DATA = pd.read_csv('TRAIN_DATA.csv')
TRAIN_DATA.fillna(0, inplace=True)

TEST_DATA = pd.read_csv('TEST_DATA.csv')
TEST_DATA.fillna(0, inplace=True)

# Step 1: Create a binary target variable for win (1 = win, 0 = loss/draw)
TRAIN_DATA['Win'] = (TRAIN_DATA['Goals'] > TRAIN_DATA['Opponent Goals']).astype(int)
TEST_DATA['Win'] = (TEST_DATA['Goals'] > TEST_DATA['Opponent Goals']).astype(int)


selected_columns = ['Goals', 'Shots', 'Opponent Shots', 'Shots on Target', 'Opponent Shots on Target',
                    'Corners', 'Opponent Corners', 'Win Rate', 'Draw Rate', 'Loss Rate',
                    'Total Wins Last N', 'Total Losses Last N', 'Total Draws Last N', 
                    'Recent Goals Scored', 'Recent Goals Conceded', 
                    'Average Goals Scored', 'Average Goals Conceded']


x_train = TRAIN_DATA[selected_columns]
y_train = TRAIN_DATA['Win']

x_test = TEST_DATA[selected_columns]
y_test = TEST_DATA['Win']

# Step 3: Handle class imbalance using class weights
# Logistic Regression with class weighting
logistic_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
logistic_model.fit(x_train, y_train)

# Predict and evaluate Logistic Regression
y_pred_logistic = logistic_model.predict(x_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"Accuracy of Logistic Regression model: {accuracy_logistic * 100:.2f}%")
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logistic))
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred_logistic))

# Step 8: Add predictions to the test set
TEST_DATA['Predicted Win'] = y_pred_logistic

# Display some sample rows from the test set
print(TEST_DATA[['Season', 'Goals', 'Opponent Goals', 'Predicted Win']].head())

TEST_DATA[['Season', 'Goals', 'Opponent Goals', 'Predicted Win']].to_csv('Predicted_Wins.csv', index=False)
