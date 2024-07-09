import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
csv_file_path = r'C:\Users\abhic\Downloads\Suicide_Detection.csv'

try:
    df = pd.read_csv(csv_file_path, on_bad_lines='skip', quoting=1)
except pd.errors.ParserError:
    print("Error parsing the CSV file. Please check the file for formatting issues.")
    df = pd.read_csv(csv_file_path, on_bad_lines='skip', quoting=1)

# Check the first few rows of the dataframe to verify column names
print(df.head())

# Ensure the correct column names are used
text_column = 'text'  # Replace with the actual column name if different
label_column = 'class'  # Replace with the actual column name if different

# Limiting the dataset for faster processing (adjust as needed)
df = df.iloc[:10000, :]

# Balance the dataset if needed
class_counts = df[label_column].value_counts()
min_class_count = class_counts.min()
balanced_df = df.groupby(label_column).apply(lambda x: x.sample(min_class_count)).reset_index(drop=True)


# Data preprocessing
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(tokens)


stop_words = set(stopwords.words("english"))
balanced_df[text_column] = balanced_df[text_column].apply(preprocess)

# Splitting data into train and test sets (70-30 split)
X = balanced_df[text_column]
y = balanced_df[label_column].apply(lambda x: 1 if x == 'suicide' else 0)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the pipeline with Logistic Regression classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),  # Uni-gram and bi-gram CountVectorizer
    ('tfidf', TfidfTransformer()),  # TF-IDF Transformer
    ('clf', LogisticRegression(max_iter=1000))  # Logistic Regression Classifier
])

# Fit the pipeline on the training data
pipeline.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred = pipeline.predict(x_test)
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# Function to classify into custom categories based on your criteria
def classify_text(label):
    if label == 1:
        return "suicidal"  # Positive class label
    else:
        return "non-suicidal"  # Negative class label


# Function to process user input and classify using Logistic Regression
def classify_user_input(input_text):
    # Preprocess the input text similar to the training data
    input_text = preprocess(input_text)

    # Predict using the model
    label = pipeline.predict([input_text])[0]

    # Classify based on predicted label
    return classify_text(label)


# Function to send an email if the user is classified as suicidal
def send_email(to_email, subject, message):
    from_email = 'guruvamshi1718@gmail.com'
    from_password = 'ezct wotf fnjw anrc'  # Use the app password generated from your Google account

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Use the classification function and send email if classified as "suicidal"
def classify_and_notify(input_text, user_email):
    result = classify_user_input(input_text)

    if result == "suicidal":
        send_email(user_email, 'Suicidal Alert', 'A user has been classified as suicidal.')

    return result


# Example of using the classification and notification function with user input
user_input = input("Enter a text to classify: ")
user_email = input("Enter your email address: ")
result = classify_and_notify(user_input, user_email)

print(f"Input Text: {user_input}")
print(f"Category: {result}")

import joblib

# Fit the pipeline on the training data
pipeline.fit(x_train, y_train)

# Save the trained model to a file
model_file = 'suicide_detection_model.joblib'
joblib.dump(pipeline, model_file)

print(f"Model saved to {model_file}")