import streamlit as st
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model
model_file = 'suicide_detection_model.joblib'
pipeline = joblib.load(model_file)

# Data preprocessing function
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(tokens)

stop_words = set(stopwords.words("english"))

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
        st.success("Email sent successfully")
    except smtplib.SMTPException as e:
        st.error(f"Failed to send email: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit application
st.title("Suicide Detection and Notification")

# Text input for user text
user_input = st.text_area("Enter a text to classify:")

# Text input for user email
user_email = st.text_input("Enter your email address:")

if st.button("Classify and Notify"):
    if user_input and user_email:
        result = classify_user_input(user_input)
        if result == "suicidal":
            send_email(user_email, 'Suicidal Alert', 'A user has been classified as suicidal.')
        st.write(f"Category: {result}")
    else:
        st.error("Please enter both text and email address.")
