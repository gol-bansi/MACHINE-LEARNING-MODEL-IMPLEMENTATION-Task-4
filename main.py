from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Email dataset
emails = [
    "Get rich quick! Click here to win a million dollars!",
    "Hello, could you please review this document for me?",
    "Discounts on luxury watches and handbags!",
    "Meeting scheduled for tomorrow, please confirm your attendance.",
    "Congratulations, you've won a free gift card!"
]

# Labels (1 = Spam, 0 = Not Spam)
labels = [1, 0, 1, 0, 1]

# Count values
spam_count = labels.count(1)
ham_count = labels.count(0)

categories = ['Spam', 'Not Spam']
counts = [spam_count, ham_count]

# Convert text into numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Test with new email
new_email = ["You've won a free cruise vacation"]
new_email_vectorized = vectorizer.transform(new_email)

predicted_label = model.predict(new_email_vectorized)

if predicted_label[0] == 0:
    print("Predicted as Not Spam.")
else:
    print("Predicted as Spam.")

# Plot
plt.figure()
plt.bar(categories, counts)

plt.title("Spam vs Not Spam Emails")
plt.xlabel("Category")
plt.ylabel("Number of Emails")

plt.show()