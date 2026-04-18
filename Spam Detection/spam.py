import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
#Create Dataframe
df = pd.read_csv(r"D:\Old Stuff\Desktop\Projects\Spam Detection\spam_ham_dataset.csv")

#Data seperation
X = df["text"]
y = df["label"].map({"ham":0, "spam":1})

#Train_Test Split
vectorizer = TfidfVectorizer(
    stop_words = "english",
    ngram_range=(1,2),
    min_df= 2
)
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

#Data Transformation
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

#NB_Model Creation
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

#LR_Model Creation
lr_model = LogisticRegression(max_iter = 1000)
lr_model.fit(X_train, y_train)

#Streamlit UI

st.header("Spam Detection Model")

st.subheader("Select Model")
selected_model = st.radio(
    "Choose model:",
    ("Naive Bayes", "Logistic Regression")
)

st.caption(f"🔍 Current model: **{selected_model}**")

st.subheader("Message Check")
user_input = st.text_area("Enter a message to check: ")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        user_vectorised = vectorizer.transform([user_input])

        if selected_model == "Naive Bayes":
            nb_prediction = nb_model.predict(user_vectorised)
            nb_proba = nb_model.predict_proba(user_vectorised)[0]
            
            nb_not_spam_proba = nb_proba[0]
            nb_spam_proba = nb_proba[1]
            
            if nb_spam_proba > 0.54:
                st.error("Spam detected!")
                st.write("Confidence Level")
                st.progress(nb_spam_proba)
                st.write(f"{nb_spam_proba*100:.2f}%")

            elif nb_spam_proba <= 0.35:
                st.success("Not Spam")
                st.write("Confidence Level")
                st.progress(nb_not_spam_proba)
                st.write(f"{nb_not_spam_proba*100:.2f}%")

            else:
                st.warning("Uncertain - Low confidence prediction")
                st.write("Spam Probability Level")
                st.progress(nb_spam_proba)
                st.write(f"{nb_spam_proba*100:.2f}%")

        else:
            lr_prediction = lr_model.predict(user_vectorised)
            lr_proba = lr_model.predict_proba(user_vectorised)[0]

            lr_not_spam_proba = lr_proba[0]
            lr_spam_proba = lr_proba[1]

            if lr_spam_proba > 0.54:
                st.error("Spam detected!")
                st.write("Confidence Level")
                st.progress(lr_spam_proba)
                st.write(f"{lr_spam_proba*100:.2f}%")

            elif lr_spam_proba <= 0.35:
                st.success("Not Spam")
                st.write("Confidence Level")
                st.progress(lr_not_spam_proba)
                st.write(f"{lr_not_spam_proba*100:.2f}%")

            else:
                st.warning("Uncertain - Low confidence prediction")
                st.write("Spam Probability Level")
                st.progress(lr_spam_proba)
                st.write(f"{lr_spam_proba*100:.2f}%")
        

        
        
        
        

#Model Predictions
nb_y_pred = nb_model.predict(X_test)
lr_y_pred = lr_model.predict(X_test)

#Naive_Bayes Stats
nb_accuracy = accuracy_score(y_test, nb_y_pred)
nb_report = classification_report(y_test, nb_y_pred)
nb_cm = confusion_matrix(y_test, nb_y_pred)

#Logistic Regression Stats
lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_report = classification_report(y_test, lr_y_pred)
lr_cm = confusion_matrix(y_test, lr_y_pred)

st.subheader("Model Statistics")

if selected_model == "Naive Bayes":
    with st.expander("Naive Bayes Statistics"):
        st.metric(label = "Model Accuracy", value = f"{nb_accuracy:.2%}")

        with st.expander("Model Classification Report"):
            st.code(nb_report)

        with st.expander("Confusion Matrix"):
            fig, ax = plt.subplots()
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            sns.heatmap(
                nb_cm,
                annot = True,
                fmt = "d",
                cmap = "PuRd",
                xticklabels= ["Not Spam", "Spam"],
                yticklabels= ["Not Spam", "Spam"],
                ax = ax,
                annot_kws = {"color":"black"}
            )

            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.set_xlabel("Predicted Label", color = "white")
            ax.set_ylabel("Actual Label", color = "white")
            ax.set_title("Confusion Matrix", color = "white")
            
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(colors="white")
            st.pyplot(fig)

else:
    with st.expander("Logistic Regression Statistics"):
        st.metric(label = "Model Accuracy", value = f"{lr_accuracy:.2%}")

        with st.expander("Model Classification Report"):
            st.code(lr_report)
        
        with st.expander("Confusion Matrix"):
            fig, ax = plt.subplots()
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            sns.heatmap(
                lr_cm,
                annot = True,
                fmt = "d",
                cmap = "PuRd",
                xticklabels= ["Not Spam", "Spam"],
                yticklabels= ["Not Spam", "Spam"],
                ax = ax,
                annot_kws = {"color":"black"}
            )

            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.set_xlabel("Predicted Label", color = "white")
            ax.set_ylabel("Actual Label", color = "white")
            ax.set_title("Confusion Matrix", color = "white")

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(colors="white")
            st.pyplot(fig)





