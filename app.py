import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Load model and data
# --------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

df = load_data()
model = load_model()

# --------------------
# Sidebar Navigation
# --------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Data Exploration", "Visualizations", "Model Prediction", "Model Performance", "About"])

# --------------------
# Home Section
# --------------------
if menu == "Home":
    st.title("Titanic Survival Prediction App")
    st.markdown("""
    This app predicts whether a passenger survived the Titanic disaster using a trained machine learning model.
    
    **Features:**
    - Explore the Titanic dataset
    - View interactive charts
    - Input passenger details for prediction
    - See model performance metrics
    """)

# --------------------
# Data Exploration
# --------------------
elif menu == "Data Exploration":
    st.header("Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.dataframe(df.head())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Filter Data")
    pclass_filter = st.selectbox("Select Pclass to view", options=[1, 2, 3])
    st.dataframe(df[df['Pclass'] == pclass_filter].head())

# --------------------
# Visualizations
# --------------------
elif menu == "Visualizations":
    st.header("Visualizations")

    st.subheader("Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Survival by Pclass")
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax)
    st.pyplot(fig)

# --------------------
# Model Prediction
# --------------------
elif menu == "Model Prediction":
    st.header("Predict Passenger Survival")

    pclass = st.selectbox('Pclass', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.slider('Age', 0, 80, 25)
    sibsp = st.number_input('Siblings/Spouses Aboard', 0, 10, 0)
    parch = st.number_input('Parents/Children Aboard', 0, 10, 0)
    fare = st.number_input('Fare', 0.0, 600.0, 32.0)
    embarked_Q = st.selectbox('Embarked Q', [0, 1])
    embarked_S = st.selectbox('Embarked S', [0, 1])

    # Convert Sex to numeric
    sex = 0 if sex == 'male' else 1

    # Create DataFrame for prediction
    input_df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S]],
                            columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S'])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][prediction] if hasattr(model, "predict_proba") else None

        result = "Survived" if prediction == 1 else "Did not survive"
        st.success(f"Prediction: {result}")
        if prob is not None:
            st.info(f"Confidence: {prob:.2f}")

# --------------------
# Model Performance
# --------------------
elif menu == "Model Performance":
    st.header("Model Evaluation")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Preprocess data like in training
    df_perf = df.copy()
    df_perf['Sex'] = df_perf['Sex'].map({'male': 0, 'female': 1})
    df_perf['Age'].fillna(df_perf['Age'].median(), inplace=True)
    df_perf['Embarked'].fillna(df_perf['Embarked'].mode()[0], inplace=True)
    df_perf = pd.get_dummies(df_perf, columns=['Embarked'], drop_first=True)
    df_perf.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

    X = df_perf.drop('Survived', axis=1)
    y = df_perf['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write("**Accuracy:**", acc)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# --------------------
# About
# --------------------
elif menu == "About":
    st.header("About This App")
    st.write("""
    **Developer:** Nithya  
    **Dataset:** Titanic Survival Data  
    **Framework:** Streamlit  
    **Description:**  
    A complete machine learning pipeline from EDA to model deployment on Streamlit Cloud.
    """)
