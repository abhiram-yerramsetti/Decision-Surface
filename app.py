import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification,make_circles, make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Create a synthetic dataset
# 'circles',
data = st.sidebar.selectbox('Select Type of Data ', ('classification','blobs', 'moons','circles'))

if data == 'classification':
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
elif data == 'circles':
    X, y = make_circles(n_samples=100, factor=0.5, noise=0.05)
    
elif data == 'blobs':
    X,y = make_blobs(n_samples=250, centers=2, n_features=2, cluster_std=1.0, random_state=42)
    
elif data == 'moons':
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    






# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def plot_decision_surface(X, y, model,title):
    plot_decision_regions(X,y,clf = model)
    plt.show()
    plt.title(title)

# Streamlit application
st.title('Decision Surfaces of KNN, Naive Bayes, and Logistic Regression')

# Dropdown menu to select the classifier
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'Naive Bayes', 'Logistic Regression'))

# Add input for KNN parameters
if classifier_name == 'KNN':
    n_neighbors = st.sidebar.slider('Number of Neighbors (k)', 1, 15, 3)
    weights = st.sidebar.selectbox('Weight Function', ('uniform', 'distance'))
    algorithm = st.sidebar.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    n_jobs = st.sidebar.number_input('Number of Parallel Jobs (n_jobs)', -1, 10, 1)
    
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)
    knn.fit(X_train, y_train)
    plot_decision_surface(X, y, knn,'KNeighborsClassifier')
    st.pyplot(plt,clear_figure=True)

elif classifier_name == 'Naive Bayes':
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    plot_decision_surface(X, y, nb, 'Naive Bayes Decision Surface')
    st.pyplot(plt,clear_figure=True)
else:
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    plot_decision_surface(X, y, lr, 'Logistic Regression Decision Surface')
    st.pyplot(plt,clear_figure=True)
# Display the plot in Streamlit
# st.pyplot(plt)
