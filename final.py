# import libraries
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib.colors import ListedColormap
import numpy as np

app_mode = st.sidebar.selectbox('Select the dataset',['Random','Iris']) #two pages

if app_mode == 'Random':

    # set page title
    #st.set_page_config(page_title="Streamlit App")

    st.title("ADABOOST")
    st.text("")

    # generate a 2-class classification problem with 1,000 data points,
    # where each data point is a 2-D feature vector
    (X, y) = make_blobs(n_samples=2000, n_features=2, centers=2, cluster_std=2, random_state=1)

    fig, ax = plt.subplots()

    #col1,col2 = st.columns(2)
    col1, padding, col2 = st.columns((10,2,10))
    #col1, col2 = st.columns(2, gap = "large")

    with col1:
        st.subheader("Dataset")
        ax.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=30)
        #ax.set_title("Data")
        st.pyplot(fig)

    depth = st.sidebar.slider("Depth of a Decision Tree", min_value=1,max_value=5,step=1)
    n_est = st.sidebar.slider("Number of estimators", min_value=1, max_value=500, step=1)
    lr = st.sidebar.slider("Learning Rate", min_value=0.1,max_value=5.0,step=0.1)

    # create and fit AdaBoost model
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),n_estimators=n_est, learning_rate=lr, random_state=42)
    ada.fit(X, y)

    # make predictions and calculate accuracy
    y_pred = ada.predict(X)
    acc = accuracy_score(y, y_pred)


    # create a mesh to plot decision boundaries
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    with col2:
    # plot the (testing) classification data with decision boundaries
        st.subheader("Decision boundaries")
        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        ax.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=30, cmap=plt.cm.RdYlBu)
        #ax.set_title("Data with Decision Boundaries")
        st.pyplot(fig)

    # print accuracy
    #st.write(f"Accuracy: {acc}")
    #print(f"Accuracy: {acc}")



    st.text("")
    st.text("")
    st.subheader("Performance")

    accuracy = metrics.accuracy_score(y, y_pred)
    error = metrics.mean_squared_error(y, y_pred)
    #st.write("Accuracy:  ",metrics.accuracy_score(y, y_pred))
    st.success("Accuracy :  {:.2f}".format(accuracy))
    #st.write("MSE",metrics.mean_squared_error(y, y_pred))
    #st.success("MSE      :  {:.2f}".format(error))

elif app_mode == 'Iris':
    st.title("ADABOOST")
    st.text("")

    # Load iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    col1, padding, col2 = st.columns((10,2,10))

    with col1:
    # Plot sepal length vs sepal width
        st.subheader("Dataset")
        st.text("")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_title("Iris Dataset")
        st.pyplot(fig)

    #st.set_page_config(page_title="Streamlit App")



    # Extract features and target
    X = iris.data[:, :2]  # we use only the first two features for visualization
    y = iris.target

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define base estimator





    depth = st.sidebar.slider("Depth of a Decision Tree", min_value=1,max_value=5,step=1)
    n_est = st.sidebar.slider("Number of estimators", min_value=1, max_value=500, step=1)
    lr = st.sidebar.slider("Learning Rate", min_value=0.1,max_value=5.0,step=0.1)

    # Define AdaBoost classifier
    n_estimators =n_est
    learning_rate = lr
    #base_estimator = DecisionTreeClassifier(max_depth=depth)
    ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

    # Train AdaBoost classifier
    ada_boost.fit(X_train, y_train)

    y_pred = ada_boost.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Create a meshgrid of points to plot the decision boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Predict class for each point in the meshgrid
    Z = ada_boost.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a colormap for the classes
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    with col2:
    # Plot the decision boundaries
        st.subheader("Decision boundaries")
        st.text("")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolormesh(xx, yy, Z, cmap=cmap_light)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=30)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        #ax.set_title("AdaBoost decision boundaries (n_estimators={}, learning_rate={})".format(n_estimators, learning_rate))
        st.pyplot(fig)

    # Show accuracy score

    st.text("")
    st.subheader("Performance")
    st.success("Accuracy :  {:.2f}".format(acc))