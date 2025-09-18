import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import streamlit as st

def tree_interpret(X, y):
    # DecisionTree depth tuning
    depths = range(1,11)
    cv_scores = [cross_val_score(
                    DecisionTreeClassifier(max_depth=d, random_state=42),
                    X, y, cv=5).mean() for d in depths]
    fig, ax = plt.subplots()
    ax.plot(depths, cv_scores, marker="o")
    ax.set(xlabel="max_depth", ylabel="CV accuracy", title="Decision Tree tuning")
    st.pyplot(fig)

    # RandomForest tuning (grid)
    param_grid = {"max_depth":[None,4,6,8], "n_estimators":[100,150,200]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid, cv=5, n_jobs=-1)
    grid.fit(X,y)
    st.write("Best RF params:", grid.best_params_)

    # Final model
    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(**grid.best_params_, random_state=42, n_jobs=-1)
    rf.fit(X_train,y_train)
    st.write("Test Accuracy:", rf.score(X_test,y_test))
    st.text(classification_report(y_test, rf.predict(X_test)))

    # Feature Importance
    fig2, ax2 = plt.subplots()
    pd.Series(rf.feature_importances_, index=X.columns).plot.barh(ax=ax2)
    ax2.set_title("Feature Importance (Random Forest)")
    ax2.set_xlabel("Feature importances")

    st.write(f"Feature importances Values: ")

    st.write(f"Recency = {rf.feature_importances_[0]:.4f},")
    st.write(f"Frequency: {rf.feature_importances_[1]:.4f},")
    st.write(f"Monetary: {rf.feature_importances_[2]:.4f}")

    st.pyplot(fig2)
