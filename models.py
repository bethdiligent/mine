def train_decision_tree(X, y):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model

def train_svm(X, y):
    from sklearn.svm import SVC
    model = SVC(random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y, model_name):
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:\n{report}")
    return accuracy, report