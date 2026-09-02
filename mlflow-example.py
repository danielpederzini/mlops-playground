import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    mlflow.log_metric("accuracy", model.score(X_test, y_test))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="random-forest-model",
        input_example=X_train[:5]
    )
    
    mlflow.end_run()
