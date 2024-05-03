import mlflow
import os

def check_non_regression():
    # Enable tracking to our remote MLFlow server
    mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))

    # Récupérer les informations sur les expériences
    best_experiment = mlflow.search_runs(order_by=["metrics.accuracy DESC"], max_results=1)
    last_experiment = mlflow.search_runs(order_by=["metrics.created DESC"], max_results=1)

    # Récupérer les métriques du best et du last
    best_accuracy = best_experiment.loc[0, "metrics.accuracy"]
    last_accuracy = last_experiment.loc[0, "metrics.accuracy"]
    print(f"last accuracy, best accuracy :{last_accuracy, best_accuracy}")

    # Vérifier si le dernier entraînement est meilleur que le précédent
    if last_accuracy == best_accuracy:
        print("Regression test passed! The new model is better than the previous one.")
    else:
        print("Regression test failed! The new model is not better than the previous one.")
    
    assert (last_accuracy >= best_accuracy) == True

if __name__ == "__main__":
    check_non_regression()