import pytest
import yaml
import builtins
from src.training.training import main
import mlflow


# Dummy run pour patcher mlflow.start_run
class DummyRun:
    info = type("Info", (), {"run_id": "dummy_run_id"})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def patch_mlflow(monkeypatch):
    monkeyatch_funcs = {
        "set_tracking_uri": lambda uri: None,
        "set_experiment": lambda name: None,
        "start_run": lambda run_name=None: DummyRun(),
        "log_metric": lambda metric, value: None,
        "log_param": lambda param, value: None,
    }
    monkeypatch.setattr(
        mlflow, "set_tracking_uri", monkeyatch_funcs["set_tracking_uri"]
    )
    monkeypatch.setattr(mlflow, "set_experiment", monkeyatch_funcs["set_experiment"])
    monkeypatch.setattr(mlflow, "start_run", monkeyatch_funcs["start_run"])
    monkeypatch.setattr(mlflow, "log_metric", monkeyatch_funcs["log_metric"])
    monkeypatch.setattr(mlflow, "log_param", monkeyatch_funcs["log_param"])
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *args, **kwargs: None)


def test_main_pipeline_execution(
    tmp_path, training_config, training_csv, patch_mlflow, monkeypatch
):
    """
    Test d'intégration simple de main() :
    - Met à jour le fichier de configuration temporaire pour y insérer le chemin du CSV d'entraînement.
    - Patch builtins.open pour rediriger "params.yaml" vers le fichier temporaire.
    - Exécute main() et vérifie qu'aucune exception n'est levée.
    """
    # Mise à jour de la configuration pour utiliser le CSV d'entraînement
    config = yaml.safe_load(training_config.read_text())
    config["train"]["input"] = str(training_csv)
    training_config.write_text(yaml.dump(config))

    original_open = builtins.open

    def fake_open(filename, *args, **kwargs):
        if filename == "params.yaml":
            return original_open(str(training_config), *args, **kwargs)
        return original_open(filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    # Exécuter main() et vérifier qu'elle se termine sans exception
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() a levé une exception inattendue : {e}")
