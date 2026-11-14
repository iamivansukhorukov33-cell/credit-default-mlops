# Запуск готового проекта

Python 3.12. Откройте терминал в этой папке после распаковки ZIP.
Данные (30 000 клиентов), обученная модель, MLflow и DVC-кеш уже приложены.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DVC_SITE_CACHE_DIR="$PWD/.dvc/site-cache"
export MPLCONFIGDIR="$PWD/.mplconfig"
python -m scripts.relocate_mlflow
pytest -q
python -m scripts.verify_artifacts
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

В другом терминале активируйте окружение и выполните:

```bash
python -m scripts.api_smoke
python -m src.monitor
```

Swagger: http://127.0.0.1:8000/docs.
MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.
Повторное обучение: `dvc repro --force train`. Это создаст пять новых запусков.
Для запуска в Docker сначала остановите локальный uvicorn, чтобы освободить порт. Затем выполните `./scripts/run_docker.sh` и повторите `python -m scripts.api_smoke` в другом терминале.

Результаты текущей проверки: `reports/verification.md`.
Полные объяснения, ограничения, GitHub и внешний DVC remote: `README.md`.
