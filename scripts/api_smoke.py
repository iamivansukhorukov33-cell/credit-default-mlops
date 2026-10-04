import json  # Пример клиента.
from pathlib import Path  # Файл примера.
import httpx  # HTTP-клиент.

if __name__ == "__main__":
    payload = json.loads(Path("examples/client.json").read_text())  # Все 23 признака.
    response = httpx.post(
        "http://localhost:8000/predict", json=payload, timeout=30
    )  # Обращение к сервису.
    response.raise_for_status()  # Падаем при ошибке.
    result = response.json()  # Ответ API.
    assert result["default_prediction"] in [0, 1]  # Проверка класса.
    assert 0 <= result["default_probability"] <= 1  # Проверка вероятности.
    print(result)  # Результат демонстрации.
