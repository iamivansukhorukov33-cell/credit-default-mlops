import argparse  # Аргументы командной строки.
import hashlib  # Контрольная сумма исходного файла.
import io  # Чтение архива в памяти.
import json  # Сохранение отчета.
from pathlib import Path  # Работа с путями.
import urllib.request  # Загрузка открытых данных.
import zipfile  # Чтение ZIP.
import pandas as pd  # Таблицы данных.
from sklearn.model_selection import train_test_split  # Разделение выборок.
from src.schema import FEATURES, TARGET  # Единая схема.
from src.validation import validate  # Правила качества.

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"  # Оригинал UCI.


def download():
    folder = Path("data/raw")  # Сырые данные.
    folder.mkdir(parents=True, exist_ok=True)  # Создаем папку.
    archive = urllib.request.urlopen(URL, timeout=60).read()  # Загружаем архив.
    (folder / "source.zip").write_bytes(archive)  # Сохраняем оригинал.
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xls"))  # Ищем Excel.
        frame = pd.read_excel(
            io.BytesIO(z.read(name)), header=1
        )  # Пропускаем X1...X23.
    frame.to_csv(folder / "credit.csv", index=False)  # Общий формат CSV.
    metadata = {
        "url": URL,
        "sha256": hashlib.sha256(archive).hexdigest(),
        "rows": len(frame),
    }  # Происхождение.
    (folder / "source.json").write_text(
        json.dumps(metadata, indent=2)
    )  # Записываем паспорт.


def clean(frame):
    frame = frame.copy()  # Не меняем исходную таблицу.
    frame.columns = frame.columns.str.strip()  # Удаляем пробелы.
    frame = frame.rename(
        columns={
            "default payment next month": TARGET,
            "default.payment.next.month": TARGET,
        }
    )  # Имя цели.
    if set(frame.columns) != set(["ID", *FEATURES, TARGET]):
        raise ValueError("Unexpected columns")  # Не скрываем ошибки схемы.
    if frame["ID"].duplicated().any():
        raise ValueError(
            "Duplicate client ID"
        )  # Не допускаем одного клиента в разных выборках.
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")  # Проверяем числа.
    frame["EDUCATION"] = frame["EDUCATION"].replace(
        {0: 4, 5: 4, 6: 4}
    )  # Неописанные коды: «другое».
    frame["MARRIAGE"] = frame["MARRIAGE"].replace(
        {0: 3}
    )  # Неописанный статус: «другое».
    return frame  # Отрицательные счета сохраняем: это возможная переплата.


def prepare():
    frame = clean(pd.read_csv("data/raw/credit.csv"))  # Очистка.
    validate(frame, "reports/validation.json")  # Проверяем до разделения.
    train, test = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=42
    )  # Сохраняем долю классов.
    Path("data/processed").mkdir(parents=True, exist_ok=True)  # Папка результатов.
    train.to_csv("data/processed/train.csv", index=False)  # Обучение.
    test.to_csv("data/processed/test.csv", index=False)  # Отложенная оценка.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Командный интерфейс.
    parser.add_argument("action", choices=["download", "prepare"])  # Выбор этапа.
    {"download": download, "prepare": prepare}[parser.parse_args().action]()  # Запуск.
