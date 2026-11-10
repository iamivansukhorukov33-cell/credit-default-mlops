"""Relocate bundled MLflow artifacts. / Обновить пути артефактов после распаковки."""

import sqlite3
from pathlib import Path


def relocate():
    root = Path.cwd().resolve()
    database = root / "mlflow.db"
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(database) as connection:
        for table, column in [
            ("experiments", "artifact_location"),
            ("runs", "artifact_uri"),
            ("logged_models", "artifact_location"),
        ]:
            columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if column not in columns:
                continue
            for rowid, uri in connection.execute(
                f"SELECT rowid, {column} FROM {table}"
            ).fetchall():
                if not uri:
                    continue
                for folder in ("mlruns", "mlartifacts"):
                    marker = f"/{folder}/"
                    if marker in uri:
                        relative = folder + "/" + uri.split(marker, 1)[1]
                        new_uri = (root / relative).as_uri()
                        connection.execute(
                            f"UPDATE {table} SET {column}=? WHERE rowid=?",
                            (new_uri, rowid),
                        )
                        break
    print("MLflow artifact paths updated:", root)


if __name__ == "__main__":
    relocate()
