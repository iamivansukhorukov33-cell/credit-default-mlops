import json  # Отчет проверки.
from pathlib import Path  # Путь отчета.
import numpy as np
import great_expectations as gx  # Data validation — проверка данных.
from src.schema import FEATURES, TARGET, PAY, PAID  # Общая схема.


def validate(frame, report=None):
    # Reject infinities as well as NaNs. / Отклоняем бесконечности и NaN.
    numeric = frame.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Great Expectations validation failed: Non-finite data")
    context = gx.get_context(mode="ephemeral")  # Независимый контекст в памяти.
    source = context.data_sources.add_pandas(name="credit")  # Источник pandas.
    asset = source.add_dataframe_asset(name="clients")  # Набор клиентов.
    definition = asset.add_batch_definition_whole_dataframe(
        "all"
    )  # Проверяем все строки.
    suite = gx.ExpectationSuite(name="credit_contract")  # Suite — набор правил.
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=["ID", *FEATURES, TARGET], exact_match=True
        )
    )  # Схема.
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1)
    )  # Не пустая таблица.
    for column in ["ID", *FEATURES, TARGET]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )  # Нет пропусков.
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInTypeList(
                column=column, type_list=["int64", "float64", "int32"]
            )
        )  # Числовой тип.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="ID")
    )  # Уникальные клиенты.
    for column, values in {
        TARGET: [0, 1],
        "SEX": [1, 2],
        "EDUCATION": [1, 2, 3, 4],
        "MARRIAGE": [1, 2, 3],
        **{c: list(range(-2, 10)) for c in PAY},
    }.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=column, value_set=values)
        )  # Коды.
    for column, lower, upper in [
        ("AGE", 18, 100),
        ("LIMIT_BAL", 1, 2000000),
        *[(c, 0, None) for c in PAID],
    ]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column, min_value=lower, max_value=upper
            )
        )  # Диапазоны.
    suite = context.suites.add(suite)  # Регистрируем правила.
    validation = gx.ValidationDefinition(
        name="credit_check", data=definition, suite=suite
    )  # Связываем данные и правила.
    result = context.validation_definitions.add(validation).run(
        batch_parameters={"dataframe": frame}
    )  # Проверка.
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)  # Папка отчета.
        Path(report).write_text(
            json.dumps(result.to_json_dict(), indent=2, default=str)
        )  # Результат GE.
        Path("reports/expectation_suite.json").write_text(
            json.dumps(suite.to_json_dict(), indent=2, default=str)
        )  # Экспорт правил.
    if not result.success:
        raise ValueError(
            "Great Expectations validation failed"
        )  # Останавливаем pipeline при аномалии.
    return result  # Успешная проверка.
