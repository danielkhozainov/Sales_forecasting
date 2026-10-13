# Прогнозирование продаж «Прилавок»

## Название базы данных

**БД:**(PostgreSQL, Yandex Cloud)

---

## Описание проекта

### Бизнес-задача

Розничная сеть «Прилавок» нуждается в еженедельном прогнозе продаж по каждому магазину и отделу. Точный прогноз позволяет:

- оптимизировать закупки и логистику;
- сократить излишки и дефицит товаров;
- планировать персонал и маркетинговые акции.

### Цель

Построить pipeline batch-инференса, который каждое воскресенье в 20:00:

1. Загружает исторические и плановые данные из PostgreSQL.
2. Предобрабатывает признаки (идентично обучению).
3. Применяет модель CatBoost для прогнозирования.
4. Записывает предсказания в витрину `predictions` в PostgreSQL.

### Подход

- Обучение модели выполнено в Jupyter Notebook (исследовательская часть).
- Логика предобработки вынесена в отдельный модуль `preprocessing.py`.
- Инференс автоматизирован через Airflow DAG.
- Модель хранится в Object Storage (S3) Yandex Cloud.
- Креды хранятся в Airflow Connections и Variables (не в коде).

---

## Описание данных

База данных PostgreSQL содержит четыре таблицы:

### Таблица `sales`

| Колонка | Тип | Описание |
|---|---|---|
| `store` | integer | Номер магазина |
| `dept` | integer | Номер отдела |
| `date` | date | Дата начала недели |
| `weekly_sales` | double precision | Продажи за неделю |
| `is_holiday` | boolean | Признак праздничной недели |

### Таблица `plan`

| Колонка | Тип | Описание |
|---|---|---|
| `store` | integer | Номер магазина |
| `dept` | integer | Номер отдела |
| `date` | date | Дата начала недели |
| `is_holiday` | boolean | Признак праздничной недели |

`weekly_sales` в `plan` отсутствует — это целевая переменная для предсказания.

### Таблица `stores`

| Колонка | Тип | Описание |
|---|---|---|
| `store` | integer | Номер магазина |
| `type` | varchar | Тип магазина (A, B, C) |
| `size` | integer | Площадь магазина |

### Таблица `features`

| Колонка | Тип | Описание |
|---|---|---|
| `store` | integer | Номер магазина |
| `dept` | integer | Номер отдела |
| `date` | date | Дата |
| `temperature` | double precision | Средняя температура |
| `fuel_price` | double precision | Цена топлива |
| `factor1`–`factor5` | double precision | Дополнительные макроэкономические факторы |
| `cpi` | double precision | Индекс потребительских цен |
| `unemployment` | double precision | Уровень безработицы |

---

## Описание признаков

### Базовые признаки (из БД)

| Признак | Тип | Источник |
|---|---|---|
| `store` | категориальный | sales / plan |
| `dept` | категориальный | sales / plan |
| `is_holiday` | числовой (0/1) | sales / plan |
| `type` | категориальный | stores |
| `size` | числовой | stores |
| `temperature` | числовой | features |
| `fuel_price` | числовой | features |
| `cpi` | числовой | features |
| `unemployment` | числовой | features |

### Сгенерированные признаки (модуль `preprocessing.py`)

**Временные признаки** (`create_temporal_features`):

| Признак | Описание |
|---|---|
| `year` | Год |
| `month` | Месяц |
| `week` | Номер недели ISO |
| `day_of_week` | День недели (0–6) |
| `is_weekend` | Выходной (1) / будний (0) |

**Средние продажи** (`create_avg_sales_feature`):

| Признак | Описание |
|---|---|
| `avg_sales_before` | Кумулятивное среднее продаж по паре store–dept до текущей недели |

**Лаговые признаки** (`create_lag_features`):

| Признак | Описание |
|---|---|
| `sales_1week_ago` | Продажи 1 неделю назад |
| `sales_2week_ago` | Продажи 2 недели назад |
| `sales_4week_ago` | Продажи 4 недели назад |

**Скользящие статистики** (`create_rolling_features`):

| Признак | Описание |
|---|---|
| `mean_sales_2week` | Скользящее среднее за 2 недели (исключая текущую) |
| `mean_sales_4week` | Скользящее среднее за 4 недели (исключая текущую) |

### Предобработка (`preprocess_data`)

1. **Аномальные продажи:** отрицательные значения `weekly_sales` заменяются на 0.
2. **Пропуски:** числовые колонки заполняются средним, категориальные — модой.
3. **Обогащение:** последовательно вызываются функции создания признаков.

---

## Логика обучения

### Модель

**CatBoostRegressor** (градиентный бустинг над деревьями решений).

Категориальные признаки: `store`, `dept`, `type`.

### Гиперпараметры

| Параметр | Значение |
|---|---|
| `iterations` | 1000 |
| `learning_rate` | 0.05 |
| `depth` | 8 |
| `loss_function` | RMSE |
| `random_state` | 42 |
| `early_stopping_rounds` | 50 |

### Метрики качества

| Метрика | Значение (на валидации) |
|---|---|
| RMSE | 1,497.40 |
| MAE | 3,120.16 |
| R² | 0.9795 |


## Архитектура пайплайна

### Схема DAG

```
load_data ──→ preprocess_features ──→ run_inference ──→ save_predictions
                                          ↑
load_model ───────────────────────────────┘
```

`load_data` и `load_model` выполняются параллельно. `run_inference` ждёт оба, затем `save_predictions` записывает результат.

### Описание задач

| Задача | Функция | Описание |
|---|---|---|
| `load_data` | `load_data_from_postgres` | Создаёт временную таблицу `inference_data_temp` с объединением sales + plan + stores + features. Передаёт первую дату плана через XCom. |
| `preprocess_features` | `preprocess_features` | Читает временную таблицу, вызывает `preprocess_data`, фильтрует строки плана, заполняет лаги нулями, удаляет NaN. Передаёт датафрейм через XCom (`.to_json()`). Удаляет временную таблицу. |
| `load_model` | `load_model_from_s3` | Загружает `catboost_model.pkl` из S3 во временный файл. Передаёт путь через XCom. |
| `run_inference` | `run_batch_inference` | Загружает датафрейм и модель, выполняет `model.predict(Pool(...))`, обрабатывает отрицательные предсказания (→ 0). Передаёт результат через XCom. |
| `save_predictions` | `save_predictions_to_postgres` | Записывает предсказания в таблицу `predictions` (upsert). Выводит количество строк и первые 5 записей. |

### Источники данных

| Источник | Назначение |
|---|---|
| PostgreSQL (`sales`, `plan`, `stores`, `features`) | Источник сырых данных |
| Yandex Cloud Object Storage (S3) | Хранение обученной модели |
| Airflow Variables | S3-креды и параметры |
| Airflow Connections (`postgres_sales_db`) | Подключение к PostgreSQL |

### Место хранения артефактов

| Артефакт | Расположение |
|---|---|
| Обученная модель | S3: `s3-ds-20260915-7e2a105f57/catboost_model.pkl` |
| Предсказания | PostgreSQL: таблица `predictions` |
| Логи выполнения | Airflow (task logs) |

### Расписание

DAG запускается каждое воскресенье в 20:00 UTC (`schedule_interval='0 20 * * 0'`).

---

## Инструкция по запуску

### 1. Настройка окружения

```bash
pip install -r requirements.txt
```

### 2. Обучение модели

1. Откройте Jupyter Notebook с исследовательской частью.
2. Выполните ячейки: загрузка данных → EDA → предобработка → обучение CatBoost → оценка метрик → PSI-анализ.
3. Сохраните модель в S3:

```python
import io, pickle, boto3

s3_client = boto3.client(
    's3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=os.getenv('S3_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
    region_name='ru-central1'
)

filebuffer = io.BytesIO()
pickle.dump(final_model, filebuffer)
filebuffer.seek(0)

s3_client.upload_fileobj(
    Fileobj=filebuffer,
    Bucket='s3-ds-20260915-7e2a105f57',
    Key='catboost_model.pkl'
)
```

### 3. Настройка Airflow

#### Connections (Admin → Connections)

| Параметр | Значение |
|---|---|
| Conn Id | `postgres_sales_db` |
| Conn Type | Postgres |
| Host | *хост БД* |
| Schema | *название БД* |
| Login | *логин* |
| Password | *пароль* |

#### Variables (Admin → Variables)

| Key | Value |
|---|---|
| `s3_bucket_name` | `s3-ds-20260915-7e2a105f57` |
| `s3_access_key` | *access key* |
| `s3_secret_key` | *secret key* |
| `s3_model_key` | `catboost_model.pkl` |


---

