"""
DAG для batch-инференса продаж "Прилавок"
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook
import pandas as pd
import numpy as np
import boto3
from catboost import CatBoostRegressor
import logging
import io
import pickle
import os
import tempfile

# ========== Функция предобработки данных ==========
from preprocessing import preprocess_data

# ========== Конфигурация ==========

S3_BUCKET = Variable.get("s3_bucket_name", default_var="your-bucket-name")
S3_ACCESS_KEY = Variable.get("s3_access_key", default_var=None)
S3_SECRET_KEY = Variable.get("s3_secret_key", default_var=None)
S3_MODEL_KEY = Variable.get("s3_model_key", default_var="catboost_sales_model.pkl")

POSTGRES_CONN_ID = "postgres_sales_db"

RANDOM_STATE = 42
CAT_FEATURES = ['store', 'dept', 'type']

logger = logging.getLogger(__name__)


# ========== Задачи DAG ==========

def load_data_from_postgres(**context):
    """
    Загрузка данных для инференса из Postgres.
    Создаём временную таблицу inference_data_temp с объединёнными данными.
    """
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    # Удаляем старую временную таблицу, если есть
    cursor.execute("DROP TABLE IF EXISTS inference_data_temp;")
    conn.commit()

    # Создаём временную таблицу: объединяем sales + plan + stores + features
    create_table_query = """
    CREATE TABLE inference_data_temp AS
    SELECT 
        s.store,
        s.dept,
        s.date,
        s.weekly_sales,
        s.is_holiday,
        st.type,
        st.size,
        f.temperature,
        f.fuel_price,
        f.factor1,
        f.factor2,
        f.factor3,
        f.factor4,
        f.factor5,
        f.cpi,
        f.unemployment
    FROM (
        SELECT store, dept, date, weekly_sales, is_holiday
        FROM sales
        UNION ALL
        SELECT store, dept, date, NULL::float AS weekly_sales, is_holiday
        FROM plan
    ) s
    LEFT JOIN stores st ON s.store = st.store
    LEFT JOIN features f ON s.store = f.store AND s.dept = f.dept AND s.date = f.date;
    """
    cursor.execute(create_table_query)
    conn.commit()

    # Выводим информацию о таблице
    cursor.execute("SELECT COUNT(*) FROM inference_data_temp;")
    row_count = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(date), MAX(date) FROM inference_data_temp;")
    min_date, max_date = cursor.fetchone()

    logger.info(f"inference_data_temp: {row_count} строк")
    logger.info(f"Минимальная дата: {min_date}")
    logger.info(f"Максимальная дата: {max_date}")

    # Первая дата плана — по ней разделяем данные
    cursor.execute("SELECT MIN(date) FROM plan;")
    plan_first_date = cursor.fetchone()[0]

    logger.info(f"Первая дата плана: {plan_first_date}")

    cursor.close()
    conn.close()

    # Передаём первую дату плана в XCom
    context['ti'].xcom_push(key='plan_first_date', value=str(plan_first_date))


def preprocess_features(**context):
    """
    Предобработка признаков для инференса.
    """
    # 1. Читаем первую дату плана из XCom
    plan_first_date = context['ti'].xcom_pull(
        task_ids='load_data', key='plan_first_date'
    )
    logger.info(f"Первая дата плана из XCom: {plan_first_date}")

    # 2. Читаем данные из таблицы inference_data_temp
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM inference_data_temp;")
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)

    logger.info(f"Загружено из inference_data_temp: {df.shape}")

    # 3. Предобработка признаков
    df = preprocess_data(df)

    # 4. Оставляем только строки плана
    df['date'] = pd.to_datetime(df['date'])
    plan_date = pd.to_datetime(plan_first_date)
    df = df[df['date'] >= plan_date].copy()

    logger.info(f"Строки плана после фильтрации: {df.shape}")

    # 5. Удаляем колонку weekly_sales (здесь она null)
    if 'weekly_sales' in df.columns:
        df.drop(columns=['weekly_sales'], inplace=True)

    # 6. Заполняем лаговые переменные нулями для первых недель
    lag_cols = [col for col in df.columns if 'week_ago' in col or 'mean_sales' in col or 'avg_sales' in col]
    for col in lag_cols:
        df[col] = df[col].fillna(0)

    # 7. Удаляем строки с оставшимися NaN
    df = df.dropna()
    logger.info(f"Финальный размер для инференса: {df.shape}")

    # 8. Передаём датафрейм в XCom
    context['ti'].xcom_push(key='inference_df', value=df.to_json(date_format='iso'))

    # 9. Удаляем временную таблицу
    cursor.execute("DROP TABLE IF EXISTS inference_data_temp;")
    conn.commit()

    cursor.close()
    conn.close()


def load_model_from_s3(**context):
    """
    Загрузка обученной модели CatBoost из S3 (Yandex Cloud) через pickle.
    """
    s3_client = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name='ru-central1'
    )

    # Загружаем модель из S3 в буфер
    response = s3_client.get_object(
        Bucket=S3_BUCKET,
        Key=S3_MODEL_KEY
    )
    model_data = response['Body'].read()

    # Сохраняем во временный файл
    model_path = os.path.join(tempfile.gettempdir(), 'catboost_model.pkl')
    with open(model_path, 'wb') as f:
        f.write(model_data)

    logger.info(f"Модель загружена из S3 и сохранена в {model_path}")

    # Передаём путь в XCom
    context['ti'].xcom_push(key='model_path', value=model_path)



def run_batch_inference(**context):
    """
    Batch-инференс: применение модели к подготовленным данным.
    """
    from catboost import Pool
    import io as io_module

    # 1. Загружаем данные из XCom
    inference_json = context['ti'].xcom_pull(
        task_ids='preprocess_features', key='inference_df'
    )
    model_path = context['ti'].xcom_pull(
        task_ids='load_model', key='model_path'
    )

    # 2. Читаем датафрейм (обёртка в StringIO убирает FutureWarning)
    df = pd.read_json(io_module.StringIO(inference_json))
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Данные для инференса: {df.shape}")

    # 3. Загружаем модель
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info("Модель загружена")

    # 4. Определяем признаки
    exclude = ['date', 'weekly_sales', 'predicted_weekly_sales']
    features = [col for col in df.columns if col not in exclude]

    # is_holiday → число (как при обучении)
    if 'is_holiday' in df.columns:
        df['is_holiday'] = df['is_holiday'].astype(int)


    # Категориальные признаки приводим к строке
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 5. Предсказание через Pool с указанием категориальных признаков
    cat_features = [col for col in CAT_FEATURES if col in features]
    pool = Pool(df[features], cat_features=cat_features)
    predictions = model.predict(pool)

    # 6. Сохраняем предсказания
    df['predicted_weekly_sales'] = predictions

    # Обрабатываем аномальные предсказания — отрицательные заменяем на 0
    df.loc[df['predicted_weekly_sales'] < 0, 'predicted_weekly_sales'] = 0

    logger.info(f"Предсказания готовы: {df.shape}")
    logger.info(f"Предсказания: min={df['predicted_weekly_sales'].min():.2f}, "
                f"max={df['predicted_weekly_sales'].max():.2f}, "
                f"mean={df['predicted_weekly_sales'].mean():.2f}")

    # 7. Передаём в XCom
    result_cols = ['store', 'dept', 'date', 'predicted_weekly_sales']
    context['ti'].xcom_push(
        key='predictions_df',
        value=df[result_cols].to_json(date_format='iso')
    )


def save_predictions_to_postgres(**context):
    """
    Запись результатов предсказаний в таблицу predictions в Postgres.
    """
    # Загружаем датафрейм с предсказаниями из XCom
    predictions_json = context['ti'].xcom_pull(
        task_ids='run_inference', key='predictions_df'
    )

    predictions_df = pd.read_json(predictions_json)

    # Преобразуем date в правильный формат для PostgreSQL
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date

    # Подключение к БД
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    drop_table_query = """
    DROP TABLE IF EXISTS predictions;
    """
    cursor.execute(drop_table_query)
    conn.commit()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        store INT,
        dept INT,
        date DATE,
        predicted_weekly_sales FLOAT,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (store, dept, date)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

    from psycopg2.extras import execute_values

    values = predictions_df[['store', 'dept', 'date', 'predicted_weekly_sales']].to_records(index=False).tolist()

    insert_query = """
    INSERT INTO predictions (store, dept, date, predicted_weekly_sales)
    VALUES %s
    ON CONFLICT (store, dept, date) 
    DO UPDATE SET 
        predicted_weekly_sales = EXCLUDED.predicted_weekly_sales,
        prediction_timestamp = CURRENT_TIMESTAMP;
    """

    execute_values(cursor, insert_query, values)
    conn.commit()

    # Выводим количество строк и первые 5 строк
    cursor.execute("SELECT COUNT(*) FROM predictions;")
    total_rows = cursor.fetchone()[0]
    logger.info(f"Количество строк в таблице predictions: {total_rows}")

    cursor.execute("SELECT store, dept, date, predicted_weekly_sales FROM predictions ORDER BY store, dept, date LIMIT 5;")
    rows = cursor.fetchall()
    logger.info("Первые 5 строк таблицы predictions:")
    for row in rows:
        logger.info(f"  store={row[0]}, dept={row[1]}, date={row[2]}, predicted_weekly_sales={row[3]:.2f}")

    cursor.close()
    conn.close()


# ========== Определение DAG ==========

default_args = {
    'owner': 'Khozhainov',
    'depends_on_past': False,
    'email': ['daniel_steff@mail.ru'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'sales_prediction_batch_inference',
    default_args=default_args,
    description='Batch-инференс прогнозирования продаж для Прилавка',
    schedule_interval='0 20 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sales', 'ml', 'batch-inference', 'production'],
)

task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_from_postgres,
    provide_context=True,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_features',
    python_callable=preprocess_features,
    provide_context=True,
    dag=dag,
)

task_load_model = PythonOperator(
    task_id='load_model',
    python_callable=load_model_from_s3,
    provide_context=True,
    dag=dag,
)

task_inference = PythonOperator(
    task_id='run_inference',
    python_callable=run_batch_inference,
    provide_context=True,
    dag=dag,
)

task_save_predictions = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions_to_postgres,
    provide_context=True,
    dag=dag,
)

task_load_data >> task_preprocess
task_load_model >> task_inference
task_preprocess >> task_inference >> task_save_predictions
