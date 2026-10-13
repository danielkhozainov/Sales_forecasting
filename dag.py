from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
import json

default_args = {
    'owner': 'test',
    'depends_on_past': False,
    'retries': 0,
}

dag = DAG(
    'check_schema',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['test'],
)

def check_all_schemas(**context):
    pg_hook = PostgresHook(postgres_conn_id="postgres_sales_db")
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    result = {}
    for table_name in ['sales', 'plan', 'stores', 'features']:
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table_name,))
        rows = cursor.fetchall()
        result[table_name] = [[row[0], row[1]] for row in rows]

    cursor.close()
    conn.close()

    # Передаём в XCom — можно посмотреть в UI
    context['ti'].xcom_push(key='schema', value=json.dumps(result))
    return json.dumps(result)

check_task = PythonOperator(
    task_id='check_schema',
    python_callable=check_all_schemas,
    provide_context=True,
    dag=dag,
)

