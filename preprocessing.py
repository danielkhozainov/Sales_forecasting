import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


def create_temporal_features(df):
    """Извлечение временных признаков из даты."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df


def create_avg_sales_feature(df):
    """Средние продажи по магазину и отделу до текущей даты."""
    df = df.copy()
    df = df.sort_values(['store', 'dept', 'date'])

    # Кумулятивное среднее по прошлым неделям (исключаем текущую)
    df['prev_sales'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(1)
    df['cumsum_sales'] = df.groupby(['store', 'dept'])['prev_sales'].cumsum()
    df['count_sales'] = df.groupby(['store', 'dept'])['prev_sales'].cumcount()
    df['avg_sales_before'] = np.where(
        df['count_sales'] > 0,
        df['cumsum_sales'] / df['count_sales'],
        0
    )

    df.drop(columns=['prev_sales', 'cumsum_sales', 'count_sales'], inplace=True)
    return df


def create_lag_features(df):
    """Создание лаговых признаков продаж."""
    df = df.copy()
    df = df.sort_values(['store', 'dept', 'date'])

    for lag in [1, 2, 4]:
        df[f'sales_{lag}week_ago'] = df.groupby(['store', 'dept'])['weekly_sales'].shift(lag)

    return df


def create_rolling_features(df):
    """Создание скользящих статистик."""
    df = df.copy()
    df = df.sort_values(['store', 'dept', 'date'])

    # Скользящее среднее по прошлым неделям (исключаем текущую)
    shifted = df.groupby(['store', 'dept'])['weekly_sales'].shift(1)
    df['mean_sales_2week'] = shifted.groupby(
        [df['store'], df['dept']]
    ).rolling(window=2, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    shifted4 = df.groupby(['store', 'dept'])['weekly_sales'].shift(1)
    df['mean_sales_4week'] = shifted4.groupby(
        [df['store'], df['dept']]
    ).rolling(window=4, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    return df


def preprocess_data(df):
    """
    Полная предобработка данных (идентична логике обучения).

    1. Обрабатывает аномальные продажи.
    2. Заполняет пропуски средним значением.
    3. Обогащает датасет признаками.
    """
    df = df.copy()

    # 1. Обработка аномальных продаж
    # Продажи не могут быть отрицательными
    if 'weekly_sales' in df.columns:
        df.loc[df['weekly_sales'] < 0, 'weekly_sales'] = 0

    # 2. Заполнение пропусков средним
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # Заполняем пропуски в категориальных признаках
    cat_cols = ['store', 'dept', 'type']
    for col in cat_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)

    # 3. Обогащение признаками
    df = create_temporal_features(df)
    df = create_avg_sales_feature(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    return df
