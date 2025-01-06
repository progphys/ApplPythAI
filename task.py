import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import aiohttp
import asyncio
from datetime import date
from functools import partial
from multiprocessing import Pool
import plotly.graph_objects as go


#####################
# Асинхронная логика
#####################

async def get_current_temperature_async(town: str, api_key: str):
    """
    Асинхронный запрос к OpenWeatherMap, возвращает (температура, ошибка).
    """
    api_url = f"http://api.openweathermap.org/data/2.5/weather?q={town}&units=metric&appid={api_key}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(api_url) as response:
                if response.status != 200:
                    # Например, 404
                    error_message = await response.json()
                    return None, error_message.get("message", "Unknown error")
                data = await response.json()
                current_temperature = data['main']['temp']
                return current_temperature, None
        except aiohttp.ClientError as e:
            # Ошибки вида таймаута, DNS и т.д.
            return None, str(e)

def get_current_temperature_sync(town: str, api_key: str):
    """
    Синхронная обёртка над асинхронной функцией.
    """
    return asyncio.run(get_current_temperature_async(town, api_key))


#####################
# Аналитика
#####################

def TStown(data_original: pd.DataFrame, town: str):
    """
    Анализ временного ряда для конкретного города:
    - расчет скользящего среднего и std
    - определение аномалий
    - сезонный профиль
    - линейная регрессия (trend_line)
    Возвращает словарь с нужными данными.
    """
    # Фильтруем данные по городу
    data = data_original[data_original['city'] == town].copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp')

    # Вычисляем скользящее среднее и стандартное отклонение
    rolling_window = 30
    data['rolling_mean'] = data['temperature'].rolling(window=rolling_window).mean()
    data['rolling_std'] = data['temperature'].rolling(window=rolling_window).std()

    # Определяем аномалии
    data['Is_Anomaly'] = abs(data['temperature'] - data['rolling_mean']) > 2 * data['rolling_std']
    anomalies = data.loc[data['Is_Anomaly']].reset_index(drop=True)

    # Профиль сезонов
    season_profile = data.groupby('season').agg(
        mean_temperature=('temperature', 'mean'),
        std_temperature=('temperature', 'std'),
        mean_rolling_temperature=('rolling_mean', 'mean'),
        std_rolling_temperature=('rolling_mean', 'std')
    ).reset_index()

    # Линейная регрессия для тренда
    x = (data['timestamp'] - data['timestamp'].min()).dt.days.astype(float)
    y = data['temperature'].astype(float)
    coeffs = np.polyfit(x, y, 1)  # [slope, intercept]
    slope, intercept = coeffs[0], coeffs[1]

    data['trend_line'] = np.polyval(coeffs, x)

    statistics = {
        'Город': town,
        'Минимальная температура': data['temperature'].min(),
        'Максимальная температура': data['temperature'].max(),
        'Средняя температура': data['temperature'].mean(),
        'Наклон тренда': slope
    }

    # Возвращаем также сырые данные для построения графика
    return {
        'Статистика': statistics, 
        'Профиль сезона': season_profile, 
        'Аномальные точки': anomalies,
        'Данные': data.reset_index(drop=True),  # Для графика
        'TrendCoeffs': (slope, intercept)
    }

def parallel_task(data, town):
    """
    Обёртка для multiprocessing.Pool.
    """
    return TStown(data, town)

def get_season():
    current_date = date.today()
    month = current_date.month
    day = current_date.day
    if (month == 12) or (1 <= month <= 2):
        return 'winter'
    elif (3 <= month <= 5):
        return 'spring'
    elif (6 <= month <= 8):
        return 'summer'
    else:
        return 'autumn'


#####################
# Streamlit-приложение
#####################

st.title("Анализ исторических данных и текущей погоды")
st.sidebar.header("Загрузка данных")
file = st.sidebar.file_uploader("Загрузите CSV-файл с историческими данными", type="csv")

if file is not None:
    data_original = pd.read_csv(file)
    
    # Список городов
    cities = data_original['city'].unique()

    # Параллельный анализ данных по всем городам
    task_with_data = partial(parallel_task, data_original)
    with Pool() as pool:
        results_for_all_cities = pool.map(task_with_data, cities)

    # Выбор города
    selected_city = st.sidebar.selectbox("Выберите город", cities)
    idx_city = list(cities).index(selected_city)
    results = results_for_all_cities[idx_city]

    # Блок настроек API
    st.sidebar.header("Настройки API")
    api_key = st.sidebar.text_input("Введите API-ключ OpenWeatherMap", type="password")

    ###################
    # Текущая погода
    ###################
    current_temp = None
    if api_key:
        st.header(f"Текущая погода для города {selected_city}")
        
        # Асинхронный вызов через синхронную обёртку
        current_temp, error = get_current_temperature_sync(selected_city, api_key)
        
        if current_temp is not None:
            st.write(f"Текущая температура: {current_temp} °C")
        elif error:
            st.error(f"Ошибка: {error}")

        # Проверка на аномальность: сравниваем с текущим сезоном
        season_now = get_season()
        profile_now = results['Профиль сезона'][results['Профиль сезона']['season'] == season_now]
        if not profile_now.empty and current_temp is not None:
            mean_temp = float(profile_now['mean_temperature'].iloc[0])
            std_temp = float(profile_now['std_temperature'].iloc[0])
            if abs(current_temp - mean_temp) > 2 * std_temp:
                st.warning("Температура сейчас аномальна для этого сезона!")
            else:
                st.info("Температура в рамках нормы для сезона.")

    ###################
    # Описательная статистика
    ###################
    st.subheader("Описательная статистика")
    st.write("Город:", results['Статистика']['Город'])
    st.write("Минимальная температура:", results['Статистика']['Минимальная температура'])
    st.write("Максимальная температура:", results['Статистика']['Максимальная температура'])
    st.write("Средняя температура:", results['Статистика']['Средняя температура'])
    st.write("Наклон тренда:", results['Статистика']['Наклон тренда'])

    ###################
    # Профиль сезона
    ###################
    st.subheader("Профиль сезона")
    st.write(results['Профиль сезона'])

    ###################
    # Построение графика
    ###################
    df = results['Данные']  # Исходный временной ряд с вычисленными полями
    anomalies = results['Аномальные точки']
    
    # Создаем фигуру Plotly
    fig = go.Figure()

    # 1. Исходные данные (температура)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        mode='lines',
        name='Температура'
    ))

    # 2. Скользящее среднее
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['rolling_mean'],
        mode='lines',
        name='Скользящее среднее'
    ))

    # 3. Тренд (линейная регрессия)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['trend_line'],
        mode='lines',
        name='Линия тренда'
    ))

    # 4. Аномальные точки
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['temperature'],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Аномалии'
        ))

    # 5. Горизонтальная линия текущей температуры (если получили current_temp)
    if current_temp is not None:
        fig.add_hline(
            y=current_temp,
            line_dash='dash',
            annotation_text=f"Текущая {current_temp} °C",
            annotation_position="top right"
        )

    # Настройки лейаута
    fig.update_layout(
        title=f"Температура во времени ({selected_city})",
        xaxis_title="Дата",
        yaxis_title="Температура (°C)",
        legend_title="Легенда",
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Пожалуйста, загрузите CSV-файл с историческими данными.")
