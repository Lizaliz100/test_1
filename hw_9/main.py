import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Расчёт стоимости недвижимости",
    layout="centered"
)

MODEL_PATH = 'realty_model.pkl'

st.title("🏡💲 Прогноз стоимости недвижимости")
st.write("Введите параметры объекта недвижимости в боковой панели для расчёта стоимости.")

st.sidebar.header("Параметры объекта:")

total_square = st.sidebar.number_input("Общая площадь (кв. м)", min_value=5.0, max_value=1000.0, value=50.0, step=0.5)
rooms = st.sidebar.number_input("Количество комнат", min_value=1, max_value=10, value=2, step=1)
floor = st.sidebar.number_input("Этаж", min_value=1, max_value=100, value=5, step=1)

lat = st.sidebar.number_input("Широта (Latitude)", min_value=40.0, max_value=80.0, value=55.75, step=0.01)
lon = st.sidebar.number_input("Долгота (Longitude)", min_value=19.0, max_value=180.0, value=37.62, step=0.01)

object_type = st.sidebar.selectbox(
    "Тип объекта",
    ("Вторичка", "Новостройка", "Другое")
)

city = st.sidebar.selectbox(
    "Город",
    ("Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Другой город")
)


input_df = pd.DataFrame(
    {
        "total_square": [float(total_square)],
        "rooms": [float(rooms)],
        "floor": [float(floor)],
        "lat": [float(lat)],
        "lon": [float(lon)],
        "object_type": [object_type],
        "city": [city]
    }
)

if not os.path.exists(MODEL_PATH):
    st.error(f"Файл модели '{MODEL_PATH}' не найден! Убедитесь, что вы запустили ячейку сохранения в блокноте.")
else:
    model = joblib.load(MODEL_PATH)

    if st.sidebar.button("Рассчитать стоимость"):
        prediction = model.predict(input_df)[0]

        st.success("### Расчет успешно завершен!")

        if prediction > 0:
            st.metric(
                label="Ориентировочная стоимость объекта",
                value=f"{prediction:,.2f} ₽".replace(",", " ")
            )
        else:
            st.warning(
                "📌 Внимание: Модель вернула отрицательную стоимость. Проверьте корректность вводимых координат и параметров.")
            st.metric(label="Результат модели", value=f"{prediction:,.2f} ₽".replace(",", " "))

        st.write("**Выбранные параметры:**")
        st.json(input_df.iloc[0].to_dict())
