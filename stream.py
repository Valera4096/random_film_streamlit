import streamlit as st 
import numpy as np
import pandas as pd


from PIL import Image
import base64
from io import BytesIO

# Загрузить изображение из файла
img = Image.open('_-fotor.jpg')

buffered = BytesIO()
img.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()


st.markdown(f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{img_str});
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)



df = pd.read_csv('movies.csv')
random_digits = np.random.choice(len(df), size=5, replace=False)



# Создать текстовое поле для ввода названия фильма
title = st.text_input('Что хотите посмотреть сегодня?')
if st.button('Подобрать фильм'):
    st.header('Я пока что в разработке 😬😬😬😬')

if st.button('Выбрать случайно'):
    for i in random_digits:
        st.title('Название фильма:')
        st.header(df['movie_title'][i])
        st.image('http://'+ df['img_url'][i])
        st.title('Описание:')
        st.write(df['description'][i])
        st.title('Жанр:')
        st.write("Нет данных" if pd.isna(df['genres'].iloc[i]) else df['genres'][i] ) 
        st.title('Оценка')
        imb = 'Нет оценки' if df['imdb'][i] == 0 else str(df['imdb'][i])
        kinopoisk = 'Нет оценки' if df['kinopoisk'][i] == 0 else str(df['kinopoisk'][i])
        st.write(f'Рейтинг imdb: {imb}')
        st.write(f'Рейтинг кинопоиск: {kinopoisk}')
        st.title('-'*45)

    
