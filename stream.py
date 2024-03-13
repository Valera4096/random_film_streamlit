import streamlit as st 
import numpy as np
import pandas as pd

df = pd.read_csv('movies.csv')
random_digits = np.random.choice(len(df), size=5, replace=False)



# Создать текстовое поле для ввода названия фильма
title = st.text_input('Что хотите посмотреть сегодня?')

if len(title) != 0:
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

    
