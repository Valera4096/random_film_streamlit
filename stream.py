import streamlit as st 
import numpy as np
import pandas as pd


from PIL import Image
import base64
from io import BytesIO

from sentence_transformers import SentenceTransformer, util
import faiss

df = pd.read_csv('movies.csv')

@st.cache_resource
def load_model():
    return SentenceTransformer("cointegrated/rubert-tiny2")
model = load_model()
        
@st.cache_resource
def load_vector():
    return model.encode(df["description"])
embeddings = load_vector()

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Функция для поиска фильмов по сходству
def find_similar_movies(user_input, top_n=5):
    user_input = model.encode(user_input)
    user_input = user_input.reshape(1,-1)
    distances, indices = index.search(user_input, top_n)
    return list(indices[0])

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


random_digits = np.random.choice(len(df), size=5, replace=False)

# Создать текстовое поле для ввода названия фильма
title = st.text_input('Что хотите посмотреть сегодня?')
if st.button('Подобрать фильм'):
    # Использование функции
    similar_movies = find_similar_movies(title)
    for i in similar_movies:
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

    
