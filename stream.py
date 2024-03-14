import streamlit as st 
import numpy as np
import pandas as pd

from PIL import Image
import base64
from io import BytesIO

from sentence_transformers import SentenceTransformer, util
import faiss

import ast

st.markdown(f'<p style="background-color: white; color: black; font-size: 40px; font-weight: bold; text-align:">Умный поиск фильмов</p>', unsafe_allow_html=True)

df = pd.read_csv("movies.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer("cointegrated/rubert-tiny2")
model = load_model()
        
@st.cache_resource
def load_embeddings():
    return np.load('embeding.npy')
embeddings = load_embeddings()


index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings))


# Функция для поиска фильмов по сходству
def find_similar_movies(user_input, top_n=5):
    user_input = model.encode(user_input)
    user_input = user_input.reshape(1,-1)
    distances, indices = index.search(user_input, top_n)
    return list(indices[0])


def display_movie(i):    
    movie_titl = df['movie_title'][i]
    year_movie = str(df['year'][i])
    descrip_movie = df['description'][i]
    imges = df['img_url'][i]
    genre_movie = "Нет данных" if pd.isna(df['genres'].iloc[i]) else df['genres'][i]
    imb = 'Нет оценки' if df['imdb'][i] == 0 else str(df['imdb'][i])
    end = '_'*28
    write_movie  = f'''<div style="background-color:white; color: black; font-size: 35px; padding: 15px; margin-bottom: 0px;>
        <p style= "font-size: 25px; font-weight: bold; text-align:">Название фильма:</p>
        <p style= "font-size: 35px; font-weight: bold; text-align:">{movie_titl}</p>
        <img src="{imges}"width="250" height="400">
        <p style="font-size: 20px; font-weight: bold; text-align:">Год: {year_movie}</p>
        <p style="font-size: 20px; font-weight: bold; text-align:">Описание: </p>
        <p style="font-size: 15px; font-weight: bold; text-align:">{descrip_movie}</p>
        <p style="font-size: 20px; font-weight: bold; text-align:">Жанр: {genre_movie}</p>
        <p style="font-size: 20px; font-weight: bold; text-align:">Оценка:</p>
    </div>'''
    st.markdown(write_movie, unsafe_allow_html=True)


# Загрузить изображение из файла
img = Image.open('_-fotor.jpg')
buffered = BytesIO()
img.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Добавить фон с изображением
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{img_str});
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)


#Фильтры
if st.toggle('Фильтр'):
    options = st.multiselect(
        'Выберите жанры',
        ['триллер','боевик','драма','комедия','мелодрама','комедия','детектив','криминал','боевик',
         'фантастика','фэнтези','детский','семейный','фантастика', 
            'короткометражный','детектив','мелодрама','драма','триллер','мультфильмы',
            'приключения','короткометражный,','мультфильмы','приключения',
            'семейный','документальный','ужасы','биография','криминал,','документальный','исторический',
            'аниме','вестерн','ужасы','спорт',
            'музыкальный','военный','мюзикл','исторический','биография','вестерн','мюзикл',
            'военный','музыкальный','детский','эротика','фэнтези','аниме'])
    target_set = set(options)

# фильтруем DataFrame
    if len(options) != 0:
            
        df = df[df['genres'].apply(lambda x: target_set.issubset(x.split(', ')))].reset_index(drop= True)
        if len(df) == 0:
            st.title('По заданым параметрам ничего не найдено ((')
        else:
            filterd_embeddings = embeddings[np.array(df.index)]  
            # создаем индекс Faiss
            index = faiss.IndexFlatIP(filterd_embeddings.shape[1])
            # добавляем вектора вложений в индекс Faiss
            index.add(filterd_embeddings)
            
            
            
    
    years = list(range(1937, 2024))  # Список всех возможных лет
    year = sorted(st.multiselect('Выберите год:', years, max_selections= 2))
    
    if len(year) == 2:
        df = df[(df['year'] >= year[0]) & (df['year'] <= year[1])].reset_index(drop= True)
        if len(df) == 0:
            st.title('По заданым параметрам ничего не найдено ((')
        else:
            filterd_embeddings = embeddings[np.array(df.index)]  
            # создаем индекс Faiss
            index = faiss.IndexFlatIP(filterd_embeddings.shape[1])
            # добавляем вектора вложений в индекс Faiss
            index.add(filterd_embeddings)
    elif len(year) ==1:
        df = df[df['year'] >= year[0]].reset_index(drop= True)
        
        if len(df) == 0:
            st.title('По заданым параметрам ничего не найдено ((') 
        else:  
            filterd_embeddings = embeddings[np.array(df.index)]  
            # создаем индекс Faiss
            index = faiss.IndexFlatIP(filterd_embeddings.shape[1])
            # добавляем вектора вложений в индекс Faiss
            index.add(filterd_embeddings)



if len(df) !=0 :
    # Создать текстовое поле для ввода названия фильма
    title = st.text_input('Что хотите посмотреть сегодня?')
    Top_K = st.slider('Количество рекомендаций?', 1, 10, 3)
    
    if st.button('Подобрать фильм'):
        # Использование функции
        similar_movies = find_similar_movies(title,Top_K)
        for i in similar_movies:
            if i == -1:
                st.title('Больше рекомендаций нет, уменьшите количество рекомендаций или измените фильтры')
                break
            else:
                display_movie(i)

    if st.button('Выбрать случайно'):
        size = 5
        if len(df) < size:
            size = len(df)
        random_digits = np.random.choice(len(df), size=size, replace=False)
        if size == 0:
            st.header('Невозможно сделать случайный выбор, т.к по таким параметрам фильмы не найдены')
        for i in random_digits:
            display_movie(i)
            
            
