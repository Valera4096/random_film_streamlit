import streamlit as st 
import numpy as np
import pandas as pd

from PIL import Image
import base64
from io import BytesIO

from sentence_transformers import SentenceTransformer
import faiss

import pickle

st.markdown(f'<p style="background-color: #262730; color: white ; font-size: 40px; font-weight: bold; text-align:">Умный поиск фильмов</p>', unsafe_allow_html=True)

df = pd.read_csv("resources/movies.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer("cointegrated/rubert-tiny2")
model = load_model()
        
@st.cache_resource
def load_embeddings():
    return np.load('resources/embeding.npy')
embeddings = load_embeddings()

@st.cache_resource
def load_lst():    
    with open('resources/lst_actor.pkl', 'rb') as f:
        lst_actor_loaded = pickle.load(f)
    return lst_actor_loaded

lst_actor_loaded = load_lst()


index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


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
    actor_movie = df['actors'][i]
    imb = 'Нет оценки' if df['imdb'][i] == 0 else str(df['imdb'][i])
    end = '_'*37
    write_movie = f'''
    <div style="background-color: white; color: black; font-size: 35px; padding: 15px; margin-bottom: 0px; display: flex;">
        <div style="margin-right: 20px;">
            <img src="{imges}" width="200" height="350">
        </div>
        <div style="text-align: left;">
            <p style="font-size: 25px; font-weight: bold;">Название фильма:</p>
            <p style="font-size: 30px; font-weight: bold;">{movie_titl}</p>
            <p style="font-size: 20px; font-weight: bold;">Год: {year_movie}</p>
            <p style="font-size: 20px; font-weight: bold;">Описание:</p>
            <p style="font-size: 15px; font-weight: bold;">{descrip_movie}</p>
            <p style="font-size: 20px; font-weight: bold;">Жанр: {genre_movie}</p>
            <p style="font-size: 15px; font-weight: bold;">Актерский состав: {actor_movie}</p>
            <p style="font-size: 20px; font-weight: bold;">Оценка: {imb}</p>
            <p style="font-size: 20px; font-weight: bold;">{end}</p>
        </div>
    </div>
    '''
    st.markdown(write_movie, unsafe_allow_html=True)


# Загрузить изображение из файла
img = Image.open('resources/_-fotor.jpg')
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
if st.sidebar.toggle('Фильтр'):
    
    st.sidebar.header('Выбор дат')        
    year_min = st.sidebar.slider("Выбор минимального года:", min_value=1937, max_value=2022, value=2015)
    year_max = st.sidebar.slider("Выбор максимального года:", min_value=year_min, max_value=2024, value=2020)
    
    options = st.sidebar.multiselect(
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
    

        
    actors = st.sidebar.multiselect('Актерский состав',lst_actor_loaded)

    actors_set = set(actors)

    if st.sidebar.toggle('Применить фильтры'):
        df = df[(df['year'] >= year_min) & (df['year'] <= year_max)].reset_index(drop= True)
        if len(df) == 0:
            st.title('По заданым параметрам ничего не найдено ((')
        else:
            filterd_embeddings = embeddings[np.array(df.index)]  
            index = faiss.IndexFlatIP(filterd_embeddings.shape[1])
            index.add(filterd_embeddings)    
            
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
        
        if len(actors_set) != 0:
            df = df[df['actors'].apply(lambda x: actors_set.issubset(x.split(', ')))].reset_index()
            if len(df) == 0:
                st.title('По заданым параметрам ничего не найдено ((')
            else:
                filterd_embeddings = embeddings[np.array(df.index)]  
                index = faiss.IndexFlatIP(filterd_embeddings.shape[1])
                index.add(filterd_embeddings)        
        
        
        


if len(df) !=0 :
    # Создать текстовое поле для ввода названия фильма
    title = st.text_input('Что хотите посмотреть сегодня?')
    Top_K = st.sidebar.slider('Количество рекомендаций?', 1, 10, 3 )
    
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
            st.markdown(f'<p style="text-align: center;"></p>', unsafe_allow_html=True)
