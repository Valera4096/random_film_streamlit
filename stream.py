import streamlit as st 
import numpy as np
import pandas as pd

from PIL import Image
import base64
from io import BytesIO

from sentence_transformers import SentenceTransformer, util
import faiss

st.title('Умный поиск фильмов')

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
        ['Боевик','Приключения','Фэнтези','Драма','Фантастика','Криминал','Комедия',
        'Триллер','Семейные','Детектив','Ужасы','История','Документальный',
        'Мюзикл','Биография','Военный','Спорт','Вестерн','Сериалы','Мультфильмы'])
    # Добавить selectbox для выбора конкретного года
    target_set = set(options)

# фильтруем DataFrame
    if len(options) != 0:
        df = df[df['genres'].apply(lambda x: target_set.issubset(set(x.replace(' ','').split(','))))].reset_index()
        embeddings = model.encode(df["description"])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)    
    
    years = list(range(1964, 2024))  # Список всех возможных лет
    year = sorted(st.multiselect('Выберите год:', years, max_selections= 2))
    
    if len(year) == 2:
        df = df[(df['year'] >= year[0]) & (df['year'] <= year[1]) ].reset_index()
        if len(df) == 0:
            st.title('По заданым параметрам ничего не найдено ((')
        else:
            embeddings = model.encode(df["description"])
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
    elif len(year) ==1:
        df = df[df['year'] == year[0]].reset_index()
        
        if len(df) == 0:
            st.title('По заданым параметрам ничего не найдено ((') 
        else:  
            embeddings = model.encode(df["description"])
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)


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
                st.title('Название фильма:')
                st.header(df['movie_title'][i])
                st.subheader('Год: ' + str(df['year'][i]))
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
        size = 5
        if len(df) < size:
            size = len(df)
        random_digits = np.random.choice(len(df), size=size, replace=False)
        if size == 0:
            st.header('Невозможно сделать случайный выбор, т.к по таким параметрам фильмы не найдены')
        for i in random_digits:
            st.title('Название фильма:')
            st.header(df['movie_title'][i])
            st.subheader('Год: ' + str(df['year'][i]))
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
   
