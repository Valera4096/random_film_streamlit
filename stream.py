import streamlit as st 

import pandas as pd

df = pd.read_csv('movies.csv')


st.image('http://kinogo.uk//uploads/mini/full/8a/afd93f08b408c6faf115c0a09732f2.jpg')

st.image('https://www.ixbt.com/img/n1/news/2022/2/6/Instagram-Direct-Message_large.jpg')

title = st.text_input('Что хотите посмотреть сегодня? ')
st.write(title)

if st.button('Say hello'):
    st.write('Why hello there')
    st.image(df['img_url'][0])