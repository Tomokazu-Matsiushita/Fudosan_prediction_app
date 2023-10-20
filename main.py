#必要なLibraryをimport
import requests, os, time, json
import pandas as pd
from geopy.geocoders import GoogleV3
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import japanize_matplotlib
import streamlit as st
import joblib
import numpy as np

# ページレイアウト
st.set_page_config(layout="wide")

# 訓練データの読み込み
df = pd.read_csv('物件情報_2.csv')

# Streamlitアプリの作成
st.title("HONNE!はいくら!")
st.markdown('#### 物件価格アプリ')

def load_model():
    # Define a function to load your trained model
    model = joblib.load("fudosan_model_file.pkl")  # Replace with your model file
    return model

def predict_price(Space, Latitude, Longitude, Walk, DaysAgo):
#def predict_price(Space, Latitude, Longitude, Walk):
#def predict_price(Space, Walk, DaysAgo):
#def predict_price(Space):
    model = load_model()
    # Prepare input data in the same format as used for training
    input_data = [[np.log1p(Space), Latitude, Longitude, np.log1p(Walk), np.log1p(DaysAgo)]]
    #input_data = [[Space, Latitude, Longitude, Walk]]
    #input_data = [[Space, Walk, DaysAgo]]
    #input_data = [[np.log1p(Space)]]
    log_predicted_price = model.predict(input_data)[0]  # Predicted price in log scale
    predicted_price = np.expm1(log_predicted_price)  # Inverse transformation to get the original scale
    return predicted_price


def main():
        # Define default values for latitude and longitude
    latitude_value = None
    longitude_value = None

    location_df = pd.DataFrame(columns=['Latitude', 'Longitude'])

    # Input fields
    # 都道府県の選択
    prefecture = st.sidebar.selectbox('都道府県を選択してください:', df['City'].unique())
    filtered_df = df[df['City'] == prefecture]  # Filter DataFrame by selected 'City'

    city = st.sidebar.selectbox('区を選択してください:', filtered_df['Street'].unique())
    # 区の選択
    filtered_df = df[df['Street'] == city]
    # 地名の選択
    street = st.sidebar.text_input('町名を入力してください:')
    number = st.sidebar.text_input('番地を入力してください:')
    #filtered_df = df[df['Number'] == number]
    type = st.sidebar.selectbox('部屋のタイプを選択してください：', filtered_df['Type'].unique())
    filtered_df = df[df['Type'] == type]

    Space = st.sidebar.number_input('広さを入力してください(㎡):')

    Walk = st.sidebar.number_input('駅からの徒歩での時間を入力してください:')

    DaysAgo = st.sidebar.number_input('築年数を日数で入力してください:')

    from geopy.geocoders import GoogleV3

    # Create a geocoder object
    geolocator = GoogleV3(api_key='AIzaSyC6_bF8DhyaTIqnX4qTW5Z5Z-e4gG1af3k') 
    # Get latitude and longitude from '町名' and '番地'
    def get_lat_long(street, number):
        address = f"{street}, {number}"  # Combine '町名' and '番地' into a full address
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None

    if street and number:
        Latitude, Longitude = get_lat_long(street, number)
        if Latitude and Longitude:
            st.write(f"Latitude: {Latitude}, Longitude: {Longitude}")
        else:
            st.write("Location not found.")
    else:
        Latitude = None  # Define a default value if condition is not met
        Longitude = None

        # Button to trigger prediction
    if st.button("予想価格をチェック"):
        prediction = predict_price(Space, Latitude, Longitude, Walk, DaysAgo)
        #prediction = predict_price(Space, Latitude, Longitude, Walk)
        #prediction = predict_price(Space, Walk, DaysAgo)
        #prediction = predict_price(Space)
        st.write(f"**予想価格は: {prediction:.2f}万円**")

    st.write('都道府県:', prefecture)
    st.write('区:', city)
    st.write('町名',street)
    st.write('番地', number)
    st.write('部屋のタイプ:', type)
    st.write('部屋の広さ:', Space)
    st.write('駅からの徒歩時間:', Walk)
    st.write('築年数:', DaysAgo)


    # Create a boxplot based on the selected data
    # min_value = min('Space')
    # max_value = max('Space')
    # num_ticks = 10
    # specific_ticks = np.linspace(min_value, max_value, num_ticks)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Space', y='Price', data=filtered_df)
    plt.xlabel('Space (㎡)')
    plt.xticks(rotation=90)
    plt.ylabel('Price(万円)')
    plt.title(f'Price Distribution by Space for {prefecture}, {city}, Type: {type}')
    st.pyplot(plt)

    # min_value_2 = min('DaysAgo')
    # max_value_2 = max('DaysAgo')
    # num_ticks_2 = 100
    # specific_ticks_2 = np.linspace(min_value_2, max_value_2, num_ticks_2)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='DaysAgo', y='Price', data=filtered_df)
    plt.xlabel('DaysAgo (days)')
    plt.xticks(rotation=90)
    plt.ylabel('Price(万円)')
    plt.title(f'Price Distribution by Space for {prefecture}, {city}, Type: {type}')
    st.pyplot(plt)

    # Assuming you have a DataFrame named 'df' and you want to select the 'Latitude' and 'Longitude' columns
    # Create a new DataFrame containing only those columns
    # Rename 'Latitude' and 'Longitude' columns to 'latitude' and 'longitude'
    location_data = pd.DataFrame({'latitude': [Latitude], 'longitude': [Longitude]})

    st.map(location_data, use_container_width=True)


if __name__ == "__main__":
    main()

st.balloons()

