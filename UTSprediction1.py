import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

model = joblib.load('Ranfor_train1.pkl')
target_encoded= joblib.load('target_encoded.pkl')
meal_plane=joblib.load('meal_plan.pkl')
room_type=joblib.load('room_type.pkl')
market_segment=joblib.load('market_segment.pkl')

def main():
    st.title('Hotel Reservation Prediction')

    # Input data dari pengguna
    Booking_ID = st.text_input("Booking_ID")
    no_of_adults = st.selectbox("Adults", options=range(1,11))
    no_of_children = st.selectbox("Children, under 17", options=range(0,11))
    no_of_weekend_nights = st.number_input("Weekend Nights", 0, 8)
    no_of_week_nights = st.number_input("Week Nights (Mon to Fri)", 0, 17)
    type_of_meal_plan = st.selectbox("Type of meal plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3","Not Selected"])
    required_car_parking_space = st.radio("Required car parking space", [0, 1])
    room_type_reserved = st.selectbox("Room type reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4","Room_Type 5","Room_Type 6","Room_Type 7"])
    lead_time = st.number_input("Lead time (in days)", 0, 365)
    arrival_year = st.number_input("Arrival year", 2016, 2020)
    arrival_month = st.selectbox("Arrival month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    arrival_date = st.date_input("Arrival date")
    market_segment_type = st.radio("Market segment type", ["Online", "Corporate", "Complementary", "Offline","Aviation"])
    repeated_guest = st.radio("Repeated guest", [0, 1])
    no_of_previous_cancellations = st.number_input("Number of previous cancellations", 0, 30)
    no_of_previous_bookings_not_canceled = st.number_input("Number of previous bookings not canceled", 0, 30)
    avg_price_per_room = st.number_input("Average price per room (in Euros)", min_value=0.0, format="%.2f")
    no_of_special_requests = st.number_input("Number of special requests", 0, 10)

    # Membuat DataFrame dari input pengguna - menggunakan nama kolom yang konsisten
    data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }

    input_df = pd.DataFrame([data])

    if st.button('Make Prediction'):
        try:
            # Encode categorical variables - using consistent column names
            input_df['type_of_meal_plan'] = meal_plane.transform(input_df['type_of_meal_plan'].astype(str))
            input_df['room_type_reserved'] = room_type.transform(input_df['room_type_reserved'].astype(str))
            input_df['market_segment_type'] = market_segment.transform(input_df['market_segment_type'].astype(str))
            
    
            # Ensure all columns are numeric
            input_df = input_df.astype(float)
            
            # Make prediction
            prediction = model.predict(input_df)
            
            # Decode prediction
            decoded_prediction = target_encoded.inverse_transform(prediction.reshape(-1, 1))
            
            # Show prediction
            st.success(f'The prediction is: {decoded_prediction[0][0]}')
            
            # Show predictio   
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()

