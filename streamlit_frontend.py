import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime

BOOKING_FILE = "bookings.csv"

model = joblib.load("ticket_model.pkl")

metadata_file = "ticket_model_metadata.json"
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
else:
    metadata = None

with open("model_features.json", "r") as f:
    features = json.load(f)


def predict_demand(day, month, quarter, day_of_week, show_time, ticket_price, film_code, cinema_code, capacity):
    input_dict = {
        "day": day,
        "month": month,
        "quarter": quarter,
        "day_of_week": day_of_week,
        "show_time": show_time,
        "ticket_price": ticket_price,
        "capacity": capacity,
        "is_weekend": int(day_of_week in [6, 7]),
        "price_per_seat": ticket_price / capacity if capacity > 0 else 0,
    }
    input_df = pd.DataFrame([input_dict])
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    film_col = f"film_code_{str(film_code)}"
    cinema_col = f"cinema_code_{str(cinema_code)}"
    if film_col in input_df.columns:
        input_df[film_col] = 1
    if cinema_col in input_df.columns:
        input_df[cinema_col] = 1
    input_df = input_df[features]
    prediction = model.predict(input_df)[0]
    return max(0, int(prediction))


st.set_page_config(page_title="🎬 Movie Ticket Demand & Booking", layout="wide")


def login():
    st.title("🎬 Movie Ticket Booking Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Please enter both username and password.")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


def book_tickets():
    st.header("🎟 Book Your Tickets")
    shows = [
        {"film": "Inception", "cinema": "Cineplex A", "time": "6:00 PM", "price": 250, "available": 45},
        {"film": "Avatar 2", "cinema": "Cineplex B", "time": "9:00 PM", "price": 300, "available": 60},
        {"film": "Oppenheimer", "cinema": "Cineplex A", "time": "3:00 PM", "price": 280, "available": 25},
    ]
    cols = st.columns(3)
    for idx, show in enumerate(shows):
        with cols[idx % 3]:
            bg_color = "#d4edda" if show["available"] > 40 else "#fff3cd" if show["available"] > 20 else "#f8d7da"
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <h4>{show['film']}</h4>
                    <p>🎥 Cinema: {show['cinema']}</p>
                    <p>🕒 Time: {show['time']}</p>
                    <p>💰 Price: ₹{show['price']}</p>
                    <p>🎟 Available: {show['available']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"Book {show['film']}"):
                new_booking = {
                    "user": st.session_state.username,
                    "film": show["film"],
                    "cinema": show["cinema"],
                    "time": show["time"],
                    "price": show["price"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                if os.path.exists(BOOKING_FILE):
                    df = pd.read_csv(BOOKING_FILE)
                    df = pd.concat([df, pd.DataFrame([new_booking])], ignore_index=True)
                else:
                    df = pd.DataFrame([new_booking])
                df.to_csv(BOOKING_FILE, index=False)
                st.success(f"✅ Ticket booked for {show['film']} at {show['time']}!")


def view_bookings():
    st.header("🧾 My Bookings")
    if not os.path.exists(BOOKING_FILE):
        st.info("No bookings found.")
    else:
        bookings = pd.read_csv(BOOKING_FILE)
        user_bookings = bookings[bookings["user"] == st.session_state.username]
        if user_bookings.empty:
            st.info("No bookings found.")
        else:
            user_bookings = user_bookings.sort_values("timestamp", ascending=False)
            for _, row in user_bookings.iterrows():
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f9f9f9;
                        padding: 15px;
                        border-radius: 12px;
                        margin-bottom: 15px;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
                        border-left: 6px solid #4CAF50;
                    ">
                        <h4 style="margin: 0;">🎬 {row['film']}</h4>
                        <p style="margin: 5px 0;"><b>🎭 Cinema:</b> {row['cinema']}</p>
                        <p style="margin: 5px 0;"><b>🕒 Show Time:</b> {row['time']}</p>
                        <p style="margin: 5px 0;"><b>💵 Price:</b> ₹{row['price']}</p>
                        <p style="margin: 5px 0; color: gray; font-size: 0.85em;">
                            📌 Booked on: {row['timestamp']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


if not st.session_state.logged_in:
    login()
else:
    menu = ["Home", "Book Tickets", "View My Bookings", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        if metadata:
            st.info(f"📊 Model v{metadata['version']} | R²: {metadata['r2']:.4f} | Trained on {metadata['timestamp']}")
        st.header("🎯 Predict Ticket Demand")
        col1, col2, col3 = st.columns(3)
        with col1:
            day = st.number_input("Day", 1, 31, 1)
            month = st.number_input("Month", 1, 12, 1)
            quarter = st.selectbox("Quarter", [1, 2, 3, 4])
        with col2:
            day_of_week = st.selectbox("Day of Week (1=Mon, 7=Sun)", list(range(1, 8)), index=0)
            show_time = st.selectbox("Show Time", {1: "Morning", 2: "Afternoon", 3: "Evening", 4: "Late Night"})
        with col3:
            ticket_price = st.number_input("Ticket Price", min_value=50, max_value=1000, value=200)
            capacity = st.number_input("Capacity", min_value=10, max_value=500, value=100)
        film_code = st.text_input("Film Code (e.g., F101)")
        cinema_code = st.text_input("Cinema Code (e.g., C05)")
        if st.button("🔮 Predict Demand"):
            if not film_code or not cinema_code:
                st.error("Please enter Film Code and Cinema Code.")
            else:
                demand = predict_demand(
                    day, month, quarter, day_of_week,
                    show_time, ticket_price, film_code, cinema_code, capacity
                )
                st.success(f"📈 Predicted Demand: **{demand} tickets**")
    elif choice == "Book Tickets":
        book_tickets()
    elif choice == "View My Bookings":
        view_bookings()
    elif choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("You have been logged out.")
