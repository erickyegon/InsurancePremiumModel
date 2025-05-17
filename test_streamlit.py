import streamlit as st

st.title("Test Streamlit App")
st.write("This is a test app to check if Streamlit is working correctly.")

# Add a simple input
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")

# Add a simple button
if st.button("Click me"):
    st.write("Button clicked!")
