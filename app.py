import os
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Function to load the data
def load_data(uploaded_file):
    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
        st.success("Data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to explore the dataset
def explore_data(df):
    try:
        st.subheader("Dataset Information:")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")
        st.write(f"**Numeric Columns:** {df.select_dtypes(include='number').shape[1]}")
        st.write(f"**Categorical Columns:** {df.select_dtypes(include='object').shape[1]}")
        st.write(f"**Missing Values:** {df.isna().sum().sum()}")
    except Exception as e:
        st.error(f"Error during data exploration: {e}")

# Function to perform univariate analysis
def univariate_analysis(df):
    try:
        st.subheader("Univariate Analysis:")
        st.write(df.describe())
    except Exception as e:
        st.error(f"Error during univariate analysis: {e}")

# Function to handle custom queries using OpenAI
def custom_query(df, llm):
    try:
        custom_query = st.text_input("Enter your query", placeholder="Ask a custom query", label_visibility="collapsed")
        if custom_query:
            agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
            response = agent.run(custom_query)
            st.write(response)
    except Exception as e:
        st.error(f"Error during custom query: {e}")

# Main function to run the Streamlit app
def main():
    st.title("GPT Data Analyst 🕹️")
    st.markdown("**🚀 Accelerate your data analysis with AI.**")

    # Load the OpenAI API key from Streamlit secrets
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0)

    # File uploader for CSV or Excel files
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.subheader("Data Preview:")
            st.write(df.head(6))

            # Create three columns for different analysis options
            col1, col2, col3 = st.columns([1, 1, 1.6])
            
            # Button for exploring the data
            with col1:
                if st.button("Explore the data"):
                    explore_data(df)
            
            # Button for univariate analysis
            with col2:
                if st.button("Univariate analysis"):
                    univariate_analysis(df)
            
            # Custom query input and execution
            with col3:
                custom_query(df, llm)

if __name__ == "__main__":
    main()
