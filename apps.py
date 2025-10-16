import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Streamlit app
st.title("Agglomerative Clustering Deployment")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())

    # Encode country column
    if "Country" in df.columns:
        le = LabelEncoder()
        df["country_encoded"] = le.fit_transform(df["Country"].astype(str))
    else:
        st.warning("⚠️ 'Country' column not found, skipping encoding.")

    # Predefined scaler columns
    predefined_columns = [
        "Birth Rate", "CO2 Emissions", "Days to Start Business", "Ease of Business",
        "Energy Usage", "Health Exp % GDP", "Hours to do Tax", "Infant Mortality Rate",
        "Internet Usage", "Lending Interest", "Life Expectancy Female", "Life Expectancy Male",
        "Mobile Phone Usage", "Number of Records", "Population 0-14", "Population 15-64",
        "Population 65+", "Population Total", "Population Urban", "country_encoded"
    ]

    # Keep only columns that exist in df
    scaler_columns = [col for col in predefined_columns if col in df.columns]

   
    # Force numeric conversion
    for col in scaler_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaN with 0
    df[scaler_columns] = df[scaler_columns].fillna(0)

    # Debug check
    
    
    # Scale data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[scaler_columns]), columns=scaler_columns)

    # Agglomerative Clustering
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = agglo.fit_predict(df_scaled[scaler_columns])

    # Add cluster labels
    df["Cluster"] = cluster_labels

    # Show clustered data
    st.write("### Clustered Data", df.head())

   

    # Plot clusters (using first 2 scaled columns for visualization)
    if len(scaler_columns) >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_scaled[scaler_columns[0]], df_scaled[scaler_columns[1]],
                    c=cluster_labels, cmap="viridis", s=50)
        plt.xlabel(scaler_columns[0])
        plt.ylabel(scaler_columns[1])
        plt.title("Agglomerative Clustering")
        st.pyplot(plt.gcf())
