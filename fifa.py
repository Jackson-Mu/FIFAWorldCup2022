import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load image
image_quatar2022 = Image.open('quatar2022.jpeg')

# Title
st.title("FIFA World Cup 2022 Data Analysis")

# Sidebar
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page', ['Introduction', 'Visualization', 'Prediction', 'Insights'])

# Introduction Page
if app_mode == 'Introduction':
    st.sidebar.subheader("Introduction")
    st.sidebar.markdown("The FIFA World Cup is one of the most anticipated events in the world of sports, bringing together nations to compete for glory on the global stage. In this dashboard, we explore data from the FIFA World Cup 2022, focusing on various aspects of the tournament.")

    st.image(image_quatar2022, width=800)

    st.markdown("### Introduction")
    st.markdown("The FIFA World Cup is one of the most anticipated events in the world of sports, bringing together nations to compete for glory on the global stage. In this dashboard, we explore data from the FIFA World Cup 2022, focusing on various aspects of the tournament.")

    # Objectives
    st.markdown("### Objectives")
    st.markdown("Our objective is to analyze key factors influencing team performance and outcomes in the FIFA World Cup 2022. By examining features such as possession, number of goals scored, corners, and more, we aim to uncover insights that can aid in understanding team dynamics and strategies.")

    # Key Variables
    st.markdown("### Key Variables")
    st.markdown("In our analysis, we focus on the following key variables:")
    st.markdown("- Team")
    st.markdown("- Possession")
    st.markdown("- Number of Goals")
    st.markdown("- Corners")
    st.markdown("- On Target Attempts")
    st.markdown("- Defensive Pressures Applied")

    # Description of Data
    st.markdown("### Description of Data")
    st.markdown("Let's take a look at some descriptive statistics of the data:")

    # Load data
    df = pd.read_csv("FIFAWorldCup2022.csv")

    # Display summary statistics
    st.dataframe(df.describe())

    # Missing Values
    st.markdown("### Missing Values")
    st.markdown("Let's examine the presence of missing values in our dataset:")

    # Calculate percentage of missing values for each column
    missing_values = df.isnull().sum() / len(df) * 100

    # Display missing value percentages
    st.write("Percentage of missing values for each column:")
    st.write(missing_values)

    # Assess overall completeness of the dataset
    completeness_ratio = df.notnull().sum().sum() / (len(df) * len(df.columns))
    st.write(f"Overall completeness ratio: {completeness_ratio:.2f}")

    if completeness_ratio >= 0.85:
        st.success("The dataset has a high level of completeness, providing us with reliable data for analysis.")
    else:
        st.warning("The dataset has a low level of completeness, which may affect the reliability of our analysis.")

    # Conclusion
    st.markdown("### Conclusion")
    st.markdown("In this dashboard, we explore the FIFA World Cup 2022 dataset and analyze key variables related to team performance. By understanding the dynamics of possession, goals, corners, and defensive pressures applied, we gain insights into the factors influencing team success in the tournament.")

# Visualization Page
elif app_mode == 'Visualization':
    st.sidebar.subheader("Visualization")
    st.sidebar.markdown("Explore visualizations of the FIFA World Cup 2022 data.")

    # Load the FIFA World Cup 2022 dataset
    df = pd.read_csv('FIFAWorldCup2022.csv')

    # Set default variables for the first five plots
    default_independent_variables = df.columns[-2:]  # Last two variables
    default_dependent_variable = df.columns[-1]  # Last variable

    # Set title
    st.title("FIFA World Cup 2022 Data Analysis - Visualization")

    # Scatter plot
    st.subheader('Scatter Plot')
    independent_variable_scatter = st.selectbox("Select Independent Variable", df.columns, index=len(df.columns)-1, key='scatter_independent')
    dependent_variable_scatter = st.selectbox("Select Dependent Variable", df.columns, index=len(df.columns)-2, key='scatter_dependent')
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=independent_variable_scatter, y=dependent_variable_scatter, ax=ax)
    ax.set_xlabel(independent_variable_scatter)
    ax.set_ylabel(dependent_variable_scatter)
    ax.set_title(f'{dependent_variable_scatter} vs {independent_variable_scatter}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Histogram
    st.subheader('Histogram')
    independent_variable_hist = st.selectbox("Select Variable", df.columns, index=len(df.columns)-1, key='hist_independent')
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=independent_variable_hist, ax=ax)
    ax.set_xlabel(independent_variable_hist)
    ax.set_title(f'Histogram of {independent_variable_hist}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Box plot
    st.subheader('Box Plot')
    independent_variable_box = st.selectbox("Select Variable", df.columns, index=len(df.columns)-1, key='box_independent')
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=independent_variable_box, ax=ax)
    ax.set_xlabel(independent_variable_box)
    ax.set_title(f'Box Plot of {independent_variable_box}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Bar plot
    st.subheader('Bar Plot')
    independent_variable_bar = st.selectbox("Select Variable", df.columns, index=len(df.columns)-1, key='bar_independent')
    dependent_variable_bar = st.selectbox("Select Dependent Variable", df.columns, index=len(df.columns)-2, key='bar_dependent')
    fig, ax = plt.subplots()
    sns.barplot(data=df, x=independent_variable_bar, y=dependent_variable_bar, ax=ax)
    ax.set_xlabel(independent_variable_bar)
    ax.set_ylabel(dependent_variable_bar)
    ax.set_title(f'Bar Plot of {dependent_variable_bar} vs {independent_variable_bar}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Pair plot
    st.subheader('Pair Plot')
    fig = sns.pairplot(df)
    st.pyplot(fig)

# Prediction Page
elif app_mode == 'Prediction':
    st.sidebar.subheader("Prediction")
    st.sidebar.markdown("Predict outcomes using machine learning models.")

    st.markdown("### Prediction")
    st.markdown("Select a machine learning model and variables to predict outcomes.")

    # Load the dataset
    df = pd.read_csv('FIFAWorldCup2022.csv')

    # Features and target variable selection
    selected_features = st.multiselect("Select Independent Variables", df.columns)
    selected_target = st.selectbox("Select Dependent Variable to Predict", df.columns)

    if not selected_features or not selected_target:
        st.warning("Please select at least one independent variable and one dependent variable.")
    else:
        # Remove non-numeric values and drop rows with missing values
        selected_columns = selected_features + [selected_target]
        df_selected = df[selected_columns].apply(pd.to_numeric, errors='coerce').dropna()

        if df_selected.empty:
            st.warning("No numeric data available for the selected variables. Please choose different variables.")
        else:
            X = df_selected[selected_features]
            y = df_selected[selected_target]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Guiding message
            st.info("Select a machine learning model.")

            # Select model
            selected_model = st.selectbox("Select Model", ['Linear Regression'])

            if selected_model == 'Linear Regression':
                # Train the model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Evaluate the model
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                # Model Performance Visualization
                st.subheader("Model Performance Visualization")

                # Create scatter plot using Matplotlib
                fig, ax = plt.subplots()

                try:
                    ax.scatter(y_test, y_pred)
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("Actual vs Predicted Values")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                except RuntimeError as e:
                    st.error("An error occurred while generating the plot.")

                    # Find problematic variable(s) causing the error
                    problematic_variables = []

                    if "x" in str(e):
                        problematic_variables.append("x")
                    if "y" in str(e):
                        problematic_variables.append("y")

                    # Remove problematic variable(s)
                    for variable in problematic_variables:
                        st.info(f"Removing variable '{variable}' to retry plot generation.")
                        # Remove the variable from the selection

                    # Retry Mechanism
                    if st.button("Retry"):
                        st.warning("Retrying plot generation...")
                        # Retry generating the plot here
                        # This could involve re-selecting the variables or re-running the plotting code


                # Display model performance metrics
                st.subheader("Model Performance Metrics")
                st.write(f"Linear Regression Model Performance:")
                st.write(f"R-squared: {r2:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")

# Insights Page
elif app_mode == 'Insights':
    st.sidebar.subheader("Insights")
    st.sidebar.markdown("Discover insights from the FIFA World Cup 2022 data.")

    st.subheader("Insights")

    # Insights about team performance
    st.write("### Team Performance Insights")
    st.write("1. **Possession vs. Goals:** There seems to be a positive correlation between possession and the number of goals scored by a team. Teams with higher possession tend to score more goals.")
    st.write("2. **Corners and On Target Attempts:** Teams that have more corners and on target attempts tend to create more scoring opportunities, leading to higher goal counts.")
    st.write("3. **Defensive Pressures Applied:** Higher defensive pressures applied by a team may indicate a more aggressive defensive strategy, potentially leading to fewer goals conceded.")

    # Insights about tournament outcomes
    st.write("### Tournament Outcome Insights")
    st.write("1. **Effect of Key Variables on Tournament Outcome:** Analyzing the impact of possession, number of goals, corners, and defensive pressures applied on overall tournament performance could provide valuable insights into successful strategies.")
    st.write("2. **Prediction Models:** Developing robust prediction models based on historical data can aid in predicting match outcomes and tournament winners, helping teams and analysts make informed decisions.")

    # Future directions
    st.write("### Future Directions")
    st.write("1. **Advanced Analytics:** Explore advanced analytics techniques such as clustering and sentiment analysis to gain deeper insights into team dynamics and fan sentiment during the tournament.")
    st.write("2. **Real-time Analysis:** Implement real-time data analysis and visualization capabilities to provide up-to-date insights during the tournament, enabling stakeholders to make timely decisions.")
