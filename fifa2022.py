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

# Welcoming message and image
st.markdown("<h1 style='text-align: center;'>Welcome to FIFA World Cup 2022 Data Analysis Dashboard!</h1>", unsafe_allow_html=True)


# Guiding message
st.markdown("<p style='text-align: center;'><b>Select a page below to explore:</b></p>", unsafe_allow_html=True)

# Page selection buttons
col1, col2, col3, col4 = st.columns(4)

# Initialize session state
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Introduction'
    

# Page selection buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('1. Introduction', key='btn_intro'):
        st.session_state.app_mode = 'Introduction'
        st.experimental_rerun()

with col2:
    if st.button('2. Visualization', key='btn_visualization'):
        st.session_state.app_mode = 'Visualization'
        st.experimental_rerun()

with col3:
    if st.button('3. Prediction', key='btn_prediction'):
        st.session_state.app_mode = 'Prediction'
        st.experimental_rerun()
with col4:
    if st.button('4. Insights', key='btn_insights'):
        st.session_state.app_mode = 'Insights'
        st.experimental_rerun()

# Introduction Page
if st.session_state.app_mode == 'Introduction':
    st.subheader("Introduction")
    st.markdown("The FIFA World Cup is one of the most anticipated events in the world of sports, bringing together nations to compete for glory on the global stage. In this dashboard, we explore data from the FIFA World Cup 2022, focusing on various aspects of the tournament.")

    st.image(image_quatar2022, width=800)
    
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
elif st.session_state.app_mode == 'Visualization':
    st.subheader("Visualization")
    st.markdown("Explore visualizations of the FIFA World Cup 2022 data.")

    # Load the FIFA World Cup 2022 dataset
    df = pd.read_csv('FIFAWorldCup2022.csv')

    # Set title
    st.title("FIFA World Cup 2022 Data Analysis - Visualization")

    # Scatter plot
    st.subheader('Scatter Plot')
    scatter_independent_default = 'corners team1'
    scatter_dependent_default = 'number of goals team1'
    independent_variable_scatter = st.selectbox("Select Independent Variable", df.columns[:-1], index=df.columns.get_loc(scatter_independent_default) if scatter_independent_default in df.columns else 0, key='scatter_independent')
    dependent_variable_scatter = st.selectbox("Select Dependent Variable", df.columns[:-1], index=df.columns.get_loc(scatter_dependent_default) if scatter_dependent_default in df.columns else 0, key='scatter_dependent')
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
    hist_default = 'passes completed team1'
    independent_variable_hist = st.selectbox("Select Variable", df.columns[:-1], index=df.columns.get_loc(hist_default) if hist_default in df.columns else 0, key='hist_independent')
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=independent_variable_hist, ax=ax)
    ax.set_xlabel(independent_variable_hist)
    ax.set_title(f'Histogram of {independent_variable_hist}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Box plot
    st.subheader('Box Plot')
    box_default = 'defensive pressures applied team2'
    index_box_default = df.columns.get_loc(box_default) if box_default in df.columns else 0
    index_box = min(index_box_default, len(df.columns[:-1]) - 1)
    print("Index for box plot:", index_box)  # Debug statement
    print("Available options:", df.columns[:-1])  # Debug statement
    independent_variable_box = st.selectbox("Select Variable", df.columns[:-1], index=index_box, key='box_independent')

    # Check if the index is within the range of options
    if index_box < len(df.columns[:-1]):
        # Generate the box plot
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=df.columns[index_box], ax=ax)
        ax.set_xlabel(df.columns[index_box])
        ax.set_title(f'Box Plot of {df.columns[index_box]}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No valid variable selected for the box plot.")


    # Bar plot
    st.subheader('Bar Plot')
    bar_independent_default = 'free kicks team2'
    bar_dependent_default = 'number of goals team2'
    independent_variable_bar = st.selectbox("Select Indepent Variable", df.columns[:-1], index=df.columns.get_loc(bar_independent_default) if bar_independent_default in df.columns else 0, key='bar_independent')
    dependent_variable_bar = st.selectbox("Select Dependent Variable", df.columns[:-1], index=df.columns.get_loc(bar_dependent_default) if bar_dependent_default in df.columns else 0, key='bar_dependent')
    fig, ax = plt.subplots()
    sns.barplot(data=df, x=independent_variable_bar, y=dependent_variable_bar, ax=ax)
    ax.set_xlabel(independent_variable_bar)
    ax.set_ylabel(dependent_variable_bar)
    ax.set_title(f'Bar Plot of {dependent_variable_bar} vs {independent_variable_bar}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Additional graphs
    st.subheader('Additional Graphs')
    st.markdown("Extra graphs based on picked variables from data set (Pick Yours to Explore as well!! ):")

    # Plot 1: On Target Attempts Team1 vs Number of Goals of Team1 (Scatter Plot)
    additional_independent_default_1 = 'on target attempts team1'
    additional_dependent_default_1 = 'number of goals team1'
    additional_independent_variable_1 = st.selectbox("Select Independent Variable", df.columns[:-1], index=df.columns.get_loc(additional_independent_default_1) if additional_independent_default_1 in df.columns else 0, key='additional_independent_1')
    additional_dependent_variable_1 = st.selectbox("Select Dependent Variable", df.columns[:-1], index=df.columns.get_loc(additional_dependent_default_1) if additional_dependent_default_1 in df.columns else 0, key='additional_dependent_1')
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=additional_independent_variable_1, y=additional_dependent_variable_1, ax=ax)
    ax.set_xlabel(additional_independent_variable_1)
    ax.set_ylabel(additional_dependent_variable_1)
    ax.set_title(f'{additional_dependent_variable_1} vs {additional_independent_variable_1}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Plot 2: AssistsVS Number of Goals (Line Plot)
    additional_independent_default_2 = 'assists team2'
    additional_dependent_default_2 = 'number of goals team2'
    additional_independent_variable_2 = st.selectbox("Select Independent Variable", df.columns[:-1], index=df.columns.get_loc(additional_independent_default_2) if additional_independent_default_2 in df.columns else 0, key='additional_independent_2')
    additional_dependent_variable_2 = st.selectbox("Select Dependent Variable", df.columns[:-1], index=df.columns.get_loc(additional_dependent_default_2) if additional_dependent_default_2 in df.columns else 0, key='additional_dependent_2')
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x=additional_independent_variable_2, y=additional_dependent_variable_2, ax=ax)
    ax.set_xlabel(additional_independent_variable_2)
    ax.set_ylabel(additional_dependent_variable_2)
    ax.set_title(f'{additional_dependent_variable_2} vs {additional_independent_variable_2}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


# Prediction Page
elif st.session_state.app_mode == 'Prediction':
    st.subheader("Prediction")
    st.markdown("Select a machine learning model and variables to predict outcomes.")

    # Load the dataset
    df = pd.read_csv('FIFAWorldCup2022.csv')

    # Set default options for independent and dependent variables
    default_independent_variables = ["goal inside the penalty area team1", "attempts inside the penalty area team1", "goal outside the penalty area team1"]
    default_dependent_variable = "number of goals team1"

    # Verify that default independent variables exist in the DataFrame columns
    default_independent_variables = [var for var in default_independent_variables if var in df.columns]

    # Features and target variable selection
    selected_features = st.multiselect("Select Independent Variables", df.columns, default=default_independent_variables)
    selected_target = st.selectbox("Select Dependent Variable to Predict", df.columns, index=df.columns.get_loc(default_dependent_variable))

    if not selected_features or not selected_target:
        st.warning("Please select at least one independent variable and one dependent variable.")
    else:
        # Extract selected columns from the dataset
        df_selected = df[selected_features + [selected_target]]

        # Remove rows with missing values
        df_selected = df_selected.dropna()

        if df_selected.empty:
            st.warning("No data available after removing rows with missing values. Please choose different variables.")
        else:
            # Check if selected variables have numeric data
            numeric_columns = df_selected.select_dtypes(include=['float', 'int']).columns

            if len(numeric_columns) != len(selected_features) + 1:  # Check if all selected variables are numeric
                non_numeric_variables = [var for var in selected_features + [selected_target] if var not in numeric_columns]
                st.error(f"The following selected variables contain non-numeric values: {', '.join(non_numeric_variables)}")
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
                    try:
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

                        st.write("Interpretation:")
                        if r2 >= 0.7:
                            st.info(f"R-squared of {r2:.2f} shows that the model explains a large proportion of the variance in the dependent variable, indicating a strong relationship between the selected features and the target.")
                        elif r2 >= 0.5:
                            st.warning(f"R-squared of {r2:.2f} shows that the model explains a moderate proportion of the variance in the dependent variable, suggesting a moderate relationship between the selected features and the target.")
                        else:
                            st.error(f"R-squared of {r2:.2f} shows that the model does not explain much of the variance in the dependent variable, indicating a weak relationship between the selected features and the target.")

                    except ValueError as e:
                        st.error(f"Error: {e}. Please ensure all selected variables are numeric.")


# Insights Page
elif st.session_state.app_mode == 'Insights':
    st.subheader("Insights")

    # Insights about team performance
    st.write("### Team Performance Insights")
    st.write("1. **On Target Attempts vs. Goals:** There seems to be a positive correlation between On target Attempts and the number of goals scored by a team. Teams with higher On Target attempts tend to score more goals.")
    st.write("2. **Corners and On Target Attempts:** Teams that have more corners and on target attempts tend to create more scoring opportunities, leading to higher goal counts.")
    st.write("3. **Assists vs. Goals:** There seems to be a strongest positive correlation between Assists created and the number of goals scored by a team. Teams with higher On number of Assists would score more goals.")
    st.write("4. **Defensive Pressures Applied:** Higher defensive pressures applied by a team may indicate a more aggressive defensive strategy, potentially leading to fewer goals conceded.")

    # Insights about tournament outcomes
    st.write("### Tournament Outcome Insights")
    st.write("1. **Effect of Key Variables on Tournament Outcome:** Analyzing the impact of possession, number of goals, corners, and defensive pressures applied on overall tournament performance could provide valuable insights into successful strategies.")
    st.write("2. **Prediction Models:** Developing robust prediction models based on historical data can aid in predicting match outcomes and tournament winners, helping teams and analysts make informed decisions.")

    # Future directions
    st.write("### Future Directions")
    st.write("1. **Advanced Analytics:** Explore advanced analytics techniques such as clustering and sentiment analysis to gain deeper insights into team dynamics and fan sentiment during the tournament.")
    st.write("2. **Real-time Analysis:** Implement real-time data analysis and visualization capabilities to provide up-to-date insights during the tournament, enabling stakeholders to make timely decisions.")
