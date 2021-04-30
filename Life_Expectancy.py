# Imports and dataset load
import csv
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot
from scipy.stats import stats
from scipy.stats.mstats import winsorize
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, \
    classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import plotly.graph_objects as go

# GLOBAL VARIABLES
# Load data into pandas data frame
pd.set_option('display.width', 480)
pd.set_option("display.max_columns", 20)
df = pd.read_csv('Life_Expectancy_WHO_Data.csv')
wins_df = pd.DataFrame()  # Winsorized dataframe holder

wins_dict = {}  # Winsorized feature dictionary
feature_vars = list(df.columns)[3:]  # List of numerical features


def main():
    # EXPLORATORY DATA ANALYSIS
    # DATA CLEANING
    view_df()  # View the dataframe
    rename_column_heads()  # Rename column heads for readability

    # NULLS
    describe_df()  # Describe view stats on dataframe
    boxplot_nulls()  # Box plots for the potential outliers
    null_summary()  # Check for and summarize missing values
    inexplicit_nan_adj()  # Adjusting implicit NANs
    null_explicit_details()  # Detail the explicit nulls
    remove_nulls()  # Drop rows/countries with explicit nulls
    imputed_data()  # Impute missing data values
    null_explicit_details()  # View null breakdown again

    # OUTLIERS
    detect_outlier_plots()  # Detect outliers with boxplots and histograms
    outlier_summary()  # Count outliers and plot

    # WINSORIZATION
    all_check_wins_plot()  # Set windsorization vars with limits
    visualize_wins_dist()  # Visual winsorized distributions
    create_wins_df()  # Create winsorized columns
    check_wins_country()  # Verify that winsorization was been applied correctly by Country
    check_wins_year()  # Verify that winsorization was been applied correctly by Year

    # DATA EXPLORATION
    #le_desity_histplot()  # Histogram of Life Expectancy density
    #le_by_status()  # Life Expectancy grouped by country and status
    #scatterplot_features_corr()  # Scatterplot showing feature correlation
    #pairplot_corr()  # Plot pairs for correlation testing

    # MULTI-COLLINEARITY
    #corr_matrix_heatmap()  # Correlation matrix of winsorized feature's correlation
    #line_plot_by_year()  # Life Expectancy time series by Year
    remove_features()  # Removing features with high correlation
    #create_data_corr()  # Create correlation table of dependent variable with independent variables

    # Life Expectancy Ranking Summaries
    #le_rankings('Top', 'Mean', 15)  # For Top Mean Countries
    #le_rankings('Top', 'Median', 15)  # For Top Median Countries
    #le_rankings('Bottom', 'Mean', 15)  # For Bottom Mean Countries
    #le_rankings('Bottom', 'Median', 15)  # For Bottom Median Countries

    #boxplot_status()  # Box plot by country status
    #country_status_lineplot()  # Time series line plot of Life Expectancy by status

    # TRAIN, TEST, and EVALUATE MODEL (MACHINE LEARNING)
    linear_regression()  # Linear regression on GDP and Percentage Expenditure
    multiple_linear_regression()  # Multiple Linear Regression of winsorized data
    polynomial_regression()  # Polynomial Regression of winsorized data
    decision_tree_regression()  # Decision Tree Regression of winsorized data
    random_forest_regression()  # Random Forest of winsorized data
    logistic_regression()  # Logistic Regression of winsorized data
    ttest()  # Ttest to determine significance of difference between developed and developing

    # TUNING THE MODEL
    # "prediction_tune_model()" is called from logistic_regression()


# EXPLORATORY DATA ANALYSIS FUNCTIONS

# View the dataframe
def view_df_to_image(text_body, file_name):
    df_styled = text_body.style.background_gradient()  # Style the text
    dfi.export(df_styled, 'charts/' + file_name + '.png')  # Create image of text


# View the dataframe
def view_df():
    print("shape(): \n", df.shape)
    print("\n")

    print("dtypes(): \n", df.dtypes)
    print("\n")
    dfi.export(pd.DataFrame(df.dtypes).style.background_gradient(), 'charts/Orig_Column_Headings.png')

    print("head(): \n", df.head(5))
    text_body = df.head(5)
    view_df_to_image(text_body, 'Head(5)')  # Create image of head()

    print("df.tail(5): \n", df.tail(5))
    text_body = df.tail(5)
    view_df_to_image(text_body, 'Tail(5)')  # Create image of tail()


# DATA CLEANING

# Rename column heads for readability
def rename_column_heads():
    global df, feature_vars
    orig_column_heads = list(df.columns)
    new_column_heads = []
    for col in orig_column_heads:  # Replace spaces with underscores and "title" capitalize text
        new_column_heads.append(col.strip().title().replace('  ', ' ').replace(' ', '_'))
    df.columns = new_column_heads

    # Rename column heads to properly reflect meaning
    df.rename(columns={'Thinness_1-19_Years': 'Thinness_10-19_Years'}, inplace=True)
    df.rename(columns={'Percentage_Expenditure': 'Pct_Expenditure'}, inplace=True)
    df.rename(columns={'Total_Expenditure': 'Ttl_Expend'}, inplace=True)
    df.rename(columns={'Bmi': 'BMI'}, inplace=True)
    df.rename(columns={'Gdp': 'GDP'}, inplace=True)
    df.rename(columns={'Hiv/Aids': 'HIV_AIDS'}, inplace=True)
    df.rename(columns={'Income_Composition_Of_Resources': 'Income_Comp_Of_Resources'}, inplace=True)
    feature_vars = list(df.columns)[3:]

    print(type(df.dtypes))
    dfi.export(pd.DataFrame(df.dtypes).style.background_gradient(),
               'charts/New_Column_Headings.png')  # Create image of text


# NULLS VALUES
# Describe view stats on dataframe
def describe_df():
    print("describe(): \n", df.describe().iloc[:, 1:])


# Box plots for the potential outliers
def boxplot_nulls():
    outlier_features = ['Adult_Mortality', 'Infant_Deaths', 'BMI', 'Under-Five_Deaths', 'GDP', 'Population']
    plt.figure(figsize=(15, 8))

    for i, col in enumerate(outlier_features, start=1):
        plt.subplot(2, 3, i)
        plt.xticks([1, 2, 3, 4, 5, 6], outlier_features)
        df.boxplot(col, notch=True, patch_artist=True)

    pyplot.suptitle('Box Plot Outliers', fontsize=20, verticalalignment='top', horizontalalignment='center',
                    fontweight='bold')
    plt.savefig('charts/BoxPlot_Outliers.png', dpi=None, facecolor='w', edgecolor='g', orientation='landscape',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
    plt.show()


# Adjusting implicit NANs
def inexplicit_nan_adj():
    global df
    mort_5_percentile = np.percentile(df.Adult_Mortality.dropna(), 5)
    df.adult_mortality = df.apply(lambda x: np.nan if x.Adult_Mortality < mort_5_percentile else x.Adult_Mortality,
                                  axis=1)
    df.Infant_Deaths = df.Infant_Deaths.replace(0, np.nan)
    df['Under-Five_Deaths'] = df['Under-Five_Deaths'].replace(0, np.nan)  # Update global dataframe var
    df.BMI = df.apply(lambda x: np.nan if (x.BMI < 10 or x.BMI > 50) else x.BMI, axis=1)
    print("df.info():", df.info())  # Look at nulls that are left


# Detail the explicit nulls
def null_explicit_details():
    global df
    df_cols = list(df.columns)
    cols_total_count = len(list(df.columns))
    cols_count = 0
    print("\n")
    for loc, col in enumerate(df_cols):  # Iterate through the columns to find nulls
        null_count = df[col].isnull().sum()
        total_count = df[col].isnull().count()
        percent_null = round(null_count / total_count * 100, 2)
        if null_count > 0:
            cols_count += 1
            print('[iloc = {}] {} has {} null values: {}% null'.format(loc, col, null_count, percent_null))
    cols_percent_null = round(cols_count / cols_total_count * 100, 2)
    print('{} out of {} columns contain null values; {}% columns contain null values.'.format(cols_count,
                                                                                              cols_total_count,
                                                                                              cols_percent_null))


# Check for and summarize missing values
# Does basically the same as "null_details(df=df):", but prints to image
def null_summary():
    na_values = df.isna().sum().reset_index()  # Find NA values
    na_values.columns = ["Features", "Missing_Values"]
    na_values["Missing_Percent"] = round(na_values.Missing_Values / len(df) * 100, 2)
    print("Summarize NULLS: \n", na_values[na_values.Missing_Values > 0])

    df_nulls = na_values[na_values.Missing_Values > 0]
    dfi.export(pd.DataFrame(df_nulls).style.background_gradient(), 'charts/Summarize_Nulls.png')  # Print to image
    df.info()


# Drop rows/countries with no life expectancy data
def remove_nulls():
    global df, feature_vars
    df.Life_Expectancy.notnull()  # Removes countries with no life expectancy data
    df.drop(columns='BMI', inplace=True)  # Most values are null for BMI so it's removed
    feature_vars = list(df.columns)[3:]  # Update global features list var


# Impute missing data values
def imputed_data():
    global df
    imputed_data = []
    for year in list(df.Year.unique()):
        year_data = df[df.Year == year].copy()
        for col in list(year_data.columns)[3:]:  # Get numerical features
            year_data[col] = year_data[col].fillna(year_data[col].dropna().mean()).copy()  # Impute the data
        imputed_data.append(year_data)
    df = pd.concat(imputed_data).copy()  # Update global dataframe var
    null_explicit_details()  # View null breakdown after imputing


# OUTLIERS

# Detect outliers with boxplots and histograms
def detect_outlier_plots():
    plt.figure(figsize=(15, 30))
    i = 0
    for col in feature_vars:
        i += 1
        plt.subplot(9, 4, i)
        plt.boxplot(df[col])
        plt.title('{}'.format(col), fontsize=9)
        i += 1
        plt.subplot(9, 4, i)
        plt.hist(df[col])
        plt.title('{}'.format(col), fontsize=9)
    pyplot.suptitle('Detect Outliers', fontsize=16, verticalalignment='top', horizontalalignment='center',
                    fontweight='bold')
    plt.savefig('charts/Detect_Outlier_Plots.png', dpi=None, facecolor='w', edgecolor='g', orientation='portrait',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.0, metadata=None)
    plt.show()


# Count outliers and plot
def outlier_summary():
    with open('charts/Outlier_Summary.txt', mode='w', newline='') as outlier_file:
        outlier_writer = csv.writer(outlier_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for col in feature_vars:
            q75, q25 = np.percentile(df[col], [75, 25])
            iqr = q75 - q25
            min_val = q25 - (iqr * 1.5)
            max_val = q75 + (iqr * 1.5)
            count = len(np.where((df[col] > max_val) | (df[col] < min_val))[0])
            percent = round(count / len(df[col]) * 100, 2)

            total_chars = 45
            len_first_chars = 7
            len_col = len(col)
            chars_needed = total_chars - len_col - len_first_chars

            print(len_first_chars * '-' + col + ' Outliers' + chars_needed * '-')
            print('Count: {}'.format(count))
            print('Percentage of Data: {}%'.format(percent))

            outlier_writer.writerow([len_first_chars * '-' + col + ' Outliers' + chars_needed * '-'])
            outlier_writer.writerow(['Count: {}'.format(count)])
            outlier_writer.writerow(['Percentage of Data: {}%'.format(percent)])
            outlier_writer.writerow(' ')


# Set windsorization vars with limits
def all_check_wins_plot():
    check_wins_plot(feature_vars[0], lower_limit=.01, show_plot=True)
    check_wins_plot(feature_vars[1], upper_limit=.04, show_plot=False)
    check_wins_plot(feature_vars[2], upper_limit=.05, show_plot=False)
    check_wins_plot(feature_vars[3], upper_limit=.0025, show_plot=False)
    check_wins_plot(feature_vars[4], upper_limit=.135, show_plot=False)
    check_wins_plot(feature_vars[5], lower_limit=.1, show_plot=False)
    check_wins_plot(feature_vars[6], upper_limit=.19, show_plot=False)
    check_wins_plot(feature_vars[7], upper_limit=.05, show_plot=False)
    check_wins_plot(feature_vars[8], lower_limit=.1, show_plot=False)
    check_wins_plot(feature_vars[9], upper_limit=.02, show_plot=False)
    check_wins_plot(feature_vars[10], lower_limit=.105, show_plot=False)
    check_wins_plot(feature_vars[11], upper_limit=.185, show_plot=True)
    check_wins_plot(feature_vars[12], upper_limit=.105, show_plot=False)
    check_wins_plot(feature_vars[13], upper_limit=.07, show_plot=False)
    check_wins_plot(feature_vars[14], upper_limit=.035, show_plot=False)
    check_wins_plot(feature_vars[15], upper_limit=.035, show_plot=False)
    check_wins_plot(feature_vars[16], lower_limit=.05, show_plot=False)
    check_wins_plot(feature_vars[17], lower_limit=.025, upper_limit=.005, show_plot=False)


# Winsorize dataframe to "wins" dictionary
def check_wins_plot(col, lower_limit=0, upper_limit=0, show_plot=True):
    global wins_dict
    wins_data = winsorize(df[col], limits=(lower_limit, upper_limit))  # Create winsorized col entries
    wins_dict[col] = wins_data  # Add the winsorized data to the wins dictionary

    if show_plot == True:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.boxplot(df[col])
        plt.title('Original Data {}'.format(col), fontweight='bold')
        plt.subplot(1, 2, 2)
        plt.boxplot(wins_data)
        pyplot.suptitle('Compare Before/After Winsorized\n', fontsize=16, verticalalignment='top',
                        horizontalalignment='center', fontweight='bold')
        plt.title('Winsorised Data=({},{}) {}'.format(0, 0, col), fontweight='bold')
        plt.savefig('charts/Compare_Winsorized' + col + '.png', dpi=None, facecolor='w', edgecolor='g', orientation='landscape',
                    format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
        plt.show()


# Visual winsorized distributions
def visualize_wins_dist():
    plt.figure(figsize=(15, 20))
    for i, col in enumerate(feature_vars, 1):
        plt.subplot(5, 4, i)
        plt.hist(wins_dict[col])
        plt.title(col)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    plt.savefig('charts/Visualize_Winsorized.png', dpi=None, facecolor='w', edgecolor='g', orientation='portrait',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
    plt.show()


# DATA EXPLORATION

# Create winsorized dataframe
def create_wins_df():
    global wins_df
    wins_df = df.iloc[:, 0:3]  # Creating new winsorized dataframe
    for col in list(df.columns)[3:]:
        wins_df[col] = wins_dict[col]  # Replace dict data with winsorized data

    print("wins_df.describe(): \n", wins_df.describe())  # Show descriptive statistics
    print("wins_df.describe(include='O'): \n", wins_df.describe(include='O'))  # Show count/frequency/max/quartiles


# Verify that winsorization was consistently applied
def check_wins_country():
    plt.figure(figsize=(15, 25))
    wins_df.Country.value_counts(ascending=True).plot(kind='barh')
    plt.title('Check of Winsorization by Country')
    plt.xlabel('Count of Rows')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig('charts/Check_Wins_by_Country.png', bbox_inches='tight')
    plt.show()


# Verify that winsorization was been applied correctly by Year
def check_wins_year():
    wins_df.Year.value_counts().sort_index().plot(kind='barh')
    plt.title('Check of Winsorization by Year')
    plt.xlabel('Count of Rows')
    plt.ylabel('Year')
    plt.savefig('charts/Check_Wins_by_Year.png', bbox_inches='tight')
    plt.show()


# Histogram of Life Expectancy density
def le_desity_histplot():
    plt.figure(figsize=(20, 5))
    plt.title('Density - Life_Expectancy', fontweight='bold')
    sns.histplot(wins_df['Life_Expectancy'].dropna(), kde=True, color="darkgoldenrod")
    plt.savefig('charts/Density - Life_Expectancy.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.title('Density - Income_Comp_Of_Resources', fontweight='bold')
    sns.histplot(wins_df['Income_Comp_Of_Resources'].dropna(), kde=True, color="darkgoldenrod")
    plt.savefig('charts/Density - Income_Comp_Of_Resources.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.title('Density - Schooling', fontweight='bold')
    sns.histplot(wins_df['Schooling'].dropna(), kde=True, color="darkgoldenrod")
    plt.savefig('charts/Density - Schooling.png', bbox_inches='tight')
    plt.show()


# Life Expectancy grouped by country and status
def le_by_status():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    wins_df.Status.value_counts().plot(kind='bar')
    plt.title('Life Expectancy by Country Status', fontweight='bold')
    plt.xlabel('Country Status')
    plt.ylabel('Number of Rows')
    plt.xticks(rotation=0)
    plt.subplot(1, 2, 2)
    wins_df.Status.value_counts().plot(kind='pie', autopct='%.2f')
    plt.ylabel('')
    plt.title('Country Status Pie Chart', fontweight='bold')
    plt.savefig('charts/Life_Expectancy_by_Country_Status.png', bbox_inches='tight')
    plt.show()


# MULTI-COLLINEARITY

# Correlation matrix of winsorized feature's correlation
def corr_matrix_heatmap():
    mask = np.triu(wins_df[feature_vars].corr())  # Get upper triangle of array
    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(bottom=.2)
    # tight_layout()
    sns.heatmap(wins_df[feature_vars].corr(), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0,
                cmap='Spectral', mask=mask)
    plt.ylim(18, 0)
    plt.title('Correlation Matrix Heatmap', fontsize=24, verticalalignment='top', horizontalalignment='center',
              color='black', fontweight='bold')
    plt.savefig('charts/Corr_Matrix_Heatmap.png', dpi=None, facecolor='w', edgecolor='g', orientation='landscape',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.25, metadata=None)
    plt.show()


# Life Expectancy time series by Year
def line_plot_by_year():
    sns.lineplot(data=wins_df, x='Year', y='Life_Expectancy', marker='^')
    plt.title('Life Expectancy by Year', fontweight='bold')
    plt.savefig('charts/Line_Plot_Life_Expectancy_by_Year.png', bbox_inches='tight')
    plt.show()

    wins_df.Year.corr(wins_df.Life_Expectancy)  # Find the pairwise correlation


# Removing features with high correlation
def remove_features():
    global wins_df, feature_vars
    rem_features = ['Thinness_5-9_Years', 'GDP', 'Infant_Deaths', 'Population']
    for f in rem_features:
        wins_df.drop(f, axis=1)
    feature_vars = list(wins_df.columns)[3:]

    print("\n Remove Features Confirmation: \n", wins_df.info)


# Create correlation table of dependent variable with independent variables
def create_data_corr():
    df_styled = pd.DataFrame(wins_df.corr()).style.background_gradient()  # Style the text
    dfi.export(df_styled, 'charts/Create_Data_Corr.png')  # Create image from text


# Scatterplot showing feature correlation
def scatterplot_features_corr():
    wins_df_copy = wins_df.drop(['Life_Expectancy'], axis=1)
    categorical_cols = wins_df_copy.select_dtypes(include="O")
    numerical_cols = wins_df_copy.select_dtypes(exclude="O")

    for col in numerical_cols.columns:
        sns.scatterplot(x=numerical_cols[col], y=wins_df["Life_Expectancy"], hue=categorical_cols.Status)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel("Life Expectancy", fontsize=10)
        plt.xlabel(col, fontsize=10)
        plt.title(col + ' to Life Expectancy Correlation', fontweight='bold', fontsize=12)
        plt.savefig('charts/Scatterplot_' + col + '_corr.png', dpi=None, facecolor='w', edgecolor='g',
                    orientation='landscape',
                    format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
        plt.show()


# Plot pairs for correlation testing
def pairplot_corr():
    sns.pairplot(wins_df,
                 vars=['Adult_Mortality', 'Infant_Deaths', 'Alcohol', 'Ttl_Expend', 'GDP', 'Population'],
                 hue='Status')
    plt.title('Pairplot Correlation', fontweight='bold', fontsize=12)
    plt.savefig('charts/Pairplot_Correlation.png', dpi=None, facecolor='w', edgecolor='g',
                orientation='landscape',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
    plt.show()


# Prepares ranking summaries to be compiled and sent to image file
def le_rankings(rank, stat_type, num_rows):
    order = False if rank == 'Top' else True

    # Generate text body for rankings
    if stat_type == 'Mean':
        text_body = pd.DataFrame(
            wins_df.groupby(['Country']).Life_Expectancy.mean().sort_values(ascending=order).head(num_rows))
    elif stat_type == 'Median':
        text_body = pd.DataFrame(
            wins_df.groupby(['Country']).Life_Expectancy.median().sort_values(ascending=order).head(num_rows))

    # Create images from text for all Top, Bottom, Mean, and Median combinations
    title = rank + ' ' + str(num_rows) + ' Life Expectancy (' + stat_type + ')'
    df_styled = text_body.style.background_gradient()  # Style the image
    dfi.export(df_styled, 'charts/' + rank + '_' + stat_type + '_' + 'Rankings.png')  # Create image from text
    print("\n" + str(title) + "\n" + str(text_body))


# Box plot by country status
def boxplot_status():
    sns.boxplot(x=wins_df['Status'], y=wins_df['Life_Expectancy'], fliersize=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.ylabel("Life Expectancy", fontsize=12)
    plt.xlabel("Status", fontsize=12)
    plt.title("Life Expectancy by Country Status", fontweight='bold', fontsize=13)
    plt.savefig('charts/Boxplot_Status.png', dpi=None, facecolor='w', edgecolor='g', orientation='landscape',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
    plt.show()


# Time series line plot of Life Expectancy by status
def country_status_lineplot():
    le_year = wins_df.groupby(by=['Year', 'Status']).mean().reset_index()
    Developed = le_year.loc[le_year['Status'] == 'Developed', :]
    Developing = le_year.loc[le_year['Status'] == 'Developing', :]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=Developing['Year'], y=Developing['Life_Expectancy'],
                              mode='lines',
                              name='Developing',
                              marker_color='blue'))
    fig1.add_trace(go.Scatter(x=Developed['Year'], y=Developed['Life_Expectancy'],
                              mode='lines',
                              name='Developed',
                              marker_color='orange'))
    fig1.update_layout(
        height=500,
        xaxis_title="Years",
        yaxis_title='Life expectancy in age',
        title_text='<b>Average Life Expectancy by Country Status</b>')
    fig1.show()


# TRAIN, TEST, EVALUATE THE MODEL (MACHINE LEARNING)

# Linear regression on GDP and Percentage Expenditure
def linear_regression():
    data = wins_df.copy() # Copy winsorized dataframe
    lr_df = data.dropna()  # Remove NA values
    regr = LinearRegression()
    x = lr_df.GDP.values.reshape(-1, 1)
    y = lr_df['Pct_Expenditure'].values.reshape(-1, 1)
    regr.fit(x, y)  # Fit the linear model

    lr_predict = regr.predict(([[10000]]))
    print("Linear regr.predict: ", lr_predict)
    lr_coef = regr.coef_
    print("Linear lr_coef: ", lr_coef)

    x_array = np.arange(min(lr_df.GDP), max(lr_df.GDP)).reshape(-1, 1)  # Create prediction line
    plt.scatter(x, y)  # Plot the regression
    y_head = regr.predict(x_array)  # Predict percentage of expenditure

    plt.plot(x_array, y_head, color="red")
    plt.ylabel("Percentage Expenditure", fontsize=12)
    plt.xlabel("GDP", fontsize=12)
    plt.title("Linear Regression - GDP & Percentage Expenditure", fontweight='bold', fontsize=13)
    plt.savefig('charts/Linear_Regression.png', dpi=None, facecolor='w', edgecolor='g', orientation='landscape',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
    plt.show()

    print("Linear r2 Score: ", r2_score(y, regr.predict(x)))
    print("Mean Absolute Error: {:.4f}".format(metrics.mean_absolute_error(x_array, y_head)))
    print("Mean Squared Error: {:.4f}".format(metrics.mean_squared_error(x_array, y_head)))
    print("Root Mean Squared Error: {:.4f}".format(np.sqrt(metrics.mean_squared_error(x_array, y_head))))


# Multiple linear regression of winsorized data
def multiple_linear_regression():
    mlr_df = wins_df.dropna().copy()  # Copy winsorized dataframe

    mlr_df.drop(["Country", "Status"], axis=1, inplace=True)  # Drop because the are categorical
    x = mlr_df.iloc[:, [-2, -1]].values  # The dependent variables/features
    print("print X: ", x)
    y = mlr_df['Pct_Expenditure'].values.reshape(-1, 1)  # The independent variable/features

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)  # split data
    regr = LinearRegression()
    model = regr.fit(x_train, y_train)  # Fit the LR model

    print("mlr_intercept_: ", regr.intercept_)
    print("intercept_coef_: ", regr.coef_)

    new_data = [[0.4, 8], [0.5, 10]]  # Random data to test
    new_data = pd.DataFrame(new_data).T
    model.predict(new_data)

    # Mean squared error regression loss
    mserl = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
    print("Mean squared error regression loss: ", mserl)

    # Train and predict model
    model.score(x_train, y_train)  # Coefficient of determination
    cross_val_score(model, x_train, y_train, cv=10, scoring="r2").mean()  # Evaluate the score by cross-validation
    y_head = model.predict(x_test)
    y_head[0:5]  # Get first 5

    # Calculate r2 score
    r2_degeri = r2_score(y_test, y_head)
    print("\nMLR r2 Error = ", r2_degeri)


# Polynomial Regression of winsorized data
def polynomial_regression():
    poly_df = wins_df.dropna().copy()  # Copy winsorized dataframe
    regr = LinearRegression()
    x = poly_df.GDP.values.reshape(-1, 1)
    y = poly_df['Pct_Expenditure'].values.reshape(-1, 1)
    regr.fit(x, y)  # Fit linear model first

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)  # Split data

    poly_regr = PolynomialFeatures(degree=15)
    x_polynomial = poly_regr.fit_transform(x)
    regr.fit(x_polynomial, y)  # Fit the polynomial features model
    y_head = regr.predict(x_polynomial)

    poly_features = PolynomialFeatures(degree=8)
    level_poly = poly_features.fit_transform(x_train)
    regr.fit(level_poly, y_train)  # Fit the trained model
    y_head = regr.predict(poly_features.fit_transform(x_train))
    y_test = np.array(range(0, len(y_train)))

    r2 = r2_score(y_train, y_head)
    print("Polynomial r2 Value: ", r2)  # percentage of significance

    plt.scatter(y_test, y_train, color="blue")
    plt.scatter(y_test, y_head, color="orange")
    plt.xlabel("GDP")
    plt.ylabel("Percentage Expenditure")
    plt.title("Polynomial Regression - Percentage Expenditure", fontweight='bold', fontsize=13)
    plt.savefig('charts/Polynomial_Regression.png', dpi=None, facecolor='w', edgecolor='g', orientation='landscape',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
    plt.show()


# Decision Tree Regression of winsorized data
def decision_tree_regression():
    dtr_df = wins_df.dropna().copy()  # Copy winsorized dataframe

    x = dtr_df.GDP.values.reshape(-1, 1)
    y = dtr_df['Pct_Expenditure'].values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)  # split data

    regr = DecisionTreeRegressor()  # created model
    regr.fit(x_train, y_train)  # fitted model according to train values

    print("Decision Tree Prediction: ", regr.predict([[1000]]))

    x_array = np.arange(min(x), max(x), 0.01).reshape(-1, 1)  # line information to be drawn as a predict
    y_head = regr.predict(x_array)  # percentage of spend estimate


# Random Forest of winsorized data
def random_forest_regression():
    rfr_df = wins_df.dropna().copy()

    x = rfr_df.GDP.values.reshape(-1, 1)
    y = rfr_df['Pct_Expenditure'].values
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(x, y)  # The ideal fit line

    print("Random Forest Prediction: ", regr.predict([[1000]]))
    print("\n")

    x_array = np.arange(min(x), max(x), 0.01).reshape(-1, 1)  # line information to be drawn as a predict
    y_head = regr.predict(x_array)  # percentage of spend predict


# Logistic Regression of winsorized data
def logistic_regression():
    logi_df = wins_df.dropna().copy()  # Copy winsorized dataframe

    logi_df.drop(["Country"], axis=1, inplace=True)  # Drop categorical feature
    logi_df['Status'].value_counts()

    plt.title('Logistic Regression', fontweight='bold')
    plt.ylabel('Number of Rows')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.title("Logistic Regression", fontweight='bold', fontsize=13)
    plt.savefig('charts/Logistic_Regression.png', dpi=None, facecolor='w', edgecolor='g', orientation='landscape',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
    logi_df.Status.value_counts().plot(kind='bar')
    logi_df.Status = [1 if each == "Developing" else 0 for each in logi_df.Status]  # Convert to nominal vals
    print("log_reg - logi_df.describe().T", logi_df.describe().T)

    # Normalize the data
    y = logi_df['Status']
    X_data = logi_df.drop(['Status'], axis=1)
    X = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)).values

    # Implement model
    logi = sm.Logit(y, X_data)
    logi_model = logi.fit()
    print("Model Summary()", logi_model.summary())
    print("\n")

    # Create & plot summary
    fig_size = (10, 8)  # w x h
    plt.figure(figsize=fig_size)
    plt.axis('off')
    plt.title("Logistical Model Summary", fontsize=12, verticalalignment='baseline', horizontalalignment='center',
              color='white', backgroundcolor='orange')
    plt.text(0, 0, str(logi_model.summary()), fontsize=10, fontproperties='monospace', verticalalignment='bottom',
             horizontalalignment='left')
    plt.savefig("charts/Logistical_Model_Summary.png", dpi=None, facecolor='w', edgecolor='g', orientation='portrait',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.05, metadata=None)

    logi = LogisticRegression(solver="liblinear")
    logi_model = logi.fit(X, y)  # The ideal fit line

    print("logi_model.intercept_ :", logi_model.intercept_)
    print("logi_model.coef_ :", logi_model.coef_)

    prediction_tune_model(logi_model, X, y)  # Call model tuning function


# Ttest to determine significance of difference between developed and developing
def ttest():
    #ttest_results = stats.ttest_ind(wins_df.loc[wins_df['Status']=='Developed','Life_Expectancy'],
    #                wins_df.loc[wins_df['Status']=='Developing','Life_Expectancy'])
    developed_le = wins_df[wins_df.Status == 'Developed'].Life_Expectancy
    developing_le = wins_df[wins_df.Status == 'Developing'].Life_Expectancy
    ttest_results = stats.ttest_ind(developed_le, developing_le, equal_var=False)

    print("ttest_results: ", ttest_results)


# TUNE MODEL
# Model tuning function
def prediction_tune_model(logi_model, X, y):
    y_pred = logi_model.predict(X)
    confusion_matrix(y, y_pred)
    accuracy_score(y, y_pred)

    y_probs = logi_model.predict_proba(X)
    y_probs = y_probs[:, 1]
    y_pred = [1 if i > 0.5 else 0 for i in y_probs]

    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))  # Evaluate the accuracy of classification
    print("\nAccuracy Score: ", accuracy_score(y, y_pred))  # Accuracy score
    print("\nClassification Report:\n", classification_report(y, y_pred))  # Show main classification metrics
    print("\nCross Value Score:\n", cross_val_score(logi_model, X, y, cv = 10).mean())

    logit_roc_auc = roc_auc_score(y, logi_model.predict(X))
    fpr, tpr, thresholds = roc_curve(y, logi_model.predict_proba(X)[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontweight='bold', fontsize=13)
    plt.savefig('charts/ROC_Curve.png', dpi=None, facecolor='w', edgecolor='g', orientation='landscape',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.10, metadata=None)
    plt.show()


# Driver Code
if __name__ == '__main__':
    # Call main function
    main()
