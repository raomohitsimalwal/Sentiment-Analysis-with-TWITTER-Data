# Sentiment Analysis of Twitter Data

A robust Python application for conducting sentiment analysis on Twitter data using the Natural Language Toolkit (NLTK) and generating insightful visualizations with Matplotlib and Seaborn.

## Project Overview
This application processes Twitter data from a CSV file to perform sentiment analysis using NLTK's VADER sentiment analyzer. It provides a comprehensive pipeline for data cleaning, text preprocessing, sentiment scoring, and visualization of sentiment distributions. Designed for data scientists, market researchers, and social media analysts, it facilitates the extraction of public sentiment insights from Twitter.

## Key Features
- **Data Import**: Seamlessly loads Twitter data into a structured Pandas DataFrame.
- **Data Validation**: Identifies and handles missing or incomplete data entries.
- **Advanced Text Preprocessing**: Removes noise (URLs, mentions, hashtags) and applies tokenization and stopword removal.
- **Sentiment Analysis**: Computes compound sentiment scores using the VADER model for nuanced sentiment detection.
- **Data Visualization**: Produces high-quality histograms with kernel density estimation for sentiment distribution analysis.
- **Customizable Pipeline**: Easily adaptable for other text-based datasets or visualization preferences.

## System Requirements
- **Python 3.8 or higher
- **Required Python Libraries**:
  - `pandas` for data manipulation
  - `nltk` for natural language processing
  - `matplotlib` for plotting
  - `seaborn` for enhanced visualizations
- **Operating System**: Windows, macOS, or Linux

## Installation Guide
1. **Install Python Libraries**:
   ```bash
   pip install pandas nltk matplotlib seaborn
   ```

2. **Download NLTK Resources**:
   Execute the following in a Python environment:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Usage Instructions
1. **Prepare Input Data**:
   Ensure your Twitter dataset is in CSV format with at least the columns: `ID`, `Brand`, `Sentiment`, and `Text`. Example structure:
   ```csv
   ID,Brand,Sentiment,Text
   1,BrandX,Positive,"Great product! #loveit"
   ```

2. **Configure the Script**:
   Modify the `file_path` variable in `sentiment_analysis.py` to point to your CSV file:
   ```python
   file_path = "path/to/twitter_training.csv"
   ```

3. **Run the Application**:
   Execute the script via the command line:
   ```bash
   python sentiment_analysis.py
   ```

## Technical Workflow
The application follows a structured pipeline:

1. **Data Loading**:
   Imports the CSV file into a Pandas DataFrame, assigning column names (`ID`, `Brand`, `Sentiment`, `Text`).

2. **Data Inspection and Cleaning**:
   - Displays initial DataFrame rows and null value counts.
   - Removes rows with missing `Text` values to ensure data quality.

3. **Text Preprocessing**:
   Applies a `preprocess_text` function to:
   - Remove URLs, mentions, hashtags, retweet tags, and non-alphabetic characters.
   - Tokenize text and filter out stop words.
   - Store results in a `cleaned_text` column.

4. **Sentiment Analysis**:
   Utilizes NLTK's VADER SentimentIntensityAnalyzer to compute compound sentiment scores, stored in a `sentiment_score` column.

5. **Visualization**:
   Generates a histogram of sentiment scores using Seaborn, enhanced with a kernel density estimate (KDE) for distribution clarity.


- **Example Visualization**:
![Example Image](https://github.com/sugin22/Sentiment-Analysis-with-Twitter-Data/blob/main/Figure_1.png)

![Example Image](https://github.com/sugin22/Sentiment-Analysis-with-Twitter-Data/blob/main/Figure_2.png)

![Example Image](https://github.com/sugin22/Sentiment-Analysis-with-Twitter-Data/blob/main/Figure_3.png)
