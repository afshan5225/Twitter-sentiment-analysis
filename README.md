# Twitter Sentiment Analysis Mini Project

## Overview
This project involves analyzing Twitter data to classify sentiments expressed in tweets. The goal is to preprocess the data, apply various text processing techniques, and evaluate different machine learning models to determine their effectiveness in sentiment classification.

## Project Details

### Data
- **Source**: [Google Colab Notebook](https://colab.research.google.com/drive/1HlweEoB0CWvKMoVmuU8zy992ywrpG-Q6)
- **File**: `twitter_validation.csv`
- **Columns**:
  - `ID`: Unique identifier for each tweet
  - `Location`: Location of the tweet's author
  - `target`: Sentiment label (Positive, Negative, Neutral)
  - `text`: Tweet content

### Methodology

1. **Data Loading**:
   - Mounted Google Drive to access the CSV file.
   - Loaded the dataset into a Pandas DataFrame.

2. **Data Exploration**:
   - Displayed the first and last few records.
   - Analyzed missing values, data types, and value counts for various columns.
   - Visualized the distribution of sentiments and locations using Seaborn and Matplotlib.

3. **Data Preprocessing**:
   - Removed irrelevant rows and reset the DataFrame index.
   - Dropped unnecessary columns (`Location`, `ID`).
   - Converted sentiment labels to numerical values (`Positive` -> 1, `Negative` -> -1, `Neutral` -> 0).
   - Tokenized tweets and performed text cleaning (removing non-alphanumeric characters).
   - Applied stemming and removed stopwords to normalize text.

4. **Feature Extraction**:
   - Used `TfidfVectorizer` to convert cleaned text into numerical features.

5. **Model Training and Evaluation**:
   - Split the data into training and test sets.
   - Trained and evaluated several machine learning models:
     - Support Vector Machine (SVM)
     - Naive Bayes (MultinomialNB)
     - K-Nearest Neighbors (KNN)
     - Random Forest Classifier
     - Decision Tree Classifier
   - Used classification metrics such as confusion matrix and classification report to assess model performance.

### Results
The project provides a comparative analysis of different classifiers for sentiment analysis on Twitter data. Performance metrics for each model are detailed in the notebook.

## Dependencies
- pandas
- matplotlib
- seaborn
- nltk
- scikit-learn

## Usage
1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the notebook `miniprojectZAINAMAM.ipynb` to replicate the analysis.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
