Sentiment Analysis of Product Reviews in R

This project demonstrates a simple sentiment analysis workflow in R using the tidytext, dplyr, and ggplot2 packages.

The program analyzes sample product and service reviews, identifies positive and negative words using the Bing sentiment lexicon, counts the sentiment words, and visualizes the results.

📌 Project Overview

The project performs the following steps:

Creates a sample dataset containing product and service reviews.

Converts the review text into individual words.

Uses the Bing sentiment lexicon to identify positive and negative words.

Counts the number of positive and negative sentiment words.

Creates a bar chart to visualize the sentiment distribution.

🛠️ Technologies Used

R

tidytext – Text mining and sentiment analysis

dplyr – Data manipulation

ggplot2 – Data visualization

📦 Installation

Install the required packages using:

install.packages("tidytext")
install.packages("dplyr")
install.packages("ggplot2")


Then load the libraries:

library(tidytext)
library(dplyr)
library(ggplot2)

📊 Sample Dataset

The project uses six sample reviews:

ID	Review
1	I love this product. It is excellent and amazing.
2	The service was very good and helpful.
3	I am happy with the quality.
4	The product is bad and disappointing.
5	I hate this service. It is terrible.
6	The experience was poor and frustrating.
🔍 How It Works
1. Tokenization

The reviews are converted into individual words using unnest_tokens():

words <- reviews %>%
  unnest_tokens(word, text)

2. Sentiment Detection

The project uses the Bing sentiment lexicon provided by tidytext:

sentiments <- words %>%
  inner_join(get_sentiments("bing"), by = "word")


Words are classified as either:

positive

negative

For example, words such as love, excellent, and amazing are identified as positive, while words such as bad, hate, and terrible are identified as negative.

3. Sentiment Counting

The number of positive and negative words is calculated using:

sentiment_count <- sentiments %>%
  count(sentiment)

4. Visualization

A bar chart is created using ggplot2:

ggplot(sentiment_count, aes(x = sentiment, y = n)) +
  geom_col() +
  labs(
    title = "Sentiment Analysis",
    x = "Sentiment",
    y = "Number of Words"
  )

📈 Expected Output

The program produces:

The original review dataset

A tokenized list of individual words

Words identified as positive or negative

A count of positive and negative sentiment words

A bar chart showing the sentiment distribution

📁 Suggested Repository Structure
sentiment-analysis-r/
│
├── sentiment_analysis.R
├── README.md
└── images/
    └── sentiment_plot.png

🚀 How to Run

Clone or download this repository.

Open sentiment_analysis.R in RStudio.

Install the required packages if they are not already installed.

Run the script.

View the sentiment counts and generated visualization.

🎯 Learning Objectives

This project is useful for learning:

Basic text mining in R

Tokenization

Sentiment analysis

Using sentiment lexicons

Data manipulation with dplyr

Data visualization with ggplot2

🔮 Possible Improvements

The project can be extended by:

Using a larger real-world review dataset

Calculating sentiment for each individual review

Adding neutral sentiment

Comparing multiple sentiment lexicons

Creating word clouds

Removing stop words

Calculating sentiment scores

Building an interactive sentiment dashboard

📄 License

This project is intended for educational and learning purposes.
