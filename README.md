
# Clustering-based multi-target forecasting for the cryptocurrency financial market

This repo contains the code for my M.S. Thesis in Big Data Analytics, discussed in _June 17, 2020_ at the University of Bari "Aldo Moro". 

Duration: Jan, 2020 - Jun, 2020 (6 months)

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)

## Introduction 

## Technologies
Project is created with:
* Scikit-learn version: 0.23.1
* Tensorflow version: 2.1

## Goals

Hypothesis to verify: *The correlation between cryptocurrencies has latent information useful in order to predict their trends for the next day*.
Overall Actions: 
1. Clustering
    - Consensus clustering based on:
      - K-means 
      - Agglomerative

2. Modelling 
   - Baseline
   - Single-target Model based on LSTM
   - Multi-target Model based on LSTM
Result: In overall, for some cluster of cryptocurrencies, the hyposesis is true.

# UNCOMPLETED DESCRIPTION..
## Dataset 

All available data on the Yahoo Finance platform have been downloaded, for each cryptocurrency, about price and volume.
Since Yahoo Finance does not offer APIs, web scarping was necessary. It automatically allowed to collect information from the website, by parsing content from within the HTML container of each specific page. For example, a web page containing historical date for bitcoin can be found
[here](https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD).

### Dataset's details
Each dataset has inter-day granularity and is composed by the following features:
* Open, it is the opening price, in US dollars;
*  High, t it is the maximum price, in US dollars;
*  Low, it is the minimum price, in US dollars;
*  Close, it is the closing price, in US dollars;
*  Adjusted close4(Adj Close), it is the closing price after adjustments for all applicable splits and dividend distributions;
*  Volume, it is the amount of money moved in a certain period, usually 24 hours.
The file format of each of 100 datasets is the *comma-separated values (csv)*.

## Data preparation

### Selection
It has been selected 17 out of 100 cryptocurrencies for this study, the oldest ones, over the period 2015-09-01 to 2019-12-31. The total number of entries is 1584, for each dataset.

The selected cryptocurrencies are:
1. Bitcoin (BTC);
2. BitShares (BTS);
3. Dash (DASH);
4. DigiByte (DGB);
5. DogeCoin (DOGE);
6. Ethereum (ETH);
7. IOCoin (IOC);
8. Litecoin (LTC);
9. MaidSafeCoin (MAID);
10. MonaCoin (MONA);
11. NavCoin (NAV);
12. Syscoin (SYS);
13. Vertcoin (VTC);
14. Counterparty (XCP);
15. Stellar (XLM);
16. Monero (XMR);
17. Ripple (XRP).

### Feature engineering
As the rows in the dataset only contains information about a specific period (i.e. one day), these are not sufficient to generate good predictions. Thus, to get better predictions, it has been decided to add more features in each dataset, computed on the base of the feature **Close** and **Volume**.
These new features are the *technical indicators* involved in the decisional support process of a trader. Their value is used for the technical analysis of the price trends and, thus, for the prediction of the price trends of the future. To accomplish this step, the library exploited is **pandas_ta5**, which is an easy to use library that is built upon Python's Pandas library with more than 80 technical indicators.

#### Features
* Simple Moving Average (SMA)
* Exponential Moving Average (EMA)
* Moving Average Convergence-Divergence (MACD)
* Relative Strenght Index (RSI)
* Ultimate Oscillator (UO)
* Bollinger Bands (BBANDS)
* Volume-weighted Average Price (VWAP)

## Labelled datasets construction
As the main goal is to **classify** the trend of each day over a period, it was necessary to create a *labelled version* of the datasets. To accomplish this need, it has been decided to compute the delta difference in percentage, between the value of the feature Close at current day and its value at the previous day.
Formula: *delta percent=((Close_current_day−Close_day_before) / (Close_day_before)*100)

The next step is to set a percentage threshold, identified by the variable X, and compare the delta percent in respect to X, to decide which label assign at a certain day:
* If delta percent is upper than +X then assign **Up**;
* If delta percent is lower than -X then assign **Down**;
* If delta percent is between -X and +X then assign **Stable**;

It has been decided to create two different version of the labelled datasets:
* X= 1%
* X= 2%

### Feature scaling
#### Min-Max scaling (for clustering purposes)
This technique is widely adopted when the distribution of data is not gaussian or, in other cases, when the data distribution is unknown. In min-max scaling data are scaled in a fixed range, usually between 0 and 1. 

#### Max-Abs scaling (for forecasting purposes)
The Max-abs scaling is very similar to min-max scaling but for the range, in this case, between -1 and 1. It does not shift/centre the data, and thus does not destroy any sparsity. This scaler is meant for data that is already centred at zero or sparse data.

## Modelling 

### Forecasting
#### Baseline
A simple predictive model that estimates the trend for the next day to be equal to the trend of the previous one.

### Artificial Neural Network

**Long short-term memory (LSTM)** which is a widely adopeted technique for time series forecasting.


## Clustering techniques

Consensus clustering which involved k-medoids and agglomerative algorithms.

## Experiments 
### Walk-Forward validation

#### Expermimental Settings
* Python
* Google Colab

#### Test Set

#### LSTM's Hyperparameters

#### Clustering
* SqrtNdiv4 = 2
* sqrtNdiv2 = 3
* sqrtN = 4
* sqrtNby2 = 6
* sqrtNby4 = 8

### Model Performance
Since the micro average, weighted macro and the accuracy are not suitable for
imbalanced datasets, only the **macro average recall** has been considered. 
Python's scikit-learn library provied a convenient functions in order to compute this value. See 
[Confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) and [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

While it is easy to measure the performance of supervised learning algorithms, such as algorithms for classification problems, it is often hard to measure the performance of unsupervised learning algorithms, such as clustering algorithms. The reason for this, is that it is subjective what makes a clustering ‘good’. The performance of a clustering depends on the goal and criteria of the clustering and may therefore differ per application.
This goal is to use clustering to obtain clusters that can be used for forecasting. Instead of making forecasts for all individual time series, only one forecast will be made for each cluster. This forecast is then applied to all time series in the corresponding cluster. To measure the quality of the clustering, it has been taken the macro average recall as performance metric.

## Author
**Francesco Giannico**, M.S. in Computer Science, specialized in Knowledge Engineering and Machine intelligence (LM-18)
* [github/francesco-giannico](https://github.com/francesco-giannico)
* [linkedin/francesco-giannico](https://linkedin.com/in/francesco-giannico)

## Collaborators
* [Gianvito Pio](http://www.di.uniba.it/~gianvitopio/), Assistant Professor at [University of Bari "Aldo Moro"](https://www.uniba.it/)
* [Michelangelo ceci](http://www.di.uniba.it/~ceci/), Full Professor at [University of Bari "Aldo Moro"](https://www.uniba.it/)




