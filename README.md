
# Cryptocurrency clustering and forecasting

This repo contains the code for my M.S. Thesis in Big Data Analytics, discussed in _June 17, 2020_ at the University of Bari "Aldo Moro". 
It is entitled: *Clustering-based multi-target forecasting for the cryptocurrency financial market*

Duration: Jan, 2019 - Jun, 2020 (6 months)

## Goals
Hypothesis: The correlation between cryptocurrencies has latent information useful in order to predict their trends for the next day.

## Steps
The trend can be:
* STABLE
* UP
* DOWN

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

It has been decided to create two different version of the labelled dataset:
* X= 1%
* X= 2%
### Feature scaling
#### Min-Max scaling (for clustering purposes)
This technique is widely adopted when the distribution of data is not gaussian or, in other cases, when the data distribution is unknown. In min-max scaling data are scaled in a fixed range, usually between 0 and 1. 

#### Max-Abs scaling (for forecasting purposes)
The Max-abs scaling is very similar to min-max scaling but for the range, in this case, between -1 and 1. It does not shift/centre the data, and thus does not destroy any sparsity. This scaler is meant for data that is already centred at zero or sparse data.

## Deep Learning technique 
**Long short-term memory (LSTM)** which is a widely adopeted technique for time series forecasting.


## Clustering techniques

Consensus clustering which involved k-medoids and agglomerative algorithms.


## Main Tools
* Python
* Scikit-learn
* Google Colab
* Keras & Tensorflow 

## Author
* Giannico Francesco

## Collaborators
* Gianvito Pio
* Michelangelo ceci

