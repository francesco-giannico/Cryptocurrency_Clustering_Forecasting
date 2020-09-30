
# Cryptocurrency clustering and forecasting

This repo contains the code for my M.S. Thesis in Big Data Analytics, discussed in _June 17, 2020_ at the University of Bari "Aldo Moro". 
It is entitled: *Clustering-based multi-target forecasting for the cryptocurrency financial market*

## Goals
Hypothesis: The correlation between cryptocurrencies has latent information useful in order to predict their trends for the next day.

# Involved cryptocurrencies
Bitcoin, BitShares, Dash, DigiByte, DogeCoin, Ethereum, IOCoin, Litecoin, MaidSafeCoin, MonaCoin, NavCoin, Syscoin, Vertcoin, Counterparty, Stellar, Monero, Ripple

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

## DAta preparation: Selection
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

## Feature engineering

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

