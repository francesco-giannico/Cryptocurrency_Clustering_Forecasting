import pandas as pd
import re
from io import StringIO
from datetime import datetime, timedelta
import requests
import os
from utility.folder_creator import folder_creator

CURRDIR = os.path.dirname(os.path.realpath(__file__))
TIMEOUT = 1
CRUMB_LINK = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
CRUMBLE_REGEX = r'CrumbStore":{"crumb":"(.*?)"}'
QUOTE_LINK = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'

def yahoo_finance_history(symbol):
    return get_quote(symbol,requests.Session())

def get_crumb(symbol,session):
    response = session.get(CRUMB_LINK.format(symbol), timeout=TIMEOUT)
    response.raise_for_status()
    match = re.search(CRUMBLE_REGEX, response.text)
    if not match:
        raise ValueError('Could not get crumb from Yahoo Finance')
    else:
       crumb = match.group(1)
    return crumb

def get_quote(symbol,session):
    #if not hasattr(self, 'crumb') or len(session.cookies) == 0:
    crumb=get_crumb(symbol,session)
    enddate = datetime(2020,1,1) #max 31/12/2019
    startdate =datetime(2010,1,2)
    dateto = int(enddate.timestamp())
    datefrom=int(startdate.timestamp())
    url = QUOTE_LINK.format(quote=symbol, dfrom=datefrom, dto=dateto, crumb=crumb)
    response = session.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), parse_dates=['Date'])

def get_all_crypto():
    dataset_name="\cryptocurrencies"
    dataset_dir=CURRDIR +dataset_name
    folder_creator(dataset_dir,0)
    currency = "-USD"
    f = open(CURRDIR +"\crypto.txt", "r")
    cryptos = f.readlines()
    for crypto in cryptos:
        crypto = crypto.replace("\n", "")
        print("getting info about "+ crypto)
        df = yahoo_finance_history(crypto + currency)
        df.to_csv(dataset_dir+"/"+crypto + ".csv", index=False)
