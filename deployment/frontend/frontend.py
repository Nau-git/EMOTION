from sklearn.metrics import adjusted_rand_score
import streamlit as st
import numpy as np
import requests
import pickle
import mplfinance as mpf
import plotly.graph_objects as go
import pandas_datareader as web
from datetime import date, timedelta
 
st.set_page_config(
    page_title='Emotion - Final Project',
    page_icon="📈",
    initial_sidebar_state="collapsed",
    layout='centered',
    menu_items={
        'Get Help': 'https://www.google.com',
        'Report a bug': "https://github.com/anugrahyogaprt",
        'About': 'This is our final project on FTDS Batch 10 Hactiv8.\n\
         Our Team: Anugrah Yoga Pratama, M Naufal Indriatmoko, and Tandya Anggergian'
    }
)
# ----------------------------------------------------------------------
st.title('EMOTION - Economic Market Outlook Prediction')
st.markdown('"Emotion" is an application that can be useful \n\
        for predicting the closing price of a major financial instrument \n\
        in the world such as Gold, US dollar index, S&P 500, \n\
        crude oil, and also the stock index in Indonesia, \n\
        called the Jakarta Composite Index.')

col1, col2, col3 = st.columns([0.5, 5, 0.5])
col2.image('finance.webp', use_column_width=True, caption='Financial Growth', width=500)
image_citation = '''
<p
        style="text-align: center;">
        Image source: fainstitute.com
</p>
'''
st.markdown(image_citation, unsafe_allow_html=True)

st.markdown('Each of the market instruments has a financial relationship \n\
        with each other and is also an important indicator of the economic situation \n\
        and events that occur at a certain time. The world economic situation will \n\
        also affect the economy in Indonesia directly or indirectly.')
st.markdown("There are 5 financial instruments available to be predicted. \n\
        Please choose one. EMOTION will predict its tomorrow's closing price. \n\
        You can also predict the market situation by imputing a news headline \n\
        in the box below.")

# ----------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
        st.subheader('Financial Instrument')
        input_form = st.selectbox(
                label='Please select your financial symbol',
                options=['Gold', 'US Dollar Index', 'S&P 500', 'Crude Oil', 'Jakarta Composite Index'],
                index=0
        )
with col2:
        st.subheader('Headline News')
        userInput = st.text_input('Please input the headline news')

# ----------------------------------------------------------------------
symbols = {
        'Gold': 'GC=F', 
        'US Dollar Index': 'DX-Y.NYB', 
        'S&P 500': '^GSPC', 
        'Crude Oil': 'CL=F', 
        'Jakarta Composite Index': '^JKSE'
}
# ----------------------------------------------------------------------
today = date.today()
quote = symbols[input_form]
n_days = 40
# ----------------------------------------------------------------------
new_df = web.DataReader(quote,
                data_source='yahoo',
                start='2002-01-01',
                end=today)

# ----------------------------------------------------------------------
days = [7, 30, 182, 365, 796, 1826]
_1week, _1month, _6month, _1year, _3year, _5year = [new_df[new_df.index < str(today - timedelta(days=n))]['Close'][-1] for n in days]
# ----------------------------------------------------------------------
scaler_name = {
        'Gold': 'gold_scaler.pkl',
        'US Dollar Index': 'dxy_scaler.pkl',
        'S&P 500': 'gspc_scaler.pkl',
        'Crude Oil': 'oil_scaler.pkl',
        'Jakarta Composite Index': 'jkse_scaler.pkl'
}

with open('scaler/' + scaler_name[input_form], 'rb') as f:
        scaler = pickle.load(f)
# ----------------------------------------------------------------------
close_scaled = scaler.transform(np.array(new_df.Close).reshape(-1, 1))
# ----------------------------------------------------------------------
data_inf = {
    'symbol': quote,
    'n_close': close_scaled.tolist()[-n_days:],
    'n_days': [n_days],
    'teks': userInput
}
# ----------------------------------------------------------------------
df = new_df.copy()
df.sort_index(inplace=True)

trace_line = go.Scatter(x=list(df.index),
                                y=list(df.Close),
                                #visible=False,
                                name="Close",
                                showlegend=False)

trace_candle = go.Candlestick(x=df.index,
                        open=df.Open,
                        high=df.High,
                        low=df.Low,
                        close=df.Close,
                        #increasing=dict(line=dict(color="#00ff00")),
                        #decreasing=dict(line=dict(color="white")),
                        visible=False,
                        showlegend=False)

trace_bar = go.Ohlc(x=df.index,
                        open=df.Open,
                        high=df.High,
                        low=df.Low,
                        close=df.Close,
                        #increasing=dict(line=dict(color="#888888")),
                        #decreasing=dict(line=dict(color="#888888")),
                        visible=False,
                        showlegend=False)

data = [trace_line, trace_candle, trace_bar]

updatemenus = list([
        dict(
                buttons=list([
                dict(
                        args=[{'visible': [True, False, False]}],
                        label='Line',
                        method='update'
                ),
                dict(
                        args=[{'visible': [False, True, False]}],
                        label='Candle',
                        method='update'
                ),
                dict(
                        args=[{'visible': [False, False, True]}],
                        label='Bar',
                        method='update'
                ),
                ]),
                direction='down',
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.02,
                xanchor='left',
                y=1,
                yanchor='top'
        ),
])

layout = dict(
        updatemenus=updatemenus,
        xaxis=dict(
                rangeselector=dict(
                        buttons=list([
                                dict(count=1,
                                label='1m',
                                step='month',
                                stepmode='backward'),
                                dict(count=3,
                                label='3m',
                                step='month',
                                stepmode='backward'),
                                dict(count=6,
                                label='6m',
                                step='month',
                                stepmode='backward'),
                                dict(count=1,
                                label='YTD',
                                step='year',
                                stepmode='todate'),
                                dict(count=1,
                                label='1y',
                                step='year',
                                stepmode='backward'),
                                dict(count=3,
                                label='3y',
                                step='year',
                                stepmode='backward'),
                                dict(count=5,
                                label='5y',
                                step='year',
                                stepmode='backward'),
                                dict(step='all')
                        ])
                ),
                rangeslider=dict(
                        visible = True
                
                ),
                type='date'
        ),
        yaxis=dict(
                autorange = True,
                fixedrange= False
        )
)

fig = go.FigureWidget(
        data=data, 
        layout=layout
)

fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=20),
        paper_bgcolor="#FFFFFF",
)

st.plotly_chart(fig, use_container_width=True)
# ----------------------------------------------------------------------
# komunikasi
URL = 'https://backend-finalprojek.herokuapp.com/price'
r = requests.post(URL, json=data_inf)
if st.button('Prediction'):
        st.markdown("---")
        if isinstance(r.json()['prediction'], list):
                # Price Prediction
                col1, col2 = st.columns(2)
                with col1:
                        st.subheader("Price Prediction")
                        #Get last price data in scaled
                        last_price_scaled = np.array(data_inf['n_close'])
                        last_price = scaler.inverse_transform(last_price_scaled)
                        
                        #Get last and yesterday price in list
                        prev_value = last_price.tolist()[-2][0]
                        last_value = last_price.tolist()[-1][0]
                        unit = 'IDR' if quote=='^JKSE' else 'USD'

                        #Get the prediction and convert result
                        result = scaler.inverse_transform(np.array(r.json()['prediction']))
                        result = result[0, 0]
                        result_gain = result/last_value - 1
                        st.metric(
                                label="Next Day Prediction", 
                                value=f"{result:.2f} {unit}", 
                                delta=f"{100*result_gain:.2f}%"
                        )

                        #Show gain of last vs yesterday price
                        last_gain = last_value/prev_value - 1
                        st.metric(
                                label="Last Price", 
                                value=f"{last_value:.2f} {unit}", 
                                delta=f"{100*last_gain:.2f}%"
                        )
                        
                with col2:
                        # Headline Prediction
                        st.subheader("Headline Sentiment")
                        label = r.json()['nlp_pred'][0]
                        sentiment = 'negative' if label==0 \
                                else 'neutral' if label==1 \
                                        else 'positive' if label==2 \
                                                else 'error'                        
                        st.image(
                                f'{sentiment}.png', 
                                caption=f'It is a {sentiment} sentiment', 
                                width=200
                        )
                        

                st.markdown("---")
                # fig = mpf.figure(style='yahoo', figsize=(16,9))
                fig, ax = mpf.plot(
                        new_df.tail(55),
                        type='candle',
                        style='yahoo',
                        title=f'{input_form}',
                        ylabel=f"Price ({'Rp' if quote=='^JKSE' else '$'})",
                        figratio=(16, 9),
                        returnfig=True
                )
                st.pyplot(fig)
                st.markdown("---")

                st.subheader(f'{input_form} Performance')
                col1, col2, col3 = st.columns(3)
                with col1:
                        _1week_gain = last_value/_1week - 1
                        st.metric(
                                label="1 Week", 
                                value=f"{_1week:.2f} {unit}", 
                                delta=f"{100*_1week_gain:.2f}%"
                        )
                        _1year_gain = last_value/_1year - 1
                        st.metric(
                                label="1 Year", 
                                value=f"{_1year:.2f} {unit}", 
                                delta=f"{100*_1year_gain:.2f}%"
                        )
                with col2:
                        _1month_gain = last_value/_1month - 1
                        st.metric(
                                label="1 Month", 
                                value=f"{_1month:.2f} {unit}", 
                                delta=f"{100*_1month_gain:.2f}%"
                        )
                        _3year_gain = last_value/_3year - 1
                        st.metric(
                                label="3 Year", 
                                value=f"{_3year:.2f} {unit}", 
                                delta=f"{100*_3year_gain:.2f}%"
                        )
                with col3:
                        _6month_gain = last_value/_6month - 1
                        st.metric(
                                label="6 Month", 
                                value=f"{_6month:.2f} {unit}", 
                                delta=f"{100*_6month_gain:.2f}%"
                        )
                        _5year_gain = last_value/_5year - 1
                        st.metric(
                                label="5 Year", 
                                value=f"{_5year:.2f} {unit}", 
                                delta=f"{100*_5year_gain:.2f}%"
                        )
                st.markdown("---")
        else:
                print('Error! result is not list')
else:
        st.write('Click to Predict')
