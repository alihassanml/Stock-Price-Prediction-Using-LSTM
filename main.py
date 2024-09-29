import streamlit as st
import pandas as pd
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()

st.title('Describe Data')
df = pd.DataFrame(pd.read_csv('BTC-Data.csv'))
st.write(df.describe())

st.subheader('Volume Of Data')
fig = df.iplot(kind='bar',x='Date',y='Volume',asFigure=True)
 # Create the figure
st.plotly_chart(fig)
