# Time Series Modeling: Crude Oil Price Prediction

## Project motivation

Crude oil is one of the most important and most traded commodity worldwide which makes crude oil price being considered as an indicator of global economy. Therefore, predicting the price in the future or at least getting an estimate becomes an important problem.

Forecasting the crude oil price is an extremely dificult task. Whether past price movements can be used to forecast the price or not is a subject of a debate. Here I will perform time series analysis using different available models, including facebook prophet, ARIMA, LSTM NN, aimed to forecast future priced based solely on previous price movements.

Last part of the project is devoted to simpler classification problem to identify the one week ahead trend, i.e. strong uptrend, not strong uptrend or downtrend (further refered as sideways), and strong downtrend.

## Dataset

In spite of the fact that oil market is global, there are different types of crude oil defined by the region it is extracted. Three primary benchmark oils are West Texas Intermediate (WTI), Brent Blend, and Dubai Crude. They have slightly different chemical composition that affects quality, ease of refinement, and hence, price. Here I will focus on WTI crude oil price, which is a high-quality oil that is easily and cheaper to refine. 

I will work with the dataset theat I downloaded from https://datahub.io/ for daily and weekly frequency. 
