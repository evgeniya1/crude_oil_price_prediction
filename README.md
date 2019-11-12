# Time Series Modeling: Crude Oil Price Prediction

## Project motivation

Crude oil is one of the most important and most traded commodity worldwide which makes crude oil price being considered as an indicator of global economy. Therefore, predicting the price in the future or at least getting an estimate becomes an important problem.

Forecasting the crude oil price is an extremely dificult task. Whether past price movements can be used to forecast the price or not is a subject of a debate. Here I will perform time series analysis using different available models, including facebook prophet, ARIMA, LSTM NN, aimed to forecast future priced based solely on previous price movements.

Last part of the project is devoted to simpler classification problem to identify the one week ahead trend, i.e. strong uptrend, not strong uptrend or downtrend (further refered as sideways), and strong downtrend.

## Dataset

In spite of the fact that oil market is global, there are different types of crude oil defined by the region it is extracted. Three primary benchmark oils are West Texas Intermediate (WTI), Brent Blend, and Dubai Crude. They have slightly different chemical composition that affects quality, ease of refinement, and hence, price. Here I will focus on WTI crude oil price, which is a high-quality oil that is easily and cheaper to refine. 

I will work with the dataset that I downloaded from https://datahub.io/ for daily and weekly frequency. For daily frequency this dataset contains 8501 data points for daily frequence and 1762 points for weekly frequency for time period from 01/02/1986 until 09/23/2019. The plot showing WTI crude oil price over specified period of time is given below.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/figs/original_data.png)

For regression problem the original price in USD is log-transformed and leveled/detrended, so that it lies within (-1,1) region, to avoid scaling issues for growing over time and large differences in magnitude.

## Forecast one month ahead using two years of data

### Facebook Prophet

First, I use recently developed by Facebook *prophet* package for time series modeling. It has three major components: non-periodic trend component (modeled as piecewise linear), periodic seasonal component (using Fourier series), user-provided  holiday component.

I performed slide-forward cross-validation (CV) using 2 years of data to train the prophet model (with trend, yearly and monthly seasonality components included). The animation below shows 100 CV intervals, where each interval is trained separately and error is computed on 1 month (22 business days) of test unseen by the model data. The mean average percentage error (MAPE) for the test data is 33.75 % ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 28.77 % for the transformed price and for 68 % confidence interval. While for transformed back to USD units price MAPE is noticeable lower and it is 12.26 % ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 9.03 %.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_fbprophet/y_smooth_w8_train_506_test_22_cv_100/FBprop_smooth_cv100_train_506_test_22.gif)

Comparison between actual price (black line) and prediction (red line) for 22 days ahead using 2 years of past data to train the model is shown below.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_fbprophet/y_smooth_w8_train_506_test_22_cv_100/compare.png)

### ARIMA

Next, I use Auto Regressive Integrated Moving Average (ARIMA) model from statsmodels package. Integrated part od the model is responsible for transforming time series into data with stationary characteristic by differencing (*d* times) the original time series. Next, the prediction of transformed data is approximated by *p* number of preceding periods/data points (auto regressive part) and *q* number of preceding error terms (moving average part). It has no seasonal component.

Similarly, I performed slide-forward CV using 2 years of data to train the ARIMA model (using found optimal parameters *p*=4, *d*=1, *q*=1). The animation below shows 100 CV intervals, where each interval is trained separately and error is computed on 22 days of test unseen by the model data. MAPE for the test data is 12.51 % ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 16.80 % for the transformed price (68 % confidence interval as well). While for transformed back to USD units price MAPE is 4.44 % ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 2.93 %. 

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_fbprophet/y_smooth_w8_train_506_test_22_cv_100/FBprop_smooth_cv100_train_506_test_22.gif)

Comparison between actual price (black line) and prediction (red line) for 22 days ahead using 2 years of past data to train the model is shown below.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_arima/y_smooth_w8_train_506_test_22_cv_100/compare.png)

Comparing *prophet* and ARIMA results, ARIMA gives noticeable better predictions with MAPE roughly three times lower that *prophet*. 
