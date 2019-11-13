# Time Series Modeling: Crude Oil Price Prediction

## Project motivation

Crude oil is one of the most important and most traded commodity worldwide which makes crude oil price being considered as an indicator of global economy. Therefore, predicting the price in the future or at least getting an estimate becomes an important problem.

Forecasting the crude oil price is an extremely dificult task. Good discussion about factors influencing the price can be found here https://www.investopedia.com/articles/economics/08/determining-oil-prices.asp. An interesting fact from this article is that the majority of trades are done by speculators. Moving to data modeling, whether past price movements can be used to forecast the price or not is a subject of a debate. Here I consider solely the previous price movements to make predictions for future behavior.

Here I will perform time series analysis using different available models, including facebook prophet, ARIMA, LSTM NN, aimed to forecast future prices. Last part of the project is devoted to simpler classification problem to identify the one week ahead trend, i.e. strong uptrend, not strong uptrend or downtrend (further refered as sideways) or strong downtrend.

## Dataset

In spite of the fact that oil market is global, there are different types of crude oil defined by the region it is extracted. Three primary benchmark oils are West Texas Intermediate (WTI), Brent Blend, and Dubai Crude. They have slightly different chemical composition that affects quality, ease of refinement, and hence, price. Here I will focus on WTI crude oil price, which is a high-quality oil that is easily and cheaper to refine. 

I will work with the dataset that I downloaded from https://datahub.io/ for daily and weekly frequency. For daily frequency this dataset contains 8501 data points for daily frequence and 1762 points for weekly frequency for time period from 01/02/1986 until 09/23/2019. The plot showing WTI crude oil price over specified period of time is given below.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/figs/original_data.png)

For regression problem the original price in USD is log-transformed and leveled/detrended, so that it lies within (-1,1) interval, to avoid scaling issues for growing over time and large differences in magnitude. After that the data is smoothed (using savgol_filter from scipy.signal package) to filted noise which leads to better prediction results. For instance, in case of ARIMA model discussed below mean absolute percentage error drops from 5.5 % (for raw data) to 4.4 % (smoothed data).

## Regression problem: forecast one month ahead using two years of data

### Facebook Prophet

First, I use recently developed by Facebook *prophet* package for time series modeling. It has three major components: non-periodic trend component (modeled as piecewise linear), periodic seasonal component (using Fourier series), user-provided  holiday component.

I performed slide-forward cross-validation (CV) using 2 years of data to train the prophet model (with trend, yearly and monthly seasonality components included). The animation below shows 100 CV intervals, where each interval is trained separately and error is computed on 1 month (22 business days) of test unseen by the model data. The mean absolute percentage error (MAPE) for the test data is 33.75 % ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 28.77 % for the transformed price and for 68 % confidence interval. While for the transformed back to USD units price MAPE is noticeable lower and it is 12.26 % ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 9.03 %.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_fbprophet/y_smooth_w8_train_506_test_22_cv_100/FBprop_smooth_cv100_train_506_test_22.gif)

Comparison between actual price (black line) and predicted (red line) for 22 days ahead using 2 years of past data to train the model is shown below.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_fbprophet/y_smooth_w8_train_506_test_22_cv_100/compare.png)

### ARIMA

Next, I use Auto Regressive Integrated Moving Average (ARIMA) model from statsmodels package. Integrated part of the model is responsible for transforming time series into data with stationary characteristics by differencing (*d* times) the original time series. Next, the prediction of transformed data is approximated linear combination of *p* number of preceding periods/data points (auto regressive part) and *q* number of preceding error terms (moving average part). It has no seasonal component.

Similarly, I performed slide-forward CV using 2 years of data to train the ARIMA model (using found optimal parameters *p*=4, *d*=1, *q*=1). The animation below shows 100 CV intervals, where each interval is trained separately and error is computed on 22 days of test unseen by the model data. MAPE for the test data is 12.51 % ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 16.80 % for the transformed price (68 % confidence interval as well). While for the transformed back to USD units price MAPE is 4.44 % ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 2.93 %. 

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_fbprophet/y_smooth_w8_train_506_test_22_cv_100/FBprop_smooth_cv100_train_506_test_22.gif)

Comparison between actual price (black line) and predicted (red line) for 22 days ahead using 2 years of past data to train the model is shown below.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_arima/y_smooth_w8_train_506_test_22_cv_100/compare.png)

Comparing *prophet* and ARIMA results, ARIMA gives noticeably better predictions with MAPE roughly three times lower that *prophet*. 

## Regression problem: forecast one week ahead

### ARIMA

Moving to the simpler problem: predicting 5 days ahead (i.e. 1 week ahead instead of 1 month), plot below shows comparison between the actual price (black line) and predicted by ARIMA model (red line) for 100 CV intervals.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/CV_arima/y_smooth_w8_train_506_test_5_cv_100/compare.png)

### LSTM

Long short-term memory (LSTM) is a special type of artificial recurrent neural network (RNN). In simple workds, RNN is a network with loop: it can be represented by multiple copies of the same network (e.g. applied to sliced chunks of a given time series) where information is passed from each one NN to the next one in a sequence. In case of LSTM NN, the passed information goes through compicated system of filters. See this post for detailed description: https://colah.github.io/posts/2015-08-Understanding-LSTMs/.  

Here I will focus on the simpler problem to predict 5 days ahead. I explore two different approaches, shown schematically below: 
- 1 LSTM model to predict all 5 days ahead
- 5 different LSTM models to predict one day ahead for each day out of 5 (model stacking)

<img src="https://github.com/evgeniya1/Flatiron_final_project/blob/master/figs/LSTM_models.png" width="337" height="284">

#### Hyperparameter Tuning
- input window size: optimal values are input = 25 days for 1 LSTM, input = [29,20,17,18,14] days for 5 LSTM models. Not that in case of 5 LSTM models input size decreases with the gap between last input point and target point/day, i.e. less short term information is needed, which is an interesting finding.
- hidden layer size: optimal values is 2

Comparison between actual price (black dots) and predicted (red lines) for 5 days ahead using optimal input window size to train the models is shown below for both 1 LSTM and 5 LSTM models for 50 intervals. 1 LSTM model has MAPE = 1.98 ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 1.95 while 5 LSTM models has slightly better result MAPE = 1.91 ![equation](https://latex.codecogs.com/gif.latex?$\pm$) 1.91 for the test data (here train-test split is 80%-20%).

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/figs/regression_5vs1.png)
From the figure above, it can be noticed that 1 LSTM model gives flatter predictions than 5 LSMT which camptures better the variarion in predicted price for 5 days. To demonstrate this, graphs below show the distribution of average slope (within predicted 5 days) for both models. 1 LSTM model clearly captures the variation in slope better.

![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/figs/regression_slope_hist_1LSTM.png)
![](https://github.com/evgeniya1/Flatiron_final_project/blob/master/figs/regression_slope_hist.png)

These LSTM models give reasonable predictions, accounting the noisiness and randomness of a given problem. However, how to achieve more reliable results? To do so, let me consider even simplier classification problem aimed to predict strong uptrend, sideways, or strong downtrend.

## Classification problem: forecast trend one week ahead

Here to predict price one week ahead I use dataset with dataset weekly frequency, i.e. predict only one point ahead. To move to classification problem first the target needs to be engineered.
