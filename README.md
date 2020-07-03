# Groupon Q42013 North American(NA) Total Gross Billing Estimation
Based on the historical data, estimate the total North American gross billing of Groupon in Q4 2013 and give suggestions on buy or sell options for Groupon's stock in the next quarter.

Q4 2013 Groupon North America Gross Billing Estimate (in $) by Segment

	Local: 445028140.83 ($445.03 million)
	Travel: 70547395.725 ($70.5 million)
	Goods: 282245469.101 ($282 million)
	Total: 797821005.65 ($797 million)
  
Approaches

	Time series analysis (ARIMA)
	Machine learning (random forest, gradient boosting and decision tree)
  
Project Platform

	Python

Data Quality

	Irregularities
  
	There are extremely small values (<1e-10) in units sold and billings, set them equal to 0. 
	Some deals have small values in the units sold but at the same time have a large value in billings. These are errors in the data, so I remove those rows from the dataset.
	There are positive large numbers (>=mean+3*standard deviation) both in daily other gross billing and daily other units sold. (I will use other gross billing to represent the sum of travel gross billing and goods gross billing in this report). These large values appear from mid-November to mid-December. However, they are not being treated as outliers in this assignment, because they are not wrong data, they represent the real gross billing and units sold. I decided not to treat them as outliers because they are valid data points. I will explain more about these large values in the patterns section.
  
	Checked Without Issue
  
	There are time gaps larger than 10 days without data recorded in the dataset. For example, no data were recorded from 11/22/2011 to 6/62012. I assume there is no other missing value in the original dataset except the 10/20/2013 to 10/30/2013 local data.
	No other segment level in the dataset rather than local, travel and goods. No other levels in inventory type rather than Third-Party and First-Party
	No duplicated deal id and deal URL in the dataset.
	No deal has negative units sold and positive gross billings or positive units sold and negative gross billings.
	There are no extremely small gross billings but with many units sold.
	There are no wrong data on start date such as 6/31/2012.
  
Exploratory Analysis, Patterns, and Adjustments

	There is a sharp growth in total daily gross billing (in this report, total daily gross billing represents local plus travel plus goods gross billing) around deals that start from September 22nd, 2013. The growth not only includes the average amount of the gross billing but also the amplitude of the variance. The increase in gross billing represents that more customers ordered services or products from those deals that start after September 22. This makes sense because the popularity of deals will decrease over time and we are only focusing on the gross billings in the fourth quarter of 2013. Therefore, as the deals’ start dates get close to October 2013, the units sold and the gross billing increase too. Before fitting an ARIMA model to estimate the missing values, I first made a log transformation to the original dataset to remove the increased amplitudes in the variance. And then I took the first difference of the transformed data to remove the trend and used the Dickey-Fuller test to prove that the series is a stationary time series.
	There is a peak in other gross billing and other units sold stand out on December 2nd, 2013. December 2nd, 2013 is Cyber Monday, Groupon may release a lot of great deals and attract many customers to order from their website. Also, the increasing trend from mid-November to early December may cause by the festival season (Thanksgiving and Christmas). Thanksgiving and Cyber Monday brought a lot of traffic to Groupon’s website. Customers start to plan their holiday travel or want to buy festival staff for their families which leads to an increase in travel gross billing and goods gross billing. After December 2nd, as Christmas gets closer and great deals’ time limit is up, fewer and fewer customers order services or goods from Groupon’s website. The negative trend shows up, so the total gross billing and total units sold start to decrease until the end of December. However, when machine learning was used to predict the missing values, those high daily gross billing values were not treated as outliers, because they are valid data points and can be used to train the model.
	Base on the time series data plot that was aggregated monthly and daily, there is a higher correlation between local gross billing and other gross billing, rather than the correlation between local and travel or local and goods separately. The same thing happens to the units sold. Therefore, I use other gross billing as one of the explanatory variables to generate the model rather than use travel gross billing and goods gross billing separately as two variables. Also, using the combination of travel and good segment as one variable will decrease the NAN values in the independent variable, because the earliest travel deal’s start date is April 9th, 2013, which means if I use it as a variable, a lot of missing values will be introduced from the earliest local deals’ start date June 6th, 2012 to April 9th, 2013. However, if I combine travel and goods, I will get complete data without missing value starts from March 23rd, 2013 as shown below.
 

	Base on the daily local gross billing plot, the value is fluctuating. However, based on the autocorrelation and partial autocorrelation plot, there isn’t a clear seasonal trend in the data. It is possible that more great deals will be published on Groupon’s website deal during weekends or holidays, and this will significantly affect the units sold and gross billing. To understand how time stamp variables affect the gross billing, the day of week, week of the year and month of year variables are introduced into the dataset. While fitting the dataset to different machine learning models, I output the importance plot that shows which variable is the most significant for predicting the local gross billing. The result comes out that the week of year is the most important predictor among other variables as shown in the plot below.
 
Time Series Estimation Approach

	The first idea that I had was fitting an ARIMA model to the local gross billing to capture the pattern of the data and estimate the missing data points base on the pattern. Therefore, I extract the local gross billing data from the original dataset and make the start date as the time index. Aggregate the local gross billing to daily data. 
	Base on the time series plot and Dickey-Fuller test result at a 95% confidence level, this series is a non-stationary time series. It includes an increasing trend, increasing variance and non-seasonality pattern. To fit an ARIMA model, the data has to be a stationary time series. If the time series is non-stationary, it’s average value and variance will fluctuate over time. There is no way to fit a model and predict anything without a fixed pattern. Therefore, I performed some transformation to make it a stationary time series. 
	There are some issues with the transformation process. First, there is a large number of zeros in daily local gross billing for deals start in 2012. And there are two negative values in the dataset. Taking the log of zero or negative values will bring infinite number and error. Because there are a lot of zeros in 2012 and the most local gross billings for deals start in 2012 are small, I decided to only use the data from 01/01/2013 to 10/19/2013 to estimate the missing data points. Furthermore, it is unreasonable to impute all those zeros values because they are real data points (without wrong information). There are only two local gross billing are negative in 2013 start deals, I replaced them with the median of the dataset.
	After transforming the series into a stationary time series, I can start working on fitting an ARIMA model to the log-transformed series. I created a “for loop” that includes AR and MA values from 1 to 5 so that the program can select the best combination of AR and MA for me. Also, as mentioned above, because I took the first difference to the transformed series, the difference parameter d was set to 1. I chose the combination of parameters that can produce the smallest AIC value as the best ARIMA model parameters. For the final time series model, both the AR and MA parameter equals to 3, and the best model is ARIMA (3,1,3).
	Base on the standardized residual plot and the Ljung-Box test results, the model has captured all the features of this time series. Furthermore, the residual is normally distributed and there is no correlation between each lag.
	To test the model accuracy, I compared the predicted result with the actual data and obtained the mean square error (MSE) which can be treated as a criterion that measures the accuracy of the model. I will use the MSE as the standard for final model selection. The MSE for this ARIMA (3,1,3) model is 7.066e+11. 
	I then used this model to predict the missing local gross billing of deals starts from 10/20/2013 to 10/30/2013. The prediction results are in the final estimation result table.
	The issue with time series analysis is, it doesn’t utilize the information from 11/01/2013 to 12/31/2013 to impute the missing data. These may lower the accuracy of the prediction. Also, building up the time series model only use local gross billing information, but didn’t capture the correlation and influence from other variables. Therefore, next I use machine-learning algorithms to estimate the missing data points.

Machine Learning Estimation Approach

	To implement machine-learning algorithms to the dataset, I first generated explanatory variables, and reshaped the data to observations by features. 
As mentioned before, there is a positive relationship between local gross billing and other gross billing, local units sold and other units sold.  Furthermore, the time stamp variables such as the day of a week, week of the year and month of year also contain useful information for the estimation. 
Therefore, I chose, other units sold, other gross billing, day of the week, week of the year and month of year to be the explanatory variables, and local gross billing as the response variable. The reason I didn’t choose the local units sold as one of the five explanatory variables is, there is no data from 10/20/2013 to 10/30/2013, and local units sold is very possible to become a significant variable for the prediction. However, there is no reason to use missing values to predict missing values. Also, it will produce a larger error if I impute those missing local units sold first, and then use the imputed values to estimate the local gross billing. To reduce the influence of missing values, I decided not to use the local units sold as one of the explanatory variables.

	As I mentioned in the patterns section, if I use the whole dataset then there will be a lot of missing values. They can’t be all imputed because it will lead to inconsistencies with reality. Therefore, I dropped the data before 3/25/2013. 

	This time, because the data is not time-indexed anymore, I can use all the information in the dataset rather than only use the data before 10/20/2013 to build up the model. Let’s call this dataset as the base dataset. The base dataset contains information from 3/25/2013 to 10/19/2013 and 10/31/2013 to 12/31/2013. This base dataset will be used to train the machine learning model. Once I have the model, I can use the model to estimate local gross billing from 10/20/2013 to 10/30/2013.
	The model training process is very similar. I trained a random forest model, a gradient boosting model, and a decision tree model for this assignment. I will describe the process of training the gradient boosting model here as an example.
	First, I separate the base data into 70% training data and 30% testing data with a random sample. And then I construct a grid search for parameter tuning, these parameters include the number of estimators, the maximum depth of each decision tree, the minimum number of samples in each leaf, the minimum number of samples in each split, the number of features to consider and the learning rate.

	Second, fitted the model with training set and used the model to predict the response variable. Once I had the predicted result, I compared it with the test set response variable which is the true values. And then calculated the mean square error and plotted the importance of each variable.

	The formula for calculating MSE is:
	MSE=∑_(i=1)^n▒〖(y_i-(y_i ) ̂)〗^2/n
	The mean square errors were calculated by MSE function in python. All the codes are in the Appendix.
	Third, bring in the 10/20/2013 to 10/30/2013 dataset and predict the missing local gross billing with the known explanatory variables. 
	The results of the random forest, gradient boosting, decision tree, and ARIMA models are shown in the table below. As you can see the gradient boosting model has the smallest MSE, so I select its result as the final estimation value.

Final Estimation Result Table

				Random Forest Gradient Boosting Decision Tree	 ARIMA
	Local Gross Billing	$444026162.39	$445028140.83	$446041574.63	$444728012.98
	Travel Gross Billing	$70547395.73	$70547395.73	$70547395.73	$70547395.73
	Goods Gross Billing	$282245469.1	$282245469.1	$282245469.1	$282245469.1
	Total		        $796819027.22	$797821005.65	$798834439.45	$79720877.81
	MSE			3.8e+11	        2.95e+11	3.9e+11	        7.07e+11

Buy or Sell Recommendation

	For the buy or sell recommendation, I would like to compare the equity reports estimation with mine. 
	Deutsche Bank Markets Research	DB 3Q 2013 (Million)	DB 4Q 2013E (Million)	Q/Q Growth Rate
	NA Local Gross Billing			$403	               N/A		N/A
	NA Travel Gross Billing			$68	               N/A	        N/A
	NA Goods Gross Billing			$195		       N/A	        N/A
	NA Total Gross Billing			$665	               $803.2	        20.78%

	JPMorgan Equity Research	JP Morgan 3Q 2013 (Million)	DB 4Q 2013E (Million)	Q/Q Growth Rate
	NA Local Gross Billing				$403	               $490.142         21.7%
	NA Travel Gross Billing				$68	               $71.020	          5%
	NA Goods Gross Billing				$195	               $275.716	        41.7%
	NA Total Gross Billing				$665		       $836.88		25.85%

	Morgan Stanley Research	MS GRPN 3Q 2013 (Million)	DB 4Q 2013E (Million)	Q/Q Growth Rate
	NA Local Gross Billing			$403		$508                      26.1%
	NA Travel Gross Billing			$68	        $67	                  -1.4%
	NA Goods Gross Billing			$195	        $295.4	                  51.5%
	NA Total Gross Billing			$665	        $870.1			  30.84%

	My Estimation			3Q 2013 (Million)	4Q 2013E (Million)	Q/Q Growth Rate
	NA Local Gross Billing		$403			$445.03			10.43%
	NA Travel Gross Billing		$68			$70.55			3.75%
	NA Goods Gross Billing		$195			$282.25			44.74%
	NA Total Gross Billing		$665			$797.82			19.97%

	From the tables above, my estimation of total gross billing is lower than all other reports’ results. Also, the quarter over quarter growth rate is lower than other results.  
	The formula I used to calculate the Q/Q growth rate of my estimation is:
	((Current Year Estimated Gross Billing-Last Year Estimated Gross Billing))/(Last Year Estimated Gross Billing )*100%
	Local∶(($445.03-403))/($445.03 )*100%=10.43%
	Travel:  (($70.55-68))/$68*100%=3.75%
	Goods:  (($282.25-195))/$195*100%=44.74%
	Total:  (($789.82-665))/$665*100%=19.97%
	In other words, both of my estimated amount of 4Q 2013 North America total gross billing and quarter over quarter gross billing growth rate are lower than the market’s expected value. Therefore, I would recommend my client to sell Groupon’s stock. 

