# Uber_Lyft_Price_Prediction

Data Science Project: Real-Time Price Prediction of Uber and Lyft in Boston

Original Analysis Report: https://medium.com/@zy145/data-science-project-real-time-price-prediction-of-uber-and-lyft-in-boston-66be98ba1724

## Table of contents
* [Introduction](#Introduction)
* [Unfold the Data](#Unfold_the_Data)
* [Data Preparation](#Data_Preparation)
* [Visualization: Overview of the Boston Market](#Visualization_Overview_of_the_Boston_Market)
* [Modeling](#Modeling)
* [Model Evaluation](#Model_Evaluation)
* [Deployment](#Deployment)
* [Works Cited](#Works_Cited)


## Introduction
The ride-hailing service industry is booming with billions of dollars in value in the United States. However, the market is dominated by the big two companies, Uber and Lyft. For the start-ups who intend to break the monopolization and enter this booming market, price war can be used as the most effective typical pricing strategy. As long as the start-up can provide a cheaper price, customers will be more likely to prioritize their purchase decisions over the big two. In order to leverage price war to win the market share, preliminary research on the competitor’s pricing strategy is essential. Also, it is unlikely to directly obtain the pricing model of Uber and Lyft due to competitive restrictions. Unlike other industries, the price per ride is fluctuating on a timely basis. Except for fixed costs, factors like time, start location, and weather also play significant roles in pricing per ride. Therefore, it is necessary to provide an effective real-time prediction model based on available data. If the start-up could utilize effective prediction models, the chance to use cheaper prices to acquire potential customers and snatch customers from competitors would be largely improved.

We, the №14 consulting group, have a client who would like to enter the ride-hailing service industry through the potential market in the center of Boston. The start-up company would like us to provide some general insights into the Boston ride-hailing market. And more importantly, they require us to make two good models for real-time price prediction per ride of

Uber and Lyft in several source locations, so that they can add the models to their algorithms as a reference to providing the competitive price on Google Map, where the customers can see all the prices provided by different carrier companies.

![image](https://user-images.githubusercontent.com/98763622/162604723-1bf8a4bb-9549-43d7-9cdf-334e7427d2b9.png)
(Picture Source: https://thenextweb.com/news/google-maps-will-now-show-ubers-on-directiontellingthingy)

## Unfold_the_Data

The datasets we used for this project are collected by Ravi Munde and Karan Barai on Github. The time frame tracked for the dataset is the week of November 18, 2018. They have tracked the ride price information every 5 minutes. The weather information was tracked every hour (Munde and Barai). A potential bias that might be happening here is the weather tracked by the hour, since weather may change suddenly at a minute level.

There were 2 original datasets: one is for ride information(cab_rides); one is for weather information(weather). Here are the variable details:

● Cab_rides: customer-level data with 693071 observations

○ distance(in miles), cab_type(Lyft/Uber), time_stamp(Epoch Time), destination, source, price, surge_multiplier, id(Identical ride id), product_id, name(Ride Product Name)

● Weather: hourly weather data with 6277 observations

○ temp(Temperature), location, clouds, pressure, rain(inch), time_stamp (Epoch Time), humidity, wind

As mentioned, for each of the companies, the real-time price of a ride is related to independent variables like distance, time of the day and the week, start locations, and weather. What’s more, hailing-service products like Uber XL or Uber Pool may also play a role in the pricing, so we used them as dummy independent variables. Finally, there is an extra price surge that may be added due to the unbalanced supply and demand of rides at a specific time. However, since we do not have enough supply information like real-time driver data, we cannot predict the surge. Thus, we decide to create a new variable called adjusted_price by price/surge_multiplyer as the dependent variable we are going to model for.

## Data_Preparation

### Dataset Merging & Cleaning

We firstly join the two datasets by the combined primary key (Source and Time_Stamp). To solve the inconsistencies of the data in two datasets, we alter the format (keep the same digits) and unit ( by date-hour-minute) of Time_Stamp to make sure the merging process is successful. Next, we drop the Null values and some uninformative variables that are not for modeling such as Product_id and Destination. As the remaining variables have categorical variables that cannot be modeled directly, we create dummy variables for each of them. It is worth noting that, to precisely predict the price considering the time factor, we use Time_Stamp to split the time of day (morning, late morning, afternoon, etc.) and create related dummy variables.

### Dealing with surge

As it is listed on Uber’s official website, surge pricing is a special pricing strategy that balances the marketplace over time(“Uber Help.”). As the dataset lacks information about the supply end, we cannot predict the surge. Therefore, we alter the initial target variable Price by dividing Surge to eliminate the influence of the marketplace to come up with a new variable Adjusted_Price. This is the final target variable that we use for modeling and predicting.

### Split the dataset for modeling

The modeling process is to compare different models to come up with the optimized models separately for Uber and Lyft. Based on this demand, we split the final joint dataset into two separate datasets by company (Uber and Lyft).

## Visualization_Overview_of_the_Boston_Market

![image](https://user-images.githubusercontent.com/98763622/162604959-cde5b1b7-3375-46d4-aade-d8c32d905d12.png)

(map source: https://bostonopendata-boston.opendata.arcgis.com/)

To provide some market insights for our client, we have made these 2 data visualizations.

The left one shows the geographic distribution of the total ride in the center Boston area. All the specific regions observed are equally distributed with 8% of the total ride number. The downtown area has the most ride.

The right visualization shows the regional ride competition by Uber and Lyft with a marginal percentage of Uber rides. A conclusion can be drawn from the picture: Uber and Lyft are almost equally distributed in the downtown area and the Back Bay area. As it is more away from the center area, Uber owns more rides.

## Modeling

We decided to use supervised data mining methods to predict our continuous dependent variable (adjusted_price) for both Uber and Lyft. All the processes below were applied twice for predicting the real-time prices of each of the companies.

Our Initial Model Estimation is :

adjusted_price= B0+B1*temp+B2*clouds+B3*pressure+B4*rain+B5*humidity+B6*wind+ B7*weekdaysfactor+B8*time_of_day_factor+B9*name_factor+ B10*source_factor

In the equation, weekdaysfactor, time_of_day_factor, name_factor, and source_factor are all categorical variables that will be automatically generated as dummy variables in the regression model.

Following are the models that we have used to predict the price of Uber and Lyft, and we have used K-fold (K=10) cross-validation to examine the best performing model:

a) Linear Regression (Initial Model)

b) Linear Regression (With ^2 Interaction)

c) Lasso (min lambda)

d) Post Lasso (min lambda)

e) Random Forest

The methodology for deciding the model types to examine is based on the concern of underfitting or overfitting of the initial model. We use the interaction model to find more cross-related factors while we use Lasso Regression and Post Lasso to reduce overfitting since we have too many variables after adding dummy variables from the categorical variables. Finally, we also choose one more method of Random Forest to diversify our model type and the possibility of finding the best model.

## Model_Evaluation

We have evaluated the result of data mining algorithms using RMSE. The Root Mean Square Error that depicts the standard deviation of residuals throws light on how spread out the residuals are. In the business application, it helps us understand how exactly well in a quantitative way the model can predict. We have chosen this parameter for the evaluation of results over the parameter of R square that reflects how well the data has been explained. The reason for giving precedence to RMSE over R square is that the client needs a method that can accurately predict the price, not explain the past data. Since the models are also related to the price war, every dollar of error is essential for being the lowest price among the 3 brands. This makes RMSE a better and preferred choice for concluding the results.

Below are the RMSE and model details for Lyft and for Uber:

![image](https://user-images.githubusercontent.com/98763622/162605009-a8b57e69-c68a-49f0-a579-af8c574c601d.png)

Utilizing the K-Fold Cross Validation, Random Forest turned out to be the best performer for Lyft as it had the lowest RMSE(1.44). Finally, we applied the random forest to the whole cleared dataset and got our best model.

![image](https://user-images.githubusercontent.com/98763622/162605016-350e7aca-e695-4895-bb55-d5c0797eea66.png)

Utilizing the same method, Linear Regression with Interactions turned out to be the best performer for Uber as it had the lowest RMSE(2.04). Finally, we applied the Linear Regression with Interactions to the whole cleared dataset and got our best model.

## Deployment

Our data mining results provide two models for the start-up company to predict their competitors’ real-time price: one linear-interaction model for Uber price prediction and one random forest model for Lyft. The start-up can use these 2 predict models as part of the pricing reference on their own algorithm.

However, they should be aware of some potential biases, for example, the weather parameters are hourly based, when they apply the model. What’s more, the prediction result does have limitations on the accuracy due to the nature of the data and the sample size.

Finally, since ride information is not publicly disclosed by either Uber or Lyft, there might be some legal concern regarding using the dataset we have in a commercial way. And there might also be some legal risk of unfair competition on predicting competitors’ prices without authorized data.

## Works_Cited

* Ravi Munde, and Karan Barai. “Uber & Lyft Cab Prices: Cab and Weather Dataset to Predict Cab Prices against Weather.” www.kaggle.com/ravi72munde/uber-lyft-cab-prices.
* “Uber Help.” Uber Help: What Is Surge?, help.uber.com/driving-and-delivering/article/what-is-surge?nodeId=e9375d5e-917b-4bc5 -8142–23b89a440eec.
* Lopez, Napier. “Google Maps Now Displays Uber Drivers in Real-Time.” TNW | Google, 12 Jan. 2017,thenextweb.com/news/google-maps-will-now-show-ubers-on-directiontellingthingy.
* “Boston Maps Open Data.” Geospatial Datasets, City of Boston’s GIS Team, bostonopendata-boston.opendata.arcgis.com/.
