###The Code File Includes the Efforts of Every Team Member, Though Some of Them May Not Contributed to Our Final Reports. 
###
### Data Cleaning Process
#Load datasets
CabRides <- read.csv("cab_rides.csv")
Weather <- read.csv("weather.csv")
summary(CabRides)
summary(Weather)

#Convert the type of the time variable "time_stamp"from character to time
CabRides$time_stamp <- as.POSIXct(CabRides$time_stamp/1000, origin = "1970-01-01")   
Weather$time_stamp <- as.POSIXct(Weather$time_stamp, origin = "1970-01-01")
CabRides$time_stamp

#Create "new_time_stamp", a new time variable with seconds removed 
#so that we can merge two datasets while keeping enough rows.
CabRides$new_time_stamp <- format(CabRides$time_stamp, format="%Y-%m-%d %H:%M")
CabRides$new_time_stamp
Weather$new_time_stamp <- format(Weather$time_stamp, format="%Y-%m-%d %H:%M")

#Change the missing value in weather$rain to 0, 
#then group by those rows whose new_time_stamp is the same, 
#and average the weather parameters.
Weather$rain[is.na(Weather$rain)] = 0
summary(Weather)
New_Weather <- aggregate(cbind(temp, clouds, pressure, rain, humidity, wind) ~ new_time_stamp+location, data = Weather, FUN = mean, na.rm = TRUE)

#Merge two processed datasets based on two columns, origin location and time.
Merged <- merge(CabRides, New_Weather, by.x=c("source", "new_time_stamp"), by.y=c("location", "new_time_stamp"))
summary(Merged)

#Drop three columns that are not needed"destination","id", and "product_id".
drop <- c("destination","id","product_id")
Dropped <- Merged[,!(names(Merged) %in% drop)]
summary(Dropped)
Dropped <- na.omit(Dropped)
summary(Dropped)

#Generate a dummy variable "company" to help separate the data from the two companies in case necessary.
unique(Dropped$cab_type)
lookup <- c("Lyft" = 0, "Uber" = 1)
Dropped$company <- lookup[Dropped$cab_type]

#Generate a new variable representing the weekdays the orders occurred and converts it to dummy variable.
Dropped$weekdays <- weekdays(Dropped$time_stamp)
Dropped$weekdaysfactor <- factor(Dropped$weekdays)
summary(Dropped)

#Generate a new variable to represent the time period of the day the orders occurred.
install.packages("tidyverse")
time_of_day <- function(time) {
  
  hr <- lubridate::hour(time)
  dplyr::case_when(hr > 6 & hr < 9 ~ 'early morning', 
                   hr >= 9 & hr < 12 ~ 'late morning',
                   hr >= 12 & hr < 17 ~ 'afternoon', 
                   hr >= 17 & hr <= 21 ~ 'evening', 
                   TRUE ~ 'night')
}
Dropped$time_of_day <- time_of_day(Dropped$time_stamp)

#Transform the three variables of time period, vehicle type, and origin into dummy variables
Dropped$time_of_day_factor <- factor(Dropped$time_of_day)
Dropped$name_factor <- factor(Dropped$name)
Dropped$source_factor <- factor(Dropped$source)
summary(Dropped)

#Drop some of the initial variables that are not needed.
drop <- c("time_stamp","source","name","weekdays","time_of_day")
Dropped <- Dropped[,!(names(Dropped) %in% drop)]

#Surge is used to adjust price to reduce the impact of supply and demand on price.
Dropped$adjusted_price <- Dropped$price/Dropped$surge
summary(Dropped)

#Divide the cleaned dataset into two datasets based on the two companies.
split <- split(Dropped, Dropped$cab_type)
Uber <- split$Uber
Lyft <- split$Lyft

#Verify the cleaning results
summary(Dropped)
summary(Uber)
summary(Lyft)

#Save cleaned datasets
write.csv(Dropped,"Cleaned_Complete_Dataset.csv")
write.csv(Uber,"Cleaned_Uber_Dataset.csv")
write.csv(Lyft,"Cleaned_Lyft_Dataset.csv")

### Data Analyzing
### visualization

#scatter and boxplot
####
group <- ifelse(Dropped$company == 1, "Uber","Lyft")
# Build the plot
unique(Dropped$name_factor)
unique(Lyft$name_factor)
unique(Uber$name_factor)
library(ggplot2)
ggplot(Dropped,
       aes(x = name_factor, 
           y = price)) + 
  geom_boxplot(aes(colour = cab_type), 
               position = position_dodge(width = .8))+
  labs(title = "Order Price Distribution by Cab Types", 
       x = "Cab Type",
       colour = "Company")

#### observe the correlation between distance and price by two companies
library(tidyverse)
theme_set(theme_bw(base_size=16))

Dropped %>%
  ggplot(aes(x=distance, 
             y=price))+
  geom_point(aes(color=cab_type))+
  geom_smooth(aes(color=cab_type), method="lm")+
  labs(title = "Distance vs Order Price by Company", colour = "Company")

### Mapping
###################
install.packages("dplyr")
install.packages("ggplot2")
install.packages("RColorBrewer")
install.packages("maps")
install.packages("mapproj")
library(plyr)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(maps)
library(mapproj)

### create number of orders dataset
order_number <- data.frame(ddply(Dropped, .(cab_type,source_factor), nrow))
order_number <- order_number[,2:3]
order_number$Uber <- order_number[13:24,2]
names(order_number)[2] <- 'Lyft'
order_number <- order_number[1:12,]
order_number$Total <- order_number$Lyft + order_number$Uber

order_number$Uber_margin <- order_number$Uber - order_number$Lyft
order_number$Uber_margin_percent <- order_number$Uber_margin/order_number$Total
order_number$Total_percent <- order_number$Total / sum(order_number$Total)
#based on the data of bostonopendata website, we give each neighborhoods their id
order_number$Neighborho <- c(2,30,28,34,27,7,14,4,13,33,26,31)
summary(order_number)
#create average price dataset
average_price <- data.frame(ddply(Dropped, .(cab_type,source_factor),summarise, mean = mean(price)))
average_price <- average_price[,2:3]
average_price$Uber <- average_price[13:24,2]
names(average_price)[2] <- 'Lyft'
average_price <- average_price[1:12,]
average_price$Uber_margin <- average_price$Uber - average_price$Lyft
average_price$Neighborho <- c(2,30,28,34,27,7,14,4,13,33,26,31)

require(mapproj)
# Turn off scientific notation
options(scipen = "999") 
# Ensure strings come in as character types
options(stringsAsFactors = FALSE)

# Install packages
library(ggplot2)
install.packages("ggmap")
library(ggmap)
install.packages("maptools")
install.packages("sp")
library(maptools)
install.packages("ggthemes")
library(ggthemes)
install.packages("rgeos")
library(rgeos)
library(broom)
library(dplyr)
library(plyr)
library(grid)
install.packages("gridExtra")
library(gridExtra)
library(reshape2)
library(scales)
install.packages("rgdal")
library(rgdal)

#get boston neighborhoods map dataset.(source:https://bostonopendata-boston.opendata.arcgis.com/datasets/)
#unzip the dataset
unzip("Boston_Neighborhoods.zip")
#read the unzipped dataset folder
neighb <- rgdal::readOGR("Boston_Neighborhoods")
#check if it is boston :)
plot(neighb)

library(ggplot2)
library(ggmap)

#Define our map theme
mapTheme <- function(base_size = 12) {
  theme(
    text = element_text( color = "black"),
    plot.title = element_text(size = 18,colour = "black"),
    plot.subtitle=element_text(face="italic"),
    plot.caption=element_text(hjust=0),
    axis.ticks = element_blank(),
    panel.background = element_blank(),
    panel.grid.major = element_line("grey80", size = 0.1),
    strip.text = element_text(size=12),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey80", color = "white"),
    plot.background = element_blank(),
    legend.background = element_blank(),
    legend.title = element_text(colour = "black", face = "italic"),
    legend.text = element_text(colour = "black", face = "italic"))
}

# Define our map palette
palette_8_colors <- c("#0DA3A0","#2D97AA","#4D8CB4","#6E81BF","#8E76C9","#AF6BD4","#CF60DE","#F055E9")

#create the edge of the map based on the summary of the order_number
bbox <- neighb@bbox
boston_bbox <- c(left = bbox[1,1], bottom = bbox[2,1],
                 right = bbox[1,2], top = bbox[2,2])

# load the basemap
basemap <- get_stamenmap(
  bbox = boston_bbox,
  zoom = 12,
  maptype = "toner-lite")
# Map it
bmMap <- ggmap(basemap) + mapTheme() + 
  labs(title="Boston basemap")
bmMap


# Plot the Uber order number margin percent
#convert it to a format that ggplot understands
neighb.tidy <- tidy(neighb)
# Recover row name 
temp_df <- data.frame(neighb@data$Neighborho)
names(temp_df) <- c("Neighborho")
# Create and append "id"
temp_df$id <- seq(0,nrow(temp_df)-1)
neighb.tidy <- join(neighb.tidy, temp_df, by="id")
#neighb.tidy <- tidy(neighb, region = c('nbrhood'))

# Look at the resulting data frame to see how it has been transformed
head(neighb.tidy)
summary(neighb.tidy)

# Now we're going to join these data frames together so that when we map the neighborhood polygons we can symbolize them using the summary
# stats we created
order_number_tidy <- join(order_number, neighb.tidy, by = c("Neighborho"), match = "all")
average_price_tidy <- join(average_price, neighb.tidy, by = "Neighborho", match = "all")

#create maps time!

#Uber order number margin map
order_number_map <- ggmap(basemap) +
  geom_polygon(data = order_number_tidy, 
               aes(x = long, y = lat, group = group, fill = Uber_margin_percent), 
               colour = "white", alpha = 0.75, size = 0.25) + 
  scale_fill_gradientn("% of Uber Rides Margin over Lyft", colors = palette_8_colors,
                       labels = scales::percent_format()) +
  mapTheme() + theme(legend.position = "bottom", 
                     legend.direction = "horizontal", 
                     legend.key.width = unit(.5, "in")) +
  labs(title="% of Uber Rides Margin over Lyft by Neighborhood, Boston")
order_number_map

#Total order number percent map
order_percent_map <- ggmap(basemap) +
  geom_polygon(data = order_number_tidy, 
               aes(x = long, y = lat, group = group, fill = Total_percent), 
               colour = "white", alpha = 0.75, size = 0.25) + 
  scale_fill_gradientn("% of Total Rides", colors = palette_8_colors,
                       labels = scales::percent_format()) +
  mapTheme() + theme(legend.position = "bottom", 
                     legend.direction = "horizontal", 
                     legend.key.width = unit(.5, "in")) +
  labs(title="% of Total Rides by neighborhood, Boston")
order_percent_map

#Uber average order price margin map
price_margin_map <- ggmap(basemap) +
  geom_polygon(data = average_price_tidy, 
               aes(x = long, y = lat, group = group, fill = Uber_margin), 
               colour = "white", alpha = 0.75, size = 0.25) + 
  scale_fill_gradientn("Marginal Avg. Price of Uber over Lyft", colors = palette_8_colors,
                       labels = scales::dollar_format(prefix = "$")) +
  mapTheme() + theme(legend.position = "bottom", 
                     legend.direction = "horizontal", 
                     legend.key.width = unit(.5, "in")) +
  labs(title="Marginal Avg. Price of Uber over Lyft by Neighborhood, Boston")
price_margin_map



#Linear Model with Bonferroni correction

Cleaned_Lyft_Dataset <- read.csv("Cleaned_Lyft_Dataset.csv")
Cleaned_Uber_Dataset <- read.csv("Cleaned_uber_Dataset.csv")
View(Cleaned_Lyft_Dataset)

lyft<-data.frame(Cleaned_Lyft_Dataset)
uber<-data.frame(Cleaned_Uber_Dataset)

######Drop useless columns
lyft = subset(lyft, select = -c(new_time_stamp,cab_type, surge_multiplier,company))
uber = subset(uber, select = -c(new_time_stamp,cab_type, surge_multiplier,company))

print('Modified dataframe:-')
lyft

######Create Dummy Variables
##Weekdaysfactor
lyft$Tuesday <- ifelse(lyft$weekdaysfactor == 'Tuesday', 1, 0)
lyft$Wednesday <- ifelse(lyft$weekdaysfactor == 'Wednesday ', 1, 0)
lyft$Thursday <- ifelse(lyft$weekdaysfactor == 'Thursday', 1, 0)
lyft$Friday <- ifelse(lyft$weekdaysfactor == 'Friday', 1, 0)
lyft$Saturday <- ifelse(lyft$weekdaysfactor == 'Saturday', 1, 0)
lyft$Sunday <- ifelse(lyft$weekdaysfactor == 'Sunday', 1, 0)

uber$Tuesday <- ifelse(uber$weekdaysfactor == 'Tuesday', 1, 0)
uber$Wednesday <- ifelse(uber$weekdaysfactor == 'Wednesday ', 1, 0)
uber$Thursday <- ifelse(uber$weekdaysfactor == 'Thursday', 1, 0)
uber$Friday <- ifelse(uber$weekdaysfactor == 'Friday', 1, 0)
uber$Saturday <- ifelse(uber$weekdaysfactor == 'Saturday', 1, 0)
uber$Sunday <- ifelse(uber$weekdaysfactor == 'Sunday', 1, 0)
##time_of_day_factor
lyft$early_morning <- ifelse(lyft$weekdaysfactor == 'early morning ', 1, 0)
lyft$evening <- ifelse(lyft$weekdaysfactor == 'evening  ', 1, 0)
lyft$late_morning <- ifelse(lyft$weekdaysfactor == 'late morning ', 1, 0)
lyft$night <- ifelse(lyft$weekdaysfactor == 'night', 1, 0)


uber$early_morning <- ifelse(uber$weekdaysfactor == 'early morning ', 1, 0)
uber$evening <- ifelse(uber$weekdaysfactor == 'evening  ', 1, 0)
uber$late_morning <- ifelse(uber$weekdaysfactor == 'late morning ', 1, 0)
uber$night <- ifelse(uber$weekdaysfactor == 'night', 1, 0)



##name_factor
lyft$Lux_Black<- ifelse(lyft$name_factor == 'Lux Black', 1, 0)
lyft$Lux<- ifelse(lyft$name_factor == 'Lux', 1, 0)
lyft$Lyft<- ifelse(lyft$name_factor == 'Lyft', 1, 0)
lyft$Lyft_XL<- ifelse(lyft$name_factor == 'Lyft XL', 1, 0)
lyft$Shared<- ifelse(lyft$name_factor == 'Shared', 1, 0)


uber$Black<- ifelse(uber$name_factor == 'Black', 1, 0)
uber$UberPool<- ifelse(uber$name_factor == 'UberPool', 1, 0)
uber$UberX<- ifelse(uber$name_factor == 'UberX', 1, 0)
uber$UberXL<- ifelse(uber$name_factor == 'UberXL', 1, 0)
uber$WAV<- ifelse(uber$name_factor == 'WAV', 1, 0)



##source_factor
lyft$Fenway<- ifelse(lyft$source_factor == 'Fenway', 1, 0)
lyft$Back_Bay<- ifelse(lyft$source_factor == 'Back Bay', 1, 0)
lyft$Beacon_Hill<- ifelse(lyft$source_factor == 'Beacon Hill', 1, 0)
lyft$Financial_District- ifelse(lyft$source_factor == 'Financial District', 1, 0)
lyft$Haymarket_Square<- ifelse(lyft$source_factor == 'Haymarket Square', 1, 0)
lyft$North_End<- ifelse(lyft$source_factor == 'North End', 1, 0)
lyft$North_Station<- ifelse(lyft$source_factor == 'North Station', 1, 0)
lyft$Northeastern_University<- ifelse(lyft$source_factor == 'Northeastern University', 1, 0)
lyft$South_Station<- ifelse(lyft$source_factor == 'South Station', 1, 0)
lyft$Theatre_District<- ifelse(lyft$source_factor == 'Theatre District', 1, 0)
lyft$Boston_University<- ifelse(lyft$source_factor == 'Boston University', 1, 0)

uber$Fenway<- ifelse(uber$source_factor == 'Fenway', 1, 0)
uber$Back_Bay<- ifelse(uber$source_factor == 'Back Bay', 1, 0)
uber$Beacon_Hill<- ifelse(uber$source_factor == 'Beacon Hill', 1, 0)
uber$Financial_District- ifelse(uber$source_factor == 'Financial District', 1, 0)
uber$Haymarket_Square<- ifelse(uber$source_factor == 'Haymarket Square', 1, 0)
uber$North_End<- ifelse(uber$source_factor == 'North End', 1, 0)
uber$North_Station<- ifelse(uber$source_factor == 'North Station', 1, 0)
uber$Northeastern_University<- ifelse(uber$source_factor == 'Northeastern University', 1, 0)
uber$South_Station<- ifelse(uber$source_factor == 'South Station', 1, 0)
uber$Theatre_District<- ifelse(uber$source_factor == 'Theatre District', 1, 0)
uber$Boston_University<- ifelse(uber$source_factor == 'Boston University', 1, 0)
######Drop factor columns 
lyft = subset(lyft, select = -c(...1,weekdaysfactor, time_of_day_factor, name_factor, source_factor))
print('Modified dataframe:-')
lyft

uber = subset(uber, select = -c(...1,weekdaysfactor, time_of_day_factor, name_factor, source_factor))
print('Modified dataframe:-')
uber
######Drop empty columns
summary(lyft)
lyft = subset(lyft, select = -c(early_morning,evening,late_morning,night,Wednesday,price))
print('Modified dataframe:-')
lyft

summary(uber)
uber = subset(uber, select = -c(early_morning,evening,late_morning,
                                night,Wednesday,price))
print('Modified dataframe:-')
summary(uber)



##############Modeling################
######################################
###Multiple Linear Regression Model
lyftlm<- lm(adjusted_price~., data=lyft)
summary(lyftlm) 


uberlm<- lm(adjusted_price~., data=uber)
summary(uberlm) 
###Bonferroni Test
sum( summary(lyftlm)$coef[,4] > 0.05/length(summary(lyftlm)$coef[,4]) )
sum( summary(uberlm)$coef[,4] > 0.05/length(summary(uberlm)$coef[,4]) )
#Create the a list of p-value
lyftPvalues<-summary(lyftlm)$coef[,4]
lyftPvalues

uberPvalues<-summary(uberlm)$coef[,4]
uberPvalues

#Identify the variables that fail to pass the Bonferroni Test
lyftGoodPvalues2<- Pvalues< 0.05/27
lyftGoodPvalues2

uberGoodPvalues2<- Pvalues< 0.05/27
uberGoodPvalues2

#Drop those variables
lyft2 = subset(lyft, select = -c(temp,clouds,pressure, rain,humidity, wind, 
                                 Tuesday, Thursday, Friday, Saturday, 
                                 Sunday,Fenway, Back_Bay, Haymarket_Square))

uber2 = subset(uber, select = -c(temp,clouds,pressure, rain,humidity, wind, 
                                 Tuesday, Thursday, Friday, Saturday, 
                                 Sunday,North_Station, South_Station ,Theatre_District))
#Multiple Linear Regression Model-Bonferroni
lyft2lm<- lm(adjusted_price~., data=lyft2)
summary(lyft2lm) 

uber2lm<- lm(adjusted_price~., data=uber2)
summary(uber2lm) 
#Te see if all variables are significant based on Bonferroni
sum( summary(lyft2lm)$coef[,4] > 0.05/length(summary(lyft2lm)$coef[,4]) )
sum( summary(uber2lm)$coef[,4] > 0.05/length(summary(uber2lm)$coef[,4]) )

#Quantile Regression
install.packages("quantreg")
# Install fastDummies:
install.packages('fastDummies')
library('fastDummies')
library(quantreg)

Lyft <- read.csv("Cleaned_Lyft_Dataset.csv")

drops<- c("company","new_time_stamp", "surge_multiplier","price","cab_type","X")
names(Lyft) %in% drops
!( names(Lyft) %in% drops )
Lyft2 <- Lyft[,!(names(Lyft) %in% drops)]
summary(Lyft2)
Lyft2

rqfit <- rq(adjusted_price ~ ., data = Lyft2)
rqfit
summary(rqfit)


Uber<-read.csv("Cleaned_Uber_Dataset.csv")



drops<- c("company","new_time_stamp", "surge_multiplier","price","cab_type","X")
names(Uber) %in% drops
!( names(Uber) %in% drops )
Uber2 <- Uber[,!(names(Uber) %in% drops)]
summary(Uber2)

Uber2

library(quantreg)
rqfit1<-rq(adjusted_price ~ ., data=Uber2)
summary(rqfit1)

#Lasso Regression
Lyft<-read.csv("Cleaned_Lyft_Dataset.csv")
summary(Lyft)

drops<- c("company","new_time_stamp", "surge_multiplier","price","cab_type","X")
names(Lyft) %in% drops
!( names(Lyft) %in% drops )
Lyft2 <- Lyft[,!(names(Lyft) %in% drops)]
summary(Lyft2)


x<-model.matrix(adjusted_price ~ .,data = Lyft2)
library(glmnet)
cv_model <- cv.glmnet(x,Lyft2$adjusted_price, alpha = 1)
summary(cv_model)
best_lambda <- cv_model$lambda.min
best_lambda
best_model <- glmnet(x,Lyft2$adjusted_price, alpha = 1, lambda = best_lambda)
summary(best_model)
coef(best_model)

Uber<-read.csv("Cleaned_Uber_Dataset.csv")
summary(Uber)

drops<- c("company","new_time_stamp", "surge_multiplier","price","cab_type","X")
names(Uber) %in% drops
!( names(Uber) %in% drops )
Uber2 <- Uber[,!(names(Uber) %in% drops)]
summary(Uber2)

x1<-model.matrix(adjusted_price ~ .,data = Uber2)
library(glmnet)
cv_model2 <- cv.glmnet(x1,Uber2$adjusted_price, alpha = 1)
summary(cv_model2)
best_lambda2<- cv_model2$lambda.min
best_lambda2
best_model2<- glmnet(x1,Uber2$adjusted_price, alpha = 1, lambda = best_lambda2)
summary(best_model2)
coef(best_model2)

#K-Fold Cross Validation

library(randomForest)
library(glmnet)
library(Metrics)

### a list of variables that we are trying to exclude because they are irrelevant to the prediction
drops<- c("company","new_time_stamp", "surge_multiplier","price","cab_type","X")

### clean Lyft dataset
Lyft<-read.csv("Cleaned_Lyft_Dataset.csv")
summary(Lyft)
names(Lyft) %in% drops
!( names(Lyft) %in% drops )
Lyft2 <- Lyft[,!(names(Lyft) %in% drops)]
summary(Lyft2) # final Lyft dataset

### clean Uber dataset
Uber<-read.csv("Cleaned_Uber_Dataset.csv")
summary(Uber)
names(Uber) %in% drops
!( names(Uber) %in% drops )
Uber2 <- Uber[,!(names(Uber) %in% drops)]
summary(Uber2) # final Uber dataset

### K Fold Cross Validation
###
###  1. Lyft

## create a vector of fold memberships (random order)
nfold <- 10
n <- nrow(Lyft2)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

## create an empty dataframe of results (linear, linear with interaction terms, random forest)
OOS <- data.frame(linear=rep(NA,nfold), linint=rep(NA, nfold), rf=rep(NA,nfold))

## Use a for loop to run through the nfold trails
for (k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  # Linear
  lyft.linear <- glm(adjusted_price~., data=Lyft2, subset=train)
  # Linear Interaction
  lyft.linint <- glm(adjusted_price~.^2, data=Lyft2, subset=train)
  # Random Forest
  lyft.rf <- randomForest(adjusted_price~., data=Lyft2, subset=train, ntree=550, keep.forest=TRUE, importance=TRUE)
  
  ## get predictions: type=response so we have probabilities
  lyft.pred.linear <- predict(lyft.linear, newdata=Lyft2[-train,], type="response")
  lyft.pred.linint <- predict(lyft.linint, newdata=Lyft2[-train,], type="response")
  lyft.pred.rf <- predict(lyft.rf, newdata=Lyft2[-train,], type="response")
  
  ## calculate rmse
  # Linear
  OOS$linear[k] <- rmse(Lyft2$adjusted_price[-train], lyft.pred.linear)
  OOS$linear[k]
  
  # Linear Interaction
  OOS$linint[k] <- rmse(Lyft2$adjusted_price[-train], lyft.pred.linint)
  OOS$linint[k]
  
  # Random Forest
  OOS$rf[k] <- rmse(Lyft2$adjusted_price[-train], lyft.pred.rf)
  OOS$rf[k]
  
  ## We will loop this nfold times (I setup for 10)
  ## this will print the progress (iteration that finished)
  print(paste("Iteration",k,"of",nfold,"completed"))
}

## Lets list the mean of the results stored in the dataframe OOS
## we have nfold values in OOS for each model, this computes the mean of them
colMeans(OOS)
m.OOS <- as.matrix(OOS)
rownames(m.OOS) <- c(1:nfold)
barplot(t(as.matrix(OOS)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0.5),
        ylab= bquote( "Out of Sample " ~ RMSE), xlab="Fold", names.arg = c(1:10))


## If you kept at least 10 folds, we can plot a box blot 
## so see how OOS RMSE fluctuates across fold
if (nfold >= 10){
  ## This plots a box plot with the performance of the three models
  names(OOS)[1]
  ## Lets zoom in  to see better the performance of 
  ## the small and the null model
  boxplot(OOS, col="plum", las = 2, ylab=expression(paste("OOS ",RMSE)), xlab="", main="10-fold Cross Validation")
  names(OOS)[1] <-"Linear regression"
}

## Lasso
## First lets set up the data for it
## the features need to be a matrix ([,-1] removes the first column which is the intercept)
Mx<- model.matrix(adjusted_price~.,data=Lyft2)[,-1]
My<- Lyft2$adjusted_price
## This defined the features we will use the matrix Mx (X) and the target My (Y)
##
## Lasso requires a penalty parameter lambda
lasso <- glmnet(Mx,My)
## By running for all values of lambda we get many models back
## now we have the summary
summary(lasso)

## Now that we can actually compute the Lasso for all values of lambda,
## the whole "path" of models can be evaluated by a OOS experiment
## we can attempt to use cross validation to actually pick lambda.
## the following command yields a cross validation procedure
## the following command takes some time.
lassoCV <- cv.glmnet(Mx,My)

## There are some rules that people like to use:
## The minimum of the mean values stored in lambda.min
## if we had to compute lambda.min we can simply write
lassoCV$lambda[which.min(lassoCV$cvm)]

## Post Lasso
## we select a model based on the proposed rules and perform cross validation

## The code is the same as the CV before so I removed the comments
PL.OOS <- data.frame(PL.min=rep(NA,nfold))
L.OOS <- data.frame(L.min=rep(NA,nfold))
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)

data.min <- data.frame(Mx[,features.min],My)

for (k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ## This is the CV for the Post Lasso Estimates
  rmin <- glm(My~., data=data.min, subset=train)
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  PL.OOS$PL.min[k] <- rmse(My[-train], predmin)
  
  ## This is the CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], lambda = lassoCV$lambda.min)
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  L.OOS$L.min[k] <- rmse(My[-train], predlassomin)
  
  print(paste("Iteration",k,"of",nfold,"completed"))
}

### RMSE Performance
rmseperformance <- cbind(PL.OOS,L.OOS,OOS)
par( mar=  c(8, 4, 4, 2) + 0.6 )
names(PL.OOS)[1] <-"PL.min"
names(L.OOS)[1] <-"Lasso.min"
names(OOS)[1] <-"Linear"
names(OOS)[2] <-"Linear (interaction)"
names(OOS)[3] <-"Random Forest"

barplot(main='Average OSS RMSE Score by Model (Lyft)',colMeans(rmseperformance), las=2,xpd=FALSE, ylim=c(1,3) , xlab="",ylab = bquote( "Average Out of Sample " ~ RMSE), col=rgb(0.8,0.1,0.1,0.6))
m.OOS <- as.matrix(rmseperformance)
boxplot(rmseperformance, col="plum", las = 2, ylab=expression(paste("OOS ",RMSE)), xlab="", main="10-fold Cross Validation")
colMeans(rmseperformance)

#### Best Lyft Model Summary
best_lyft_model <- randomForest(adjusted_price~., data=Lyft2, ntree=550, keep.forest=TRUE, importance=TRUE)
summary(best_lyft_model)


###
###
###  2. Uber
nfold <- 10
n <- nrow(Uber2)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
## create an empty dataframe of results
OOS <- data.frame(linear=rep(NA,nfold), linint=rep(NA, nfold), rf=rep(NA,nfold))

## Use a for loop to run through the nfold trails
for (k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  #Linear
  uber.linear <- glm(adjusted_price~., data=Uber2, subset=train)
  #Linear Interaction
  uber.linint <- glm(adjusted_price~.^2, data=Uber2, subset=train)
  #Random Forest
  uber.rf <- randomForest(adjusted_price~., data=Uber2, subset=train, ntree=550, keep.forest=TRUE, importance=TRUE)
  
  ## get predictions: type=response so we have probabilities
  uber.pred.linear <- predict(uber.linear, newdata=Uber2[-train,], type="response")
  uber.pred.linint <- predict(uber.linint, newdata=Uber2[-train,], type="response")
  uber.pred.rf <- predict(uber.rf, newdata=Uber2[-train,], type="response")
  
  ## calculate rmse
  # Linear
  OOS$linear[k] <- rmse(Uber2$adjusted_price[-train], uber.pred.linear)
  OOS$linear[k]
  
  # Linear Interaction
  OOS$linint[k] <- rmse(Uber2$adjusted_price[-train], uber.pred.linint)
  OOS$linint[k]
  
  # Random Forest
  OOS$rf[k] <- rmse(Uber2$adjusted_price[-train], uber.pred.rf)
  OOS$rf[k]
  
  ## We will loop this nfold times (I setup for 10)
  ## this will print the progress (iteration that finished)
  print(paste("Iteration",k,"of",nfold,"completed"))
}

### Lets list the mean of the results stored in the dataframe OOS
### we have nfold values in OOS for each model, this computes the mean of them)
colMeans(OOS)
m.OOS <- as.matrix(OOS)
rownames(m.OOS) <- c(1:nfold)
barplot(t(as.matrix(OOS)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=-0.2),
        ylab= bquote( "Out of Sample " ~ RMSE), xlab="Fold", names.arg = c(1:10))

### If you kept at least 10 folds, we can plot a box blot 
### so see how OOS RMSE fluctuates across fold
if (nfold >= 10){
  ### This plots a box plot with the performance of the three models
  names(OOS)[1]
  ### Lets zoom in  to see better the performance of 
  ### the small and the null model
  boxplot(OOS, col="plum", las = 2, ylab=expression(paste("OOS ",RMSE)), xlab="", main="10-fold Cross Validation")
  names(OOS)[1] <-"Linear regression"
}

#### Lets run Lasso
#### First lets set up the data for it
#### the features need to be a matrix ([,-1] removes the first column which is the intercept)
Mx<- model.matrix(adjusted_price~.,data=Uber2)[,-1]
My<- Uber2$adjusted_price
## This defined the features we will use the matrix Mx (X) and the target My (Y)
##
## Lasso requires a penalty parameter lambda
lasso <- glmnet(Mx,My)
## By running for all values of lambda we get many models back
## now we have the summary
summary(lasso)

## Now that we can actually compute the Lasso for all values of lambda,
## the whole "path" of models can be evaluated by a OOS experiment
## we can attempt to use cross validation to actually pick lambda.
## the following command yields a cross validation procedure
## the following command takes some time.
lassoCV <- cv.glmnet(Mx,My)

## There are some rules that people like to use:
## The minimum of the mean values stored in lambda.min
## if we had to compute lambda.min we can simply write
lassoCV$lambda[which.min(lassoCV$cvm)]

## Post Lasso
## we select a model based on the proposed rules and perform cross validation

## The code is the same as the CV before so I removed the comments
PL.OOS <- data.frame(PL.min=rep(NA,nfold))
L.OOS <- data.frame(L.min=rep(NA,nfold))
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)

data.min <- data.frame(Mx[,features.min],My)

for (k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ## This is the CV for the Post Lasso Estimates
  rmin <- glm(My~., data=data.min, subset=train)
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  PL.OOS$PL.min[k] <- rmse(My[-train], predmin)
  
  ## This is the CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], lambda = lassoCV$lambda.min)
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  L.OOS$L.min[k] <- rmse(My[-train], predlassomin)
  
  print(paste("Iteration",k,"of",nfold,"completed"))
}
### RMSE Performance
rmseperformance <- cbind(PL.OOS,L.OOS,OOS)
par( mar=  c(8, 4, 4, 2) + 0.6 )
names(PL.OOS)[1] <-"PL.min"
names(L.OOS)[1] <-"Lasso.min"
names(OOS)[1] <-"Linear"
names(OOS)[2] <-"Linear (interaction)"
names(OOS)[3] <-"Random Forest"

barplot(main='Average OSS RMSE Score by Model (Uber)',colMeans(rmseperformance), las=2,xpd=FALSE, ylim=c(2,3) , xlab="",ylab = bquote( "Average Out of Sample " ~ RMSE), col=rgb(0.8,0.1,0.1,0.6))
m.OOS <- as.matrix(rmseperformance)
boxplot(rmseperformance, col="plum", las = 2, ylab=expression(paste("OOS ",RMSE), xlab="", main="10-fold Cross Validation"))
colMeans(rmseperformance)

#### Best Uber Model Summary
best_uber_model <-  glm(adjusted_price~.^2, data=Uber2)
summary(best_uber_model)




