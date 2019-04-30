#For each month:

#read a csv file that contains the year, month, # of rain days, and total rain days
	JanData <- read.csv("C:/Users/narut/Desktop/MA431/Model Data-Jan.csv")

#in variable “nonRainDays”, subtract total rain days - rain days
	nonRainDays<-cbind(JanData$RAIN.DAYS, JanData$Total.Days - JanData$RAIN.DAYS)

#views table
	View(nonRainDays)

#In variable “logr1”, use glm function with binomial distribution
	logr1<-glm(nonRainDays~JanData$JAN, family=binomial)

#get summary information of the variable
	summary(logr1)

#Plot the model
	plot(JanData$JAN,fitted.values(logr1))

#Plot the points
	janPoints<-points(JanData$JAN, JanData$RAIN.DAYS/JanData$Total.Days)

#import ggplot package
	library(ggplot2)

#plot trendline
	abline(janPoints)