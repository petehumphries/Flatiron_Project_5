# Facile Volant
## How we help consumers with the stress of flying

It is a well known pop culture trusim that people find flying to be a stressful endeavour. Comics and sitcom situations abound where a flustered parent tries to organise the family holiday,  or there is some other form of inconvenience or disaster (think the cult comedy Airplane! for example). The issues, however, run deeper than humourous observations of the bureaucratic and confusing nature of air travel. Flight delays can have serious and long lasting impacts on people's lives if left to occur without preventative or predictive measures. Visiting a sick relative, getting to a hospital, or going on a holiday for which one has saved for years. All of these reasons are hugely important, and would be significantly impacted by the stresses of problems with a flight. As mental health issues continue to rise [throughout the world](https://www.theguardian.com/society/2019/jun/03/mental-illness-is-there-really-a-global-epidemic), people need to have as much solidity and information as possible to make sure that they are able to complete their journeys as hassle free as initially possible.

## Data Selection, Cleaning and Project Aim

The aim of this project was to utilise supervised machine learning models to investigate the delays in flights leaving airports. The dataset used was [the kaggle 2015 flight delays and cancellations](https://www.kaggle.com/usdot/flight-delays). In order to make flying a less stressful endeavour for all involved, it is our intention to demonstrate the use that machine learning has in informing conmsumers about the best decisions to make regarding their travel. By inputting current flight data into our model, we will be able to inform whether their flight is like to be delayed. The data set has in excess of five million rows, and as a result we are targeting state by state, lacking the computational power to run such a high volume of rows in one go. We started by working with the state of Wisconsin, as its group of 50,000 data points seemed a good jumping off point.

## Modelling

We chose four models to use, a basic linear regression, K Nearest Neighbours, Random Forest and a Support Vector Machine. Each of these were created as predictors for whether a flight would be delayed on takeoff or not. Each model will take the input of the flight data and inform the individual who makes a query. 

## Conclusion

Given the current state of the flight industry during the current pandemic, we feel that this tool will be invaluable in the coming months as the economic and social world changes. In order for people to be adequately informed of potential issue with their flights, we are offering a service which will allow people easy access to this relevant information. As the quarantine lifts in the future, the air economy could be vastly oversubscribed as people seek to travel after their ban. We hope to offer some level of protection to the consumer once this occurs.
