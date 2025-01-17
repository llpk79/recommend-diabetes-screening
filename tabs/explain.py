import dash_core_components as dcc
import dash_html_components as html


layout = [dcc.Markdown("""
This model was produced using data from behavioral health surveys taken in King County, WA from 2011-2017. 
You can have fun with your own copy by going 
[here](https://www.doh.wa.gov/DataandStatisticalReports/DataSystems/BehavioralRiskFactorSurveillanceSystemBRFSS).

After organizing the data into something our model can understand, it's important to evaluate the model and 
determine if it's useful.

One way to see if it's useful is to look at the predictions it makes. We can classify each prediction as a 
true or false positive or a true or false negative. Plotting the true positive rate against the false positive rate 
gives us a useful tool to visualize how well the model is doing. A perfect model's blue line would make a perfect 
triangle with the red line. A model that is no better than random guessing would put the blue line on top of the 
red line which is given as a baseline for comparison."""),

          html.Iframe(width="800", height="500", src="//plot.ly/~pkutrich/32.embed"),
          dcc.Markdown("""
The area under the blue tells us how well the model can use the variations between the examples of people's 
information it has to tell if a given person is a diabetic or not. For this model the area under the blue line 
is 0.7733. A perfect model would score a 1.0.

The next step is examining the model's features and looking at how helpful each feature is in making a prediction. 
To do this we run the model multiple times, filling in one column with random noise and comparing the results with 
each other. Doing so and plotting the results gives you something like this."""),

          html.Iframe(width="800", height="500", src="//plot.ly/~pkutrich/34.embed"),
          dcc.Markdown("""
After trimming the features with zero or negative explanatory power, we run the model again and test for the area 
under the blue line again. We also check to see that the features we kept are indeed useful according to the model.

`[0]	validation_0-auc:0.783355	validation_1-auc:0.770201
XGBRFClassifier validation ROC AUC: 0.7702009468059364`"""),

          html.Iframe(width="800", height="500",  src="//plot.ly/~pkutrich/37.embed"),
          dcc.Markdown("""
So far we've made all of our prediction directly from the model. We can dig in to see what probability the model 
gives for each prediction being positive or negative and take a look whether we might adjust that to improve our 
guesses.

Given that our goal is to effectively allocate resources we want to balance false positives, people we're sending 
to get poked and prodded who won't end up being diabetic, with false negatives, people who are really diabetic that 
we're not recommending screening for."""),
          html.Iframe(width="900", height="500", src="//plot.ly/~pkutrich/27.embed"),
          ]

