# Predicting sentiment using SageMaker XGBoost on IMDB movie reviews dataset
Materials are from Udacity Machine Learning Engineer Nanodegree Program
https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t


## Mini-Project: Building Your First Model

https://youtu.be/ouLvRqMMbbY
https://youtu.be/utUxiW-tZrY

If you look at the deployment Gitub repository, inside of the Mini-Projects folder is a notebook called IMDB Sentiment Analysis - XGBoost (Batch Transform).ipynb. Inside of the notebook are some tasks for you to complete.

As you progress through the notebook you will construct an XGBoost model that tries to determine the sentiment, positive or negative, of a movie review using the IMDB Dataset. Moving forward, most of the mini-projects that you will complete throughout this module will use the IMDB dataset.

Note: For the most part, creating this XGBoost model is pretty similar to the Boston Housing example that we just looked at so you can look there if you get stuck. In addition, a solution has been provided and in the next video we will go over my solution to this notebook.

## Deploying and Using a Sentiment Analysis Model

https://youtu.be/r7XVQEojRKk

You've learned how to create and train models in SageMaker and how you can deploy them. In this example we are going to look at how we can make use of a deployed model in a simple web app.

In order for our simple web app to interact with the deployed model we are going to have to solve a couple problems.

The first obstacle is something that has been mentioned earlier.

The endpoint that is created when we deploy a model using SageMaker is secured, meaning that only entities that are authenticated with AWS can send or receive data from the deployed model. This is a problem since authenticating for the purposes of a simple web app is a bit more work than we'd like.

So we will need to find a way to work around this.

The second obstacle is that our deployed model expects us to send it a review after it has been processed. That is, it assumes we have already tokenized the review and then created a bag of words encoding. However, we want our user to be able to type any review into our web app.

We will also see how we can overcome this.

To solve these issues we are going to need to use some additional Amazon services. In particular, we are going to look at Amazon Lambda and API Gateway.

In the mean time, I would encourage you to take a look at the IMDB Sentiment Analysis - XGBoost - Web App.ipynb notebook in the Tutorials folder. In the coming videos we will go through this notebook in detail, however, each of the steps involved is pretty well documented in the notebook itself.


#### Text Processing

https://youtu.be/A7M1z8yLl0w

I mentioned that one of our tasks will be to convert any user input text into data that our deployed model can see as input. You've seen a few examples of text pre-processing and the steps usually go something like this:

Get rid of any special characters like punctuation
Convert all text to lowercase and split into individual words
Create a vocabulary that assigns each unique word a numerical value or converts words into a vector of numbers
This last step is often called word tokenization or vectorization.

And in the next example, you'll see exactly how I do these processing steps; I'll also be vectorizing words using a method called bag of words. If you'd like to learn more about bag of words, please check out the video below, recorded by another of our instructors, Arpan!

Bag of Words

You can read more about the bag of words model, and its applications, [on this page](https://en.wikipedia.org/wiki/Bag-of-words_model). It's a useful way to represent words based on their frequency of occurrence in a text.

#### Building and Deploying the Model

https://youtu.be/JCiQhhXbeuc

To begin with, we are going to extend the mini-project that you worked on in the last lesson by deploying it. There are a couple of changes made to the way that data is processed in this version and the reason for this is to simplify some of what follows.

For the most part, however, we simply add on an extra deployment step to the sentiment analysis mini-project and then test that our deployed endpoint is working correctly.

Once this is done we know that we have a sentiment analysis model that has been trained, is performing well and is working, a great place to start!

Don't forget to SHUT DOWN your endpoint!

#### How to use a Deployed Model

https://youtu.be/WTwj-7XcTro

As mentioned earlier, there are two obstacles we are going to need to overcome. The first is the security issue and the second is data processing. The way that we are going to approach solving these issues is by making use of Amazon Lambda and API Gateway.

The structure for our web app will look like the diagram below.

!img[Diagram](https://github.com/austinlasseter/xgboost-sentiment-analysis/blob/master/code/tutorials/Web%20App%20Diagram.svg "Simple Web App Data Path")

What this means is that when someone uses our web app, the following will occur.

To begin with, a user will type out a review and enter it into our web app.

Then, our web app will send that review to an endpoint that we created using API Gateway. This endpoint will be constructed so that anyone (including our web app) can use it.

API Gateway will forward the data on to the Lambda function

Once the Lambda function receives the user's review, it will process that review by tokenizing it and then creating a bag of words encoding of the result. After that, it will send the processed review off to our deployed model.

Once the deployed model performs inference on the processed review, the resulting sentiment will be returned back to the Lambda function.

Our Lambda function will then return the sentiment result back to our web app using the endpoint that was constructed using API Gateway.

Don't forget!
Currently our endpoint is running. The reason for this is that in the next few videos we are going to interact with our deployed endpoint. If you are following along, don't forget that your endpoint is running. If you need to take a break, don't forget to shut down your endpoint!

#### Creating and Using Endpoints

You've just learned a lot about how to use SageMaker to deploy a model and perform inference on some data. Now is a good time to review some of the key steps that we've covered. You have experience processing data and creating estimators/models, so I'll focus on what you've learned about endpoints.

An endpoint, in this case, is a URL that allows an application and a model to speak to one another.

!img[Diagram](https://github.com/austinlasseter/xgboost-sentiment-analysis/blob/master/code/tutorials/endpoints.png "Endpoints")

**Endpoint steps**
You can start an endpoint by calling .deploy() on an estimator and passing in some information about the instance.
xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')
Then, you need to tell your endpoint, what type of data it expects to see as input (like .csv).
    from sagemaker.predictor import csv_serializer

xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
Then, perform inference; you can pass some data as the "Body" of a message, to an endpoint and get a response back!
response = runtime.invoke_endpoint(EndpointName = xgb_predictor.endpoint,   # The name of the endpoint we created
                                       ContentType = 'text/csv',                     # The data format that is expected
                                       Body = ','.join([str(val) for val in test_bow]).encode('utf-8'))
The inference data is stored in the "Body" of the response, and can be retrieved:

response = response['Body'].read().decode('utf-8')
print(response)
Finally, do not forget to shut down your endpoint when you are done using it.

    xgb_predictor.delete_endpoint()

#### Building a Lambda Function

https://youtu.be/jOXETK4AerU

In general, a Lambda function is an example of a 'Function as a Service'. It lets you perform actions in response to certain events, called triggers. Essentially, you get to describe some events that you care about, and when those events occur, your code is executed.

For example, you could set up a trigger so that whenever data is uploaded to a particular S3 bucket, a Lambda function is executed to process that data and insert it into a database somewhere.

One of the big advantages to Lambda functions is that since the amount of code that can be contained in a Lambda function is relatively small, you are only charged for the number of executions.

In our case, the Lambda function we are creating is meant to process user input and interact with our deployed model. Also, the trigger that we will be using is the endpoint that we will create using API Gateway.

Create a Lambda Function
The steps to create a lambda function are outlined in the notebook and here, for convenience.

Setting up a Lambda function The first thing we are going to do is set up a Lambda function. This Lambda function will be executed whenever our public API has data sent to it. When it is executed it will receive the data, perform any sort of processing that is required, send the data (the review) to the SageMaker endpoint we've created and then return the result.

**Part A: Create an IAM Role for the Lambda function**

Since we want the Lambda function to call a SageMaker endpoint, we need to make sure that it has permission to do so. To do this, we will construct a role that we can later give the Lambda function.

**Part B: Create a Lambda function**

Now it is time to actually create the Lambda function. Remember from earlier that in order to process the user provided input and send it to our endpoint we need to gather two pieces of information:

-- The name of the endpoint, and the vocabulary object.
-- We will copy these pieces of information to our Lambda function, after we create it.

/code/tutorials/lambdafunction.py

#### Building an API

https://youtu.be/AzBQ-aDQSG4

At this point we've created and deployed a model, and we've constructed a Lambda function that can take care of processing user data, sending it off to our deployed model and returning the result. What we need to do now is set up some way to send our user data to the Lambda function.

The way that we will do this is using a service called API Gateway. Essentially, API Gateway allows us to create an HTTP endpoint (a web address). In addition, we can set up what we want to happen when someone tries to send data to our constructed endpoint.

In our application, we want to set it up so that when data is sent to our endpoint, we trigger the Lambda function that we created earlier, making sure to send the data to our Lambda function for processing. Then, once the Lambda function has retrieved the inference results from our model, we return the results back to the original caller.

#### Using the Final Web Application

https://youtu.be/VgG41Q_a15I

Now we get to reap the rewards of all our hard work, we get to deploy our web app!

The back end of our app has been set up so at this point all we need to do is finish up the user facing portion, the website itself. To do this we just need to tell our website where it should send data to.

Don't forget!
In order for our web app to work, we need to have our model deployed. This means that we are incurring a cost. So, once you have finished playing with your newly created web app, make sure to shut it down!

You may also want to clean up the endpoint that you constructed and the Lambda function. This isn't too important, however, since each of these services only incur a cost when used.

Some notes on Lambda and Gateway usage
For Lambda functions you are only charged per execution, which for this class will be very few and still within the free tier. Deleting a lambda function is just a good cleanup step; you won't be charged if you just leave it there (without executing it). Similarly, for APIs created using API Gateway you are only charged per request, and the number of requests we require in this course should still fall under the free tier.
