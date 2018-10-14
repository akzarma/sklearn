import json
import re
import random
import numpy
from matplotlib import pyplot
import tweepy
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tweepy import OAuthHandler
from textblob import TextBlob

tweets = []


class TwitterClient(object):

    def __init__(self):

        consumer_key = '#'
        consumer_secret = '#'
        access_token = '#'
        access_token_secret = '#'

        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):

        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w+:\ / \ / \S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):

        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count=10):
        global tweets

        try:
            fetched_tweets = self.api.search(q=query, count=count)

            for tweet in fetched_tweets:
                parsed_tweet = dict()

                parsed_tweet['text'] = tweet.text
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            return tweets

        except tweepy.TweepError as e:
            print("Error  here: " + str(e))


def main():
    api = TwitterClient()

    topics = ['BJP', 'Congress']
    positive_tweets = []
    negative_tweets = []
    neutral_tweets = []

    for topic in topics:

        tweets = api.get_tweets(query=topic, count=2000)

        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        num_ptweets = 100 * len(ptweets) / len(tweets)
        print("Positive tweets percentage: {} %".format(num_ptweets))
        positive_tweets.append(num_ptweets)

        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        num_ntweets = 100 * len(ntweets) / len(tweets)
        print("Negative tweets percentage: {} %".format(num_ntweets))
        negative_tweets.append(num_ntweets)

        print(
            "Neutral tweets percentage: {} % ".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))
        neutral_tweets.append(100 - num_ntweets - num_ptweets)

        print("\n\nPositive tweets:")
        for tweet in ptweets[:len(ptweets)]:
            print("tweet +ve : ", tweet['text'])

        print("\n\nNegative tweets:")
        for tweet in ntweets[:len(ntweets)]:
            print("tweet -ve : ", tweet['text'])

        # legend = ['Positive Tweets', 'Negative Tweets', 'Neutral Tweets']
        # sizes = [num_ntweets, num_ptweets, (100-num_ptweets-num_ntweets)]
        # color = ['green', 'red', 'yellow']
        # pyplot.pie(sizes, labels=legend, colors=color,
        #         autopct='%1.1f%%', shadow=True, startangle=140)
        # pyplot.axis('equal')
        # pyplot.tight_layout()
        # pyplot.title(topic)
        # pyplot.show()

    # pyplot.hist([positive_tweets,negative_tweets,neutral_tweets], color=['green','red','yellow'])
    legend = ['Positive Tweets', 'Negative Tweets', 'Neutral Tweets']
    # pyplot.bar(topics, positive_tweets, alpha=0.5, label='Negative Tweets', color='green')
    # pyplot.bar(topics, negative_tweets, alpha=0.5, label='Negative Tweets', color='red')
    # pyplot.bar(topics, neutral_tweets, alpha=0.5, label='Neutral Tweets', color='yellow')
    # pyplot.xlabel('Frequency(percentage)')
    # pyplot.ylabel('Classes')
    # pyplot.title('Percentage of different tweets')
    # pyplot.legend(legend,loc='upper right')

    # pyplot.show()


if __name__ == "__main__":
    main()




#============================================================================================================




#========================================================TEST 2====================================================



    pre_data = open('train_data_pos_neg.bin', 'r')
    pre_data_labels = open('train_data_labels_pos_neg.bin', 'r')

    data = json.load(pre_data)
    data_labels = json.load(pre_data_labels)

    print('Now with LogisticRegression')


    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,

    )

    global tweets
    data = data
    tweets_data =   [tweet['text'] for tweet in tweets]
    features = vectorizer.fit_transform(
        data
    )
    test_vectors = vectorizer.transform(tweets_data)


    from sklearn.model_selection import train_test_split


    features_nd = features.toarray()  # for easy usage
    # print(features_nd.dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.99,
    shuffle=False)


    # clean before after
    # confusion
    # compare logistic
    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(test_vectors)
    # classifier_rbf = svm.SVC()
    # classifier_rbf.fit(features_nd, test_vectors.toarray())
    # y_pred = classifier_rbf.predict(test_vectors)

    # print(y_pred, X_test, '\n')
    import random

    j = ((test_vectors.shape[0]))
    for i in range(j):
        print(y_pred[i])
        # ind = features_nd.tolist().index(test_vectors.toarray()[i].tolist())
        print(tweets_data[i].strip())

    # from sklearn.metrics import accuracy_score, confusion_matrix
    #
    # print('\n=================================\nAccuracy of LogisticRegression: ' + str(accuracy_score(y_test, y_pred)))
    #
    # print('\n==================================\nConfusion Matrix: \n' + str(confusion_matrix(y_test, y_pred)) +
    #       '\n\n\n================================================\n\n\n\n\n\n')