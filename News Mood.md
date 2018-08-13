

```python
# Dependencies
import pandas as pd
import tweepy
import time
import json
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from config import consumer_key, consumer_secret, access_token, access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
sources = ['@BBC', '@CBS', '@CNN', '@FoxNews', '@nytimes']
listofdicts = []
#Loop through sources
for x in sources:
    # Target User Account
    target_user = x
    i = 0
    # Loop through 5 pages of tweets (total 100 tweets)
    for y in range(1, 6):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(target_user, page=y)

        # Loop through all tweets
        for tweet in public_tweets:

            tweetdict = {}

            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])

            # Add each value to temporary dictionary
            tweetdict['source'] = target_user
            tweetdict['content'] = tweet['text']
            tweetdict['date'] = time.strftime('%#m-%#d-%Y', time.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
            tweetdict['compound'] = results["compound"]
            tweetdict['pos'] = results["pos"]
            tweetdict['neu'] = results["neu"]
            tweetdict['neg'] = results["neg"]
            tweetdict['tweetsago'] = i
            i += 1
            #Add copy of dictionary to list
            listofdicts.append(tweetdict.copy())
print(len(listofdicts))
```

    500
    


```python
#Convert list of dictionaries into dataframe
df = pd.DataFrame(listofdicts)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
      <th>content</th>
      <th>date</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>source</th>
      <th>tweetsago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>🎬✨ @itsanitarani learns the secrets behind Bol...</td>
      <td>8-13-2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.3182</td>
      <td>Ballet-loving Pollyana lost her leg as a toddl...</td>
      <td>8-13-2018</td>
      <td>0.099</td>
      <td>0.901</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2500</td>
      <td>Follow 26-year-old Teleri Fielden as she pursu...</td>
      <td>8-13-2018</td>
      <td>0.000</td>
      <td>0.895</td>
      <td>0.105</td>
      <td>@BBC</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>The US pack dogs taking on the rhino poachers ...</td>
      <td>8-13-2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>From the Arctic to chocolate, here are some th...</td>
      <td>8-13-2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Reorder columns
df = df[['source', 'tweetsago', 'content', 'date', 'compound', 'pos', 'neu', 'neg']]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>tweetsago</th>
      <th>content</th>
      <th>date</th>
      <th>compound</th>
      <th>pos</th>
      <th>neu</th>
      <th>neg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>0</td>
      <td>🎬✨ @itsanitarani learns the secrets behind Bol...</td>
      <td>8-13-2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBC</td>
      <td>1</td>
      <td>Ballet-loving Pollyana lost her leg as a toddl...</td>
      <td>8-13-2018</td>
      <td>-0.3182</td>
      <td>0.000</td>
      <td>0.901</td>
      <td>0.099</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBC</td>
      <td>2</td>
      <td>Follow 26-year-old Teleri Fielden as she pursu...</td>
      <td>8-13-2018</td>
      <td>0.2500</td>
      <td>0.105</td>
      <td>0.895</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBC</td>
      <td>3</td>
      <td>The US pack dogs taking on the rhino poachers ...</td>
      <td>8-13-2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBC</td>
      <td>4</td>
      <td>From the Arctic to chocolate, here are some th...</td>
      <td>8-13-2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Print dataframe to CSV
df.to_csv('MediaTweets.csv', index=False)
```


```python
#Display and print scatter plot
plt.figure(figsize=(15,10))
plt.title('Sentiment Analysis of Media Tweets 8/13/2018', fontsize = 30)
plt.xlabel('Tweets Ago', fontsize = 20)
plt.ylabel('Polarity Score', fontsize = 20)
color_gen = cycle(('lightblue', 'green', 'red', 'blue', 'yellow'))
#Loop through scources and plot each one with it's own color
for z in sources:
    plt.scatter(df[df.source == z]['tweetsago'], 
                df[df.source == z]['compound'], 
                c=next(color_gen),
                label=z)
plt.legend(loc='best')
plt.savefig('Scatter.png')
```


![png](output_5_0.png)



```python
#Group tweets by source and find mean scores
df2 = df.groupby('source').mean()
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweetsago</th>
      <th>compound</th>
      <th>pos</th>
      <th>neu</th>
      <th>neg</th>
    </tr>
    <tr>
      <th>source</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>49.5</td>
      <td>0.123694</td>
      <td>0.08806</td>
      <td>0.86847</td>
      <td>0.04346</td>
    </tr>
    <tr>
      <th>@CBS</th>
      <td>49.5</td>
      <td>0.139859</td>
      <td>0.08090</td>
      <td>0.89036</td>
      <td>0.02874</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>49.5</td>
      <td>-0.128398</td>
      <td>0.06468</td>
      <td>0.83172</td>
      <td>0.10360</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>49.5</td>
      <td>-0.034312</td>
      <td>0.06524</td>
      <td>0.84791</td>
      <td>0.08681</td>
    </tr>
    <tr>
      <th>@nytimes</th>
      <td>49.5</td>
      <td>-0.066002</td>
      <td>0.06138</td>
      <td>0.85516</td>
      <td>0.08345</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Display and print bar chart
plt.figure(figsize=(15,10))
plt.title('Sentiment Analysis of Media Tweets 8/13/2018', fontsize = 30)
plt.ylabel('Average Polarity Score', fontsize = 20)
plt.bar(df2.index, df2['compound'], width = 1, color = ['lightblue', 'green', 'red', 'blue', 'yellow'], align='center')
plt.savefig('Bar.png')
```


![png](output_7_0.png)

