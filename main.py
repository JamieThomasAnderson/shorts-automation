import praw
import os
import json
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def reddit_init():

    with open('reddit_keys.json', 'r') as keys:
        data = json.load(keys)
    
    client_id = data['client_id']
    client_secret = data['client_secret']
    user_agent = data['user_agent']

    reddit = praw.Reddit(
            client_id = client_id,
            client_secret = client_secret,
            user_agent = user_agent
            )
    
    return reddit

def load_seen():

    seen = []

    if os.path.isfile('seen.pkl'):
        with open('seen.pkl', 'rb') as f:
            seen = pickle.load(f)

    return seen

def dump_seen(seen):

    with open('seen.pkl', 'wb') as f:
        pickle.dump(seen, f)

def get_post(reddit, sub="confessions", time_filter="all"):
    
    seen = load_seen()

    BATCH_SIZE = 10

    subreddit = reddit.subreddit(sub)

    for post in subreddit.top(time_filter=time_filter, limit=BATCH_SIZE):
        if post.id not in seen:
            seen.append(post.id)
            dump_seen(seen)
            return post.title, post.url, post.selftext.strip().replace("\n", "")
        BATCH_SIZE *= 2

def get_image(post_url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(post_url)

reddit = reddit_init()
post = get_post(reddit)
get_image(post[1])
