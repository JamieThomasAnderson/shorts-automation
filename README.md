# What is this?
Short-form video platforms, like YouTube Shorts and TikTok, often feature Reddit TTS videos: [example](https://www.tiktok.com/@silly_reddit/video/7277221299498470702?is_from_webapp=1&sender_device=pc&web_id=7283333285886477826). 

Breaking them down, they feature 1) an image of the post that lasts a few seconds and 2) centred captions keeping pace with the TTS voice. I developed the above script to mimic these videos, scraping Reddit, compositing, rendering and uploading in a headless server environment. The script uses several popular Python libraries that make it possible: `moviepy`, `selenium`, `openai`, `edge_tts`. Using OpenAI's ChatGPT may seem unnecessary. But, when the TTS voice directly transcribes post content, poor grammar can significantly degrade video quality. While it's likely these Reddit creators are using some automation techniques - as making them manually at high frequency is tedious - I wanted to try it myself.

### Warning
Script is still untested - very preliminary state. 

### Setup

1) Install [Python](https://www.python.org/downloads/)
2) Follow Instructions:
```
git clone https://github.com/JamieThomasAnderson/shorts-automation
cd ./shorts-automation
pip install -r requirements.txt
```

3) Add `client_id`, `client_secret`, and `user_agent` to config.json
4) Add OpenAI Authentication to config.json (or, set `use_gpt` to false)
5) While many authentication techniques exist for Selenium, TikTok has strict login blocking, so instead of automating the login for every upload or logging in once and storing the cookies, I've just set the Selenium TikTok upload script to use the Google Chrome profile path. To configure this, find where your Google Chrome profile and change the config accordingly (or copy and paste it to the project directory).
6) run `main.py`
```
python3 main.py
```
