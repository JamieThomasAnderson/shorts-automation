import praw
import os
import time
import json
import pickle
import re
import random
import openai
import textwrap
from nltk.tokenize import sent_tokenize
from post import post_to_tiktok

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import subprocess
import asyncio
import edge_tts
from edge_tts import VoicesManager
import pysubs2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2

from PIL import Image
from moviepy import *
from moviepy.editor import *
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import concatenate_audioclips, AudioFileClip

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

def clean_post(post_text):
    post_text = re.sub(r'http\S+', '', post_text)
    post_text = re.sub(r'[^a-zA-Z0-9.,\s]', '', post_text)
    post_text = post_text.strip().replace("\n", "")
    return post_text
    

def get_post(reddit, sub="confession", time_filter="all"):
    
    seen = load_seen()

    BATCH_SIZE = 10

    subreddit = reddit.subreddit(sub)

    while True:
        for post in subreddit.top(time_filter=time_filter, limit=BATCH_SIZE):
            if post.id not in seen and len(post.selftext.split()) < 1600:
                seen.append(post.id)
                dump_seen(seen)
                return post.title, post.url, clean_post(post.selftext)
        BATCH_SIZE *= 2

def get_image(name, username, tweet_content, output_path="post.png"):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1920, 1080)
    driver.get("https://www.tweetgen.com/create/tweet-classic.html")
    time.sleep(10)

    driver.find_element(By.ID, "nameInput").send_keys(name)
    driver.find_element(By.ID, "usernameInput").send_keys(username)
    driver.find_element(By.ID, "tweetTextInput").send_keys(tweet_content)    
    time.sleep(10)

    tweet_container = driver.find_element(By.ID, "tweetInnerContainer")
    tweet_container.screenshot(output_path)

    driver.quit()

def get_voiceover(post_text, speed=1.2, language="en", gender="Male", output_audio="voiceover_slow.mp3", output_subs="voiceover.vtt"):
    async def async_get_voiceover():
        voices = await VoicesManager.create()

        voice = voices.find(Gender=gender, Language=language)

        if not voice:
            print("No voice found for the specified criteria.")
            return

        voice_name = voice[0]["Name"]

        communicate = edge_tts.Communicate(post_text, voice_name)
        submaker = edge_tts.SubMaker()

        with open(output_audio, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

        with open(output_subs, "w", encoding="utf-8") as subs_file:
            subs_file.write(submaker.generate_subs())

    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(async_get_voiceover())
    finally:
        loop.close()
    subprocess.run([
        'ffmpeg', 
        '-y', 
        '-i', 
        output_audio, 
        '-filter:a', 
        f"atempo={speed}", 
        'voiceover.mp3'
        ])


def process_image(img_path):
    img = Image.open(img_path)  
    width, height = img.size

    padding = 5
    padded_img = Image.new('RGBA', (width + padding, height + padding), (0, 0, 0, 0))
    padded_img.paste(img, (padding // 2, padding // 2))
    padded_img.save(img_path)

def process_background(background_path, audio_duration):
    background = VideoFileClip(background_path).loop(duration=audio_duration)

    max_start_time = max(0, background.duration - 120)
    start_time = random.uniform(0, max_start_time)
    background = background.subclip(
            max_start_time,
            start_time + audio_duration
            ).loop(duration=audio_duration)
    return background

def process_captions(t, caption_file, video_duration):
    subtitles = pysubs2.load(caption_file)
    captions = [(event.start / 1000, event.end / 1000, event.text) for event in subtitles]

    height, width = 1920, 1080  # Adjust as per your video dimensions
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = [0, 0, 0]  # RGB - Black color

    # Find the caption for the current time
    for start, end, text in captions:
        if start <= t < end:
            # Create an image using PIL
            img = Image.new('RGB', (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Load a font
            base_font_size = 125
            font = ImageFont.truetype("m.otf", base_font_size)

            words = text.split()
            total_text_height = len(words) * base_font_size
            y = (height - total_text_height) // 2
            word_duration = (end - start) / len(words)
            color = (255, 255, 255)  # White color
            for i, word in enumerate(words):

                word_start = start + i * word_duration
                word_end = word_start + word_duration

                relative_t = max(0, min((t - word_start) / word_duration, 1))
                pop_scale = 1 + 0.04 * (1 - abs(relative_t - 0.5) * 2)  # adjust the 0.2 as needed

                font_size = int(base_font_size * pop_scale)
                font = ImageFont.truetype("m.otf", font_size)

                text_width = draw.textlength(word, font)
                x = (width - text_width) // 2
                draw.text((x, y), word, font=font, fill=color)
                y += base_font_size


            frame = np.array(img)
            break

    return frame

def process_text(post_text):
    data = ""
    with open('gpt-keys.json', 'r') as keys:
        data = json.load(keys)
    
    openai.api_key = data['API_KEY']    
    chunks = [post_text[i:i+2000] for i in range(0, len(post_text), 2000)]
    
    improved_texts = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Please correct the grammar and make the following text more readable for a text-to-speech voice."
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ]
        )

        improved_texts.append(response.choices[0].message['content'])

    post_text = " ".join(improved_texts)
    post_text.strip().replace("\n", "")
    return post_text
       

def apply_opacity(get_frame, t):
    frame = get_frame(t)
    if t < 3:
        return frame * 0
    else:
        return frame

def threshold(frame, thr=0.5):
    mask = (frame > thr).astype(np.uint8) * 255  # Convert 1 to 255 for a white mask
    return np.stack([mask] * 3, axis=-1)

def write_title(post_title):
    with open('filename.txt', 'w') as file:
        file.write(post_title)

def pre_process():
    reddit = reddit_init()
    post_title, post_url, post_text = get_post(reddit)
    post_text = process_text(post_text)
    write_title(post_title)

    get_voiceover(f"{post_title} {post_text}")
    get_image("LetsNotMeet Reddit", "LetsNotMeet", post_title)

    process_image("post.png")

def composite():
    audio = AudioFileClip("voiceover.mp3")
    img = ImageClip("post.png").set_duration(3)
    background = process_background("background.mp4", audio.duration) 

    captions = VideoClip(lambda t: process_captions(t, "voiceover.vtt", audio.duration), duration=audio.duration)
    captions = captions.fl(apply_opacity)
    gray_clip = captions.fl_image(lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY))

    mask_clip = gray_clip.fl_image(lambda frame: threshold(frame))
    isolated_text_clip = mask_clip.set_mask(mask_clip.to_mask())
    isolated_text_clip = isolated_text_clip.set_position("center")
    captions = isolated_text_clip.set_duration(audio.duration)

    video = CompositeVideoClip([background, img.set_position("center"), captions]).set_audio(audio)
    video.write_videofile("video.mp4", codec="libx264", fps=24)

if __name__ == "__main__":
    pre_process()
    composite()
    post_to_tiktok("/home/shorts/shorts_generator/video.mp4")
