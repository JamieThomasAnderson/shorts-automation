import praw
import os
import time
import json
import pickle
import re
import random
import openai
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
from PIL import Image, ImageDraw, ImageFont
import cv2

from PIL import Image
from moviepy import *
from moviepy.editor import *
from moviepy.editor import AudioFileClip


def load_config(config_path="config.json"):
    """
    Load configuration data from a JSON file.
    
    :param config_path: Path to the configuration file.
    :return: Loaded configuration data.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def reddit_init(client_id, client_secret, user_agent):
    """
    Initialize a Reddit instance using PRAW.
    
    :param client_id: Reddit API client ID.
    :param client_secret: Reddit API client secret.
    :param user_agent: User agent string.
    :return: Initialized Reddit instance.
    """
    reddit = praw.Reddit(
        client_id=client_id, client_secret=client_secret, user_agent=user_agent
    )
    return reddit


def load_seen(seen_path="./data/seen.pkl"):
    """
    Load seen posts data from a pickle file.
    
    :param seen_path: Path to the seen posts file.
    :return: List of seen posts.
    """
    seen = []
    if os.path.isfile(seen_path):
        with open("seen.pkl", "rb") as f:
            seen = pickle.load(f)
    return seen


def clean_tmp(tmp_path="./tmp"):
    """
    Remove all files from the temporary directory.
    
    :param tmp_path: Path to the temporary directory.
    """
    for filename in os.listdir(tmp_path):
        file_path = os.path.join(tmp_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def dump_seen(seen, seen_path="./data/seen.pkl"):
    """
    Save seen posts data to a pickle file.
    
    :param seen: List of seen posts.
    :param seen_path: Path to save the seen posts file.
    """
    with open(seen_path, "wb") as f:
        pickle.dump(seen, f)


def clean_post(post_text):
    """
    Clean and format a Reddit post's text.
    
    :param post_text: Original post text.
    :return: Cleaned post text.
    """
    post_text = re.sub(r"http\S+", "", post_text)
    post_text = re.sub(r"[^a-zA-Z0-9.,\s]", "", post_text)
    post_text = post_text.strip().replace("\n", "")
    return post_text

def get_post(reddit, sub="confession", time_filter="all", seen_path="./data/seen.pkl"):
    """
    Retrieve a Reddit post from a specified subreddit that hasn't been seen before.
    
    :param reddit: Initialized Reddit instance.
    :param sub: Name of the subreddit to retrieve posts from.
    :param time_filter: Time filter for top posts ("all", "year", "month", etc.).
    :param seen_path: Path to the seen posts file.
    :return: Tuple containing the post's title, URL, and cleaned text.
    """

    seen = load_seen(seen_path=seen_path)

    BATCH_SIZE = 10

    subreddit = reddit.subreddit(sub)

    while True:
        for post in subreddit.top(time_filter=time_filter, limit=BATCH_SIZE):
            if post.id not in seen and len(post.selftext.split()) < 1000:
                seen.append(post.id)
                dump_seen(seen, seen_path=seen_path)
                return post.title, post.url, clean_post(post.selftext)
        BATCH_SIZE *= 2


def get_image(name, username, tweet_content, output_path="post.png"):
    """
    Generate an image of a simulated Twitter post using TweetGen.
    
    :param name: Display name for the simulated Twitter post.
    :param username: Username for the simulated Twitter post.
    :param tweet_content: Content of the simulated tweet.
    :param output_path: Path to save the generated image.
    """

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


def get_voiceover(
    post_text,
    language="en",
    gender="Male",
    output_audio="voiceover.mp3",
    output_subs="voiceover.vtt",
):
    """
    Generate a voiceover and its corresponding subtitles for the provided text.

    :param post_text: Text content to be converted to voiceover.
    :param language: Language of the voiceover (default is "en").
    :param gender: Gender preference for the voice ("Male" or "Female").
    :param output_audio: Path to save the generated audio file.
    :param output_subs: Path to save the generated subtitles in VTT format.
    """
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
                    submaker.create_sub(
                        (chunk["offset"], chunk["duration"]), chunk["text"]
                    )

        with open(output_subs, "w", encoding="utf-8") as subs_file:
            subs_file.write(submaker.generate_subs())

    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(async_get_voiceover())
    finally:
        loop.close()


def process_image(img_path):
    """
    Add padding to an image.

    :param img_path: Path to the image file to be processed.
    """

    img = Image.open(img_path)
    width, height = img.size

    padding = 5
    padded_img = Image.new("RGBA", (width + padding, height + padding), (0, 0, 0, 0))
    padded_img.paste(img, (padding // 2, padding // 2))
    padded_img.save(img_path)


def process_background(background_path, audio_duration):
    """
    Process a video background to match the duration of an audio clip.

    :param background_path: Path to the video background.
    :param audio_duration: Duration of the audio clip to match.
    :return: Processed video background.
    """
    background = VideoFileClip(background_path).loop(duration=audio_duration)

    max_start_time = max(0, background.duration - 120)
    start_time = random.uniform(0, max_start_time)
    background = background.subclip(max_start_time, start_time + audio_duration).loop(
        duration=audio_duration
    )
    return background


def process_captions(t, caption_file, font):
    """
    Generate a frame with captions for a given timestamp.

    :param t: Timestamp in seconds to generate the caption frame for.
    :param caption_file: Path to the .srt subtitle file.
    :param font: Path to the font file used for rendering the captions.
    :return: An RGB frame with rendered captions.
    """
    subtitles = pysubs2.load(caption_file)
    captions = [
        (event.start / 1000, event.end / 1000, event.text) for event in subtitles
    ]

    height, width = 1920, 1080  # Adjust as per your video dimensions
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = [0, 0, 0]  # RGB - Black color

    # Find the caption for the current time
    for start, end, text in captions:
        if start <= t < end:
            # Create an image using PIL
            img = Image.new("RGB", (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            base_font_size = 125
            font = ImageFont.truetype("m.otf", base_font_size)

            words = text.split()
            total_text_height = len(words) * base_font_size
            y = (height - total_text_height) // 2
            word_duration = (end - start) / len(words)
            color = (255, 255, 255)  # White color
            for i, word in enumerate(words):
                word_start = start + i * word_duration

                relative_t = max(0, min((t - word_start) / word_duration, 1))
                pop_scale = 1 + 0.04 * (1 - abs(relative_t - 0.5) * 2)

                font_size = int(base_font_size * pop_scale)
                font = ImageFont.truetype(font, font_size)

                text_width = draw.textlength(word, font)
                x = (width - text_width) // 2
                draw.text((x, y), word, font=font, fill=color)
                y += base_font_size

            frame = np.array(img)
            break

    return frame


def process_text(post_text, key):
    """
    Refine the grammar and readability of the provided text using OpenAI GPT-3.5.

    :param post_text: The input text to be improved.
    :param key: API key for the OpenAI service (unused in the current implementation).
    :return: Improved text optimized for text-to-speech readability.
    """
    data = ""
    with open("gpt-keys.json", "r") as keys:
        data = json.load(keys)

    openai.api_key = data["API_KEY"]
    chunks = [post_text[i : i + 2000] for i in range(0, len(post_text), 2000)]

    improved_texts = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Please correct the grammar and make the following text more readable for a text-to-speech voice.",
                },
                {"role": "user", "content": chunk},
            ],
        )

        improved_texts.append(response.choices[0].message["content"])

    post_text = " ".join(improved_texts)
    post_text = post_text.strip().replace("\n", "")
    return post_text


def apply_opacity(get_frame, t):
    """
    Apply opacity to a frame based on a given time.

    :param get_frame: Function to retrieve the frame at the given time.
    :param t: Time for which the frame needs to be fetched.
    :return: Adjusted frame with applied opacity.
    """
    frame = get_frame(t)
    if t < 3:
        return frame * 0
    else:
        return frame


def threshold(frame, thr=0.5):
    """
    Apply a threshold to a given frame.

    :param frame: Input frame to be thresholded.
    :param thr: Threshold value (default is 0.5).
    :return: Frame after thresholding.
    """
    mask = (frame > thr).astype(np.uint8) * 255  # Convert 1 to 255 for a white mask
    return np.stack([mask] * 3, axis=-1)


def write_title(post_title, title_path):
    """
    Write the provided title to a file.

    :param post_title: The title to be written.
    :param title_path: Path of the file where the title should be written.
    """
    with open(title_path, "w") as file:
        file.write(post_title)


def pre_process(config):
    """
    Pre-process the data required for video composition.
    
    :param config: Dictionary containing configuration parameters.
    """
    credentials = {
        "client_id": config["client_id"],
        "client_secret": config["client_secret"],
        "user_agent": config["user_agent"],
    }
    reddit = reddit_init(**credentials)

    post_title, post_url, post_text = get_post(
        reddit,
        sub=config["subreddit"],
        time_filter=config["time_filter"],
        seen_path=config["seen_path"],
    )
    if config["use_gpt"]:
        post_text = process_text(post_text, config["gpt_key"])
    write_title(post_title, title_path=config["title_file"])

    get_voiceover(
        f"{post_title} {post_text}",
        language=config["language"],
        gender=config["voice_sound"],
        output_audio=config["voiceover_file"],
        output_subs=config["captions_file"],
    )
    get_image(
        config["img_username"],
        config["img_@"],
        post_title,
        output_path=config["img_file"],
    )
    process_image(config["img_file"])


def composite(config):
    """
    Compose the final video using pre-processed data and assets.
    
    :param config: Dictionary containing configuration parameters.
    """
    audio = AudioFileClip(config["voiceover_file"])
    img = ImageClip(config["img_file"]).set_duration(3)
    background = process_background(config["background_video"], audio.duration)

    captions = VideoClip(
        lambda t: process_captions(t, config["captions_file"], config["font"]),
        duration=audio.duration,
    )
    captions = captions.fl(apply_opacity)
    gray_clip = captions.fl_image(
        lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    )

    mask_clip = gray_clip.fl_image(lambda frame: threshold(frame))
    isolated_text_clip = mask_clip.set_mask(mask_clip.to_mask())
    isolated_text_clip = isolated_text_clip.set_position("center")
    captions = isolated_text_clip.set_duration(audio.duration)

    video = CompositeVideoClip(
        [background, img.set_position("center"), captions]
    ).set_audio(audio)
    video.write_videofile("video.mp4", codec="libx264", fps=24)
    clean_tmp()


if __name__ == "__main__":
    config = load_config()

    pre_process(config)
    composite(config)
    post_to_tiktok(config["video_path"])
