from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent
from selenium_stealth import stealth

import time
import json


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def driver_init(profile_path):
    """
    Initializse a Selenium Chrome webdriver with specific options and configurations.

    This function sets up the Chrome webdriver with specified browser options for
    automation tasks. Additionally, it applies stealth settings to make the browser
    automation less detectable by anti-bot mechanisms.

    Parameters:
    - profile_path (str): Path to the Chrome user profile directory. Defaults to "./google-chrome".

    Returns:
    - WebDriver: A configured instance of the Chrome webdriver.

    Example:
    driver = driver_init("/path/to/custom-profile")
    """

    options = webdriver.ChromeOptions()
    options.add_argument("start_maximized")
    options.add_argument(f"--user-data-dir={profile_path}")
    options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)

    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    return driver


def upload_video(driver, file_path):
    """
    Uploads a video using the provided Selenium Chrome webdriver.

    Parameters:
    - driver (WebDriver): Initialized Chrome webdriver.
    - file_path (str): Path to the video file.

    Assumes the driver is already on the correct webpage.
    """

    time.sleep(20)

    driver.switch_to.frame(0)
    upload_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
    upload_input.send_keys(file_path)
    driver.switch_to.default_content()

    time.sleep(250)  # Wait for upload


def write_title(driver, title, tags):
    """
    Inputs a title and predefined tags into a caption field on a TikTok upload form.

    Parameters:
    - driver (WebDriver): Initialized Chrome webdriver.
    - title (str): The text to be written as the video's caption.

    Assumes the driver is already on the correct webpage and at the caption input stage.
    """

    tags = ["#reddit", "#redditstories"]

    driver.switch_to.frame(0)
    caption_div = driver.find_element(
        By.XPATH, '//div[@aria-label="Caption"]//div[@contenteditable="true"]'
    )
    caption_div.click()

    for existing in range(20):
        ActionChains(driver).send_keys(Keys.BACKSPACE).perform()
        time.sleep(1)

    for letter in title:
        ActionChains(driver).send_keys(letter).perform()
        time.sleep(1)

    for tag in tags:
        ActionChains(driver).send_keys(tag).perform()
        time.sleep(1)
        ActionChains(driver).send_keys(Keys.RETURN).perform()


def post_video(driver):
    """
    Posts the uploaded video on TikTok and navigates to the 'Manage your posts' page.

    Parameters:
    - driver (WebDriver): Initialized Chrome webdriver.

    Assumes the driver is already on the correct webpage and at the final posting stage.
    """

    time.sleep(30)
    post_button = driver.find_element(By.XPATH, '//button[div/div[text()="Post"]]')
    post_button.click()

    time.sleep(20)
    manage_button = driver.find_element(By.XPATH, "//*[text()='Manage your posts']")
    manage_button.click()
    driver.quit()


def post_to_tiktok(
    file_path, profile_path="./google-chrome", caption_file="filename.txt"
):
    """
    Automate the process of posting a video to TikTok using Selenium.

    Parameters:
    - file_path (str): Path to the video file to be uploaded.
    - profile_path (str, optional): Path to the Chrome profile. Defaults to "./google-chrome".
    - caption_file (str, optional): Path to the file containing the video caption. Defaults to "filename.txt".

    Uses helper functions to initialize the driver, upload the video, set the title, and post the video.
    """

    config = load_config()

    FILE_PATH = config["file_path"]
    PROFILE_PATH = config["profile_path"]
    TITLE_FILE = config["title_file"]
    TAGS = config["tags"]
    URL = config["url"]

    with open(TITLE_FILE, "r") as file:
        title = file.read()

    driver = driver_init(PROFILE_PATH)
    driver.get(URL)

    upload_video(driver, FILE_PATH)
    write_title(driver, title, TAGS)
    post_video(driver)

    driver.quit()
