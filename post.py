from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent
from selenium_stealth import stealth
import time


def post_to_tiktok(file_path, profile_path="./google-chrome", caption_file="filename.txt"):
    # Initialize user agent
    ua = UserAgent()

    # Read caption from file
    with open(caption_file, "r") as file:
        title = file.read()

    # Set browser options
    options = webdriver.ChromeOptions()
    options.add_argument("start_maximized")
    options.add_argument(f"--user-data-dir={profile_path}")
    options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)

    # Apply stealth settings
    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
            )

    # Open TikTok upload page
    driver.get("https://tiktok.com/creator-center/upload")

    time.sleep(20)
    driver.switch_to.frame(0)
    upload_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
    upload_input.send_keys(file_path)

    driver.switch_to.default_content()

    time.sleep(250)

    tags = ['#reddit', '#redditstories']

    driver.switch_to.frame(0)
    caption_div = driver.find_element(By.XPATH, '//div[@aria-label="Caption"]//div[@contenteditable="true"]')
    caption_div.click()

    # Clear existing caption
    for i in range(20):
        ActionChains(driver).send_keys(Keys.BACKSPACE).perform()
        time.sleep(1)

    # Type in the caption
    for letter in title:
        ActionChains(driver).send_keys(letter).perform()
        time.sleep(2)

    # Add tags
    for tag in tags:
        ActionChains(driver).send_keys(tag).perform()
        time.sleep(2)
        ActionChains(driver).send_keys(Keys.RETURN).perform()

    time.sleep(30)
    post_button = driver.find_element(By.XPATH, '//button[div/div[text()="Post"]]')
    post_button.click()

    time.sleep(20)

    manage_button = driver.find_element(By.XPATH, "//*[text()='Manage your posts']")
    manage_button.click()

    print("FINISHED")

    time.sleep(30)

    driver.quit()
