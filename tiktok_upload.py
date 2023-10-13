from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent
from selenium_stealth import stealth

ua = UserAgent()

with open("filename.txt", "r") as file:
	title = file.read()

profile_path = "./google-chrome"

options = webdriver.ChromeOptions()
# Your existing options
options.add_argument("start_maximized")
options.add_argument(f"--user-data-dir={profile_path}")
options.add_argument("--headless")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options)

stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
        )

driver.get("https://tiktok.com/creator-center/upload")
file_path = "/home/shorts/shorts_generator/video.mp4"

time.sleep(20)
driver.switch_to.frame(0)
upload_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
upload_input.send_keys(file_path)

driver.switch_to.default_content()

time.sleep(500)

tags = ['#reddit', '#redditstories']

driver.switch_to.frame(0)
caption_div = driver.find_element(By.XPATH, '//div[@aria-label="Caption"]//div[@contenteditable="true"]')
caption_div.click()
title = title + " "

# caption_div.send_keys(title + " ")

for i in range(20):
    ActionChains(driver).send_keys(Keys.BACKSPACE).perform()
    time.sleep(1)

for letter in title:
    ActionChains(driver).send_keys(letter).perform()
    time.sleep(2)

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


