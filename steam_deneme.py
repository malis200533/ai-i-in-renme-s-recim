from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import chromedriver_autoinstaller
from time import sleep
import os
import requests



driver = webdriver.Chrome()
site_url = "https://store.steampowered.com/charts/mostplayed"

driver.get(site_url)
driver.maximize_window()
input("bu noktada konsol biraz dinlenecek ekranda oyunlar kesin kes yüklenince bas")

games = driver.find_elements(By.CLASS_NAME,"_2dODJrHKWs6F9v9QpgzihO")

links = []
for i in games:
    links.append(i.get_attribute("src"))

sleep(3)
driver.quit()