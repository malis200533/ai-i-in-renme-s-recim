from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from time import sleep

import os
import subprocess
import requests


driver = webdriver.Chrome()

driver.get("https://www.google.com/?hl=tr")
driver.maximize_window()
sleep(1)

while True:
    sev = input("Merhaba indirme işlemine başlamak için indirmek istediğiniz sayfayı açın ve entera basın buradan. Ya da exit yaz çık= ")
    if sev == "exit" or sev == "Exit":
        break
    os.system("cls")

    sayi = int(input("Kaç fotoğraf indirmek istersiniz: "))

    if sayi < 6:
        print("Sayi en az 6 olmalıydı.")
        sleep(1)
        break

    linkler = []

    while True:
        if len(linkler) >= sayi:
            break

        fotolar = driver.find_elements(By.CLASS_NAME,"YQ4gaf")

        yeni_linkler = []
        for element in fotolar:
            resim_gen = int(element.get_attribute("width"))
            if resim_gen >= 50:
                yeni_linkler.append(element.get_attribute("src"))
        
        tam_liste = linkler+yeni_linkler
        tam_liste = list(set(tam_liste))

        if len(tam_liste) == len(linkler):
            break
        
        linkler = tam_liste
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        WebDriverWait(driver,10).until(ec.presence_of_all_elements_located((By.CLASS_NAME,"YQ4gaf")))
    
    if len(linkler) > sayi:
        linkler = linkler[0:sayi-1]

    os.system("cls")

    print(f"İndirilecek resim sayısı {len(linkler)}")
    input("Onaylamak için enter tuşuna basın.")

    os.system("cls")

    yol = input("Lütfen resimlerin kaydedileceği klasörün adını girin: ")

    ana_yol = "google_foto//"

    ana_yol = ana_yol + yol

    if not os.path.exists(ana_yol):
        os.mkdir(ana_yol)

    for index, urlr in enumerate(linkler):
        try:
            response = requests.get(urlr,stream=True)
            response.raise_for_status()

            dosya = os.path.join(ana_yol,f"{yol}{index+1}.jpg")

            with open(dosya,"wb") as file:
                for part in response.iter_content(8192):
                    file.write(part)
        except:
            print("hatadır lo")

    os.system("cls")

    temiz_yol = ana_yol.replace('//', "\\")
    os.system(f"start {temiz_yol}")

    print("Tüm indirme işlemi tamamlandı yeni döngü için entera basınız.")
    input()


# driver.execute_script("window.scrollBy(0, 600);")

driver.quit()
