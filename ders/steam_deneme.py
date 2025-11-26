from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import os
import requests


# burada ana yolda bulunan chrome driverımızı çağırıyoruz
driver = webdriver.Chrome()

# hedeflenen siteyi açıp fullekran yapıyoruz yanı pencereyi dolduran cinsten
site_url = "https://store.steampowered.com/charts/mostplayed"
driver.get(site_url)
driver.maximize_window()

# burada kullanıcının steam ekranında oyunların gelmesini beklemesi için bir input oluşturdum
input("bu noktada konsol biraz dinlenecek ekranda oyunlar kesin kes yüklenince bas")

# burada driverın oyunların fotoğraflarına karşılık gelen elementlerden liste yapmasını istiyorum
games = driver.find_elements(By.CLASS_NAME,"_2dODJrHKWs6F9v9QpgzihO")

# burada ise daha sonra indirme işleminin yapılabilmesi için fotoğrafların kaynak urllerini çıkartıp liste haline getiriyoruz
links = []
for i in games:
    links.append(i.get_attribute("src"))

# burada show amaçlı linkleri yazdırıyoruz
# print(links)

# şimdi tüm bu linkleri bulduğumuza göre fotoğrafları indirip kaydetmeye başlayabiliriz

# resimler bu klasöre kaydedilecektir

indirme_yolu = "veriler//steam"

# klasör varsa devam yoksa oluşturalım

if not os.path.exists(indirme_yolu):
    os.mkdir(indirme_yolu)

print(f"Resimler {indirme_yolu} konumuna indirilecektir.")

# burada indirme döngüsünü yazalım

for index,yol in enumerate(links):
    try:
        # şimdi kaynak linki için bir istek oluşturalım
        response = requests.get(yol,stream=True)
        response.raise_for_status() # http hatalarına karşı istisna fırlatır

        # şimdi for döngüsünde sıradaki fotoğraf için bir dosya adı/yolu yazalım

        dosya = os.path.join(indirme_yolu,f"resim_{index+1}.jpg")

        # dosyayı binary modda açalım ve sitedeki resmin içeriğini 1 ve 0lar olarak aynen aktaralım şimdi

        with open(dosya,"wb") as file:
            # resim dosyasının binary halini 8192 baytlık parçalar halinde yazalım
            for chunk in response.iter_content(8192):
                file.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"hata var = {e}")

print("İndirme işlemi tamamlandı.")


sleep(3)
driver.quit()
