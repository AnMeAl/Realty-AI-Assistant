import requests
from bs4 import BeautifulSoup
import time
import random

flats = []

def time_delay(min_seconds=3, max_seconds=5):
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)

for i in range(1, 26):
    response = requests.get(f"https://realty.yandex.ru/moskva/snyat/kvartira/?roomsTotal=1&roomsTotal=2&roomsTotal=3&roomsTotal=PLUS_4&page={i}")
    soup = BeautifulSoup(response.text, 'html.parser')
    flats_list = soup.find('div', class_='OffersSerp')
    try:
        for li in flats_list.find_all('li'):
            try:
                flat = {}
                div = li.find('div', class_ = 'OffersSerpItem__main')
                div_info = div.find('div', class_='OffersSerpItem__generalInfo')
                main = div_info.find('div', class_='OffersSerpItem__generalInfoInnerContainer')
                main_data = main.find('a').find('span').find('span').get_text(strip=True).encode('latin1').decode('utf-8').split()
                flat['Площадь'] = float(main_data[0].replace(',', '.'))
                flat['Количество комнат'] = int(main_data[3].split('-')[0])
                flat['Этаж'] = int(main_data[6])
                flat['Количество этажей в доме'] = int(main_data[9])
                            
                price = div.find('div', class_='OfferPriceLabel__priceWithTrend--1_AZI').find('div').find('span')
                price = price.get_text().encode('latin1').decode('utf-8').replace('\xa0', '')
                flat['Цена'] = float(price.replace(',', '.'))
                                
                location = main.find('div', class_='OffersSerpItem__location')
                try:
                    adress = location.get_text(strip=True).encode('latin1').decode('utf-8')
                    try:
                        flat['Адрес'] = adress.split('.')[1]
                    except IndexError:
                        flat['Адрес'] = adress
                except UnicodeDecodeError:
                    continue
                images = []
                try:
                    a = li.find('a')
                    for div in a.find_all('div', class_='Gallery__item'):
                        img = div.find('img')
                        srcset = img.get('srcset')
                        parts = srcset.split(',')
                        for part in parts:
                            part = part.strip()
                            if ' 2x' in part:
                                url = (part.split(' 2x')[0].strip())[2:]
                            else:
                                url = (parts[0].split()[0].strip())[2:]
                        images.append(url)
                except AttributeError:
                    continue
                flat['Изображения'] = images
                
                try:
                    description = div_info.find('p', class_='OffersSerpItem__description')
                    flat['Описание'] = description.get_text(strip=True).encode('latin1').decode('utf-8', errors='ignore')
                except AttributeError:
                    flat['Описание'] = ''
            except AttributeError:
                continue
            flats.append(flat)
    except AttributeError:
        pass
    time_delay()