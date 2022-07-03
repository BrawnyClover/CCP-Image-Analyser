import urllib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from urllib.request import urlretrieve
from getpass import getpass
import time, os


def login(ID,PW):
    driver = webdriver.Chrome()
    driver.get("https://instagram.com/")
    username = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, "username")))
    username.send_keys(ID)
    password = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, "password")))
    password.send_keys(PW)
    password.send_keys(Keys.ENTER)

    # WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, '//button[text()]="나중에 하기"]/..'))).click()
    # WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME,"HoLwm"))).click()
    print("로그인완료")

    return driver

def find_tag(driver, TagName):
    driver.get(f"https://www.instagram.com/explore/tags/{TagName}/")

def get_posturl(driver,scroll_num=1):
    url_list = []
    for i in range(scroll_num):
        elems=driver.find_elements_by_xpath('//*[contains(@href, "/p/")]')

        for elem in elems:
            url_list.append(elem.get_attribute("href"))
        
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(4)

    url_list = list(set(url_list))

    return url_list

def get_imgurl(driver,url):
    driver.get(url)
    time.sleep(3)
    photo_list = []

    try:
        for i in range(11):
            elems = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//div[@class='KL4Bh' and @style]/img[contains(@src,1080) or contains(@alt,'Photo') or contains(@src,'scontent')]")))

            for elem in elems:
                photo_list.append(elem.get_attribute("src"))
        
            driver.find_element_by_xpath("//button[contains(@class, '6CZ')]/div").click()
            print(f'{i+1}번째 옆으로가기')

    except:
        photo_list = list(set(photo_list))
        print(f'현재 포스트에서 {len(photo_list)}개의 사진주소 수집')
        if len(photo_list)==0:
            print(url)
        return photo_list

def save_photo(driver, url_list,folder='photo',filename='a'):
    photo_list=[]

    print("이미지 주소 수집 시작")
    for url in url_list:
        photo_list += get_imgurl(driver,url)
    print("이미지 주소 수집 끝")

    if not os.path.exists(folder):
        print("저장 폴더 생성")
        os.mkdir(folder)
    
    print("이미지 저장 시작")
    for i, photo in enumerate(photo_list):
        if photo != None:
            urllib.request.urlretrieve(photo,f'./{folder}/{filename}{i+1}.jpg')
        else:
            pass
    print(f"{len(photo_list)}개 사진저장 끝")

if __name__ == "__main__":
    ID = input("아이디를 입력해주세요: ")
    PW = getpass("비밀번호를 입력후 엔터: ")
    
    driver = login(ID, PW)

    tag = input("검색할 태그를 입력해주세요: ")
    find_tag(driver, tag)

    url_list = get_posturl(driver,scroll_num=2)
    save_photo(driver,url_list,folder=tag,filename=tag)

    driver.quit()