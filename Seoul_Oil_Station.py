from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'UTF-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'UTF-8')

# 드라이버 옵션 설정(파일 저장 경로 지정)
# chrome_options = Options()
# prefs = {
#             "download.default_directory" : r"D:\AtomProject\python\section7\oil_price",
#             "download.prompt_for_download": False,
#             "download.directory_upgrade": True,
#             "safebrowsing.enabled": True
#     }
# chrome_options.add_experimental_option("prefs",prefs)

# 브라우저 드라이버 선언
# browser = webdriver.Chrome(executable_path="D:/AtomProject/python/webdriver/chrome/chromedriver.exe", options = chrome_options)

# 창 크기 조정
# browser.set_window_size(1920, 1080)

# 브라우저 대기
# browser.implicitly_wait(3)

# 목표 홈페이지 접속
# browser.get("http://www.opinet.co.kr")
# sytle 속성이 None일 경우와, 링크가 javascript 문으로 되어있을 경우의 코드
# container = browser.find_element_by_id("gnb_sub")
# browser.execute_script("arguments[0].style.display = 'block';", container)
# browser.execute_script("javascript:goPageNet(0,0,'B2')")

# 서울 지역 선택
# WebDriverWait(browser, 2.5).until(EC.presence_of_element_located((By.XPATH, "//*[@id='SIDO_NM0']"))).click()
# WebDriverWait(browser, 2).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#SIDO_NM0 > option:nth-child(2)"))).click()
# time.sleep(1)

# 서울시 구 리스트 저장
# soup = BeautifulSoup(browser.page_source, 'html.parser')
# gu_list = soup.select('select#SIGUNGU_NM0 > option')
# time.sleep(1)

# 엑셀 저장
# for idx in range(1, len(gu_list)):
    # 구 선택
    # WebDriverWait(browser, 2).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#SIGUNGU_NM0 > option:nth-child({})".format(idx)))).click()
    # time.sleep(2)
    # 엑셀 파일 다운로드
    # browser.execute_script("javascript:fn_excel_download('os_btn');")
    # time.sleep(5)

# browser.quit()
