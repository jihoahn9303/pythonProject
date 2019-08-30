import matplotlib.pyplot as plt
import seaborn as sns
import platform
import simplejson as json
import folium
import googlemaps
import warnings
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import sys
import io

warnings.simplefilter(action = "ignore", category = FutureWarning)
# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'UTF-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'UTF-8')

# pyplot과 seaborn에서의 한글 폰트 꺠짐 현상 대비 코드
path = "c:/Windows/Fonts/malgun.ttf"
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')

class Price_Visualize():
    def __init__(self, dataFrame):
        # 데이터 프레임 생성
        self.dataFrame = dataFrame

    def show_Boxplot(self):
        plt.figure(figsize = (12, 8))
        sns.boxplot(x = "상표", y = "가격", hue = '셀프', data = self.dataFrame, palette = "Set3")
        plt.show()

    def show_Swarmplot(self):
        plt.figure(figsize = (12, 8))
        sns.swarmplot(x = "상표", y = "가격", data = self.dataFrame, color = ".6")
        plt.show()

    def show_Box_Plus_Swarm(self):
        plt.figure(figsize = (12, 8))
        sns.boxplot(x = "상표", y = "가격", hue = '셀프', data = self.dataFrame, palette = "Set3")
        sns.swarmplot(x = "상표", y = "가격", data = self.dataFrame, color = ".6")
        plt.show()

    def show_Choropleth(self, dataFrame):
        geo_path = 'D:/AtomProject/python/section7/skorea_geo_simple.json'
        # 파일 역직렬화
        geo_data = json.load(open(file = geo_path, mode = 'r', encoding = 'UTF-8'))
        # 지도 생성
        map = folium.Map(location = [37.5502, 126.982], zoom_start = 10.5, tiles= 'Stamen Toner')
        # Choropleth 이식
        folium.Choropleth(geo_data = geo_data, data = dataFrame, columns = [dataFrame.index, '가격'],
                    key_on = 'feature.id', fill_color='PuRd').add_to(map)
        # Choropleth map 저장
        map.save('D:/AtomProject/python/section7/map.html')


    def show_Pricemap_TopBottom10(self):
        oil_price_top10 = self.dataFrame.sort_values(by='가격', ascending=False).head(10)
        # print(oil_price_top10)
        oil_price_bottom10 = self.dataFrame.sort_values(by='가격', ascending=True).head(10)
        # print(oil_price_bottom10)

        # google geocoding API 활성화
        gmap_key = 'Input your google geocoding API key.'
        gmaps = googlemaps.Client(key = gmap_key)

        # google geocoding API를 이용해 각 주유소의 위도 및 경도 값 저장
        lat = []
        lng = []

        # 1. 주유 가격 상위 10개 주유소의 위도 및 경도 값 저장
        for idx in list(oil_price_top10.index):
        # for idx in tqdm_notebook(list(oil_price_top10.index)):
            try:
                tmp_add = str(oil_price_top10.loc[idx:idx, '주소']).split('(')[0]
                tmp_map = gmaps.geocode(tmp_add)
                # print(tmp_map)

                tmp_loc = tmp_map[0].get('geometry')
                lat.append(tmp_loc['location']['lat'])
                lng.append(tmp_loc['location']['lng'])

            except:
                lat.append(np.nan)
                lng.append(np.nan)
                print("Here is nan !")

        oil_price_top10['lat'] = lat
        oil_price_top10['lng'] = lng

        # 2. 주유 가격 하위 10개 주유소의 위도 및 경도 값 저장
        lat = []
        lng = []

        for idx in list(oil_price_bottom10.index):
        # for idx in tqdm_notebook(list(oil_price_bottom10.index)):
            try:
                tmp_add = str(oil_price_bottom10.loc[idx:idx, '주소']).split('(')[0]
                tmp_map = gmaps.geocode(tmp_add)

                tmp_loc = tmp_map[0].get('geometry')
                lat.append(tmp_loc['location']['lat'])
                lng.append(tmp_loc['location']['lng'])

            except:
                lat.append(np.nan)
                lng.append(np.nan)
                print("Here is nan !")

        oil_price_bottom10['lat'] = lat
        oil_price_bottom10['lng'] = lng

        # 위도, 경도 정보를 이용해 주유 가격에 따른 지도 시각화
        map = folium.Map(location=[37.5202, 126.975], zoom_start=10.5)

        for idx in list(oil_price_top10.index):
            if pd.notnull(oil_price_top10['lat'][idx]):
                station_name = oil_price_top10['Oil_store'][idx]
                folium.CircleMarker([oil_price_top10['lat'][idx], oil_price_top10['lng'][idx]],
                                        radius=15, color='#CD3181', popup = station_name,
                                            fill_color='#CD3181', fill=True).add_to(map)

        for idx in list(oil_price_bottom10.index):
            if pd.notnull(oil_price_bottom10['lat'][idx]):
                station_name = oil_price_bottom10['Oil_store'][idx]
                folium.CircleMarker([oil_price_bottom10['lat'][idx], oil_price_bottom10['lng'][idx]],
                                        radius=15, color='#3186cc', popup = station_name,
                                            fill_color='#3186cc', fill=True).add_to(map)

        map.save('D:/AtomProject/python/section7/oil_price_top_bottom_10_map.html')
