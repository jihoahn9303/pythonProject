import sys
import io
import pandas as pd
import numpy as np
from glob import glob
import Price_Visualization
from Price_Visualization import Price_Visualize

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'UTF-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'UTF-8')


if __name__ == '__main__':
    # glob.glob(pathname, *, recursive=False) : 서브 디렉토리들을 재귀적으로 탐색하고 싶으면, ** 패턴과 recursive = True 옵션을 사용해야 함.
    # ex) : C:/flower_v1/training-images 디렉토리의 모든 서브 디렉토리에서 모든 jpg 파일을 찾고 싶은 경우
    # -> glob.glob(C:/flower_v1/training-images/**/*.jpg, recursive = True)
    file_list = glob('D:/AtomProject/python/section7/oil_price/지역*.xls')
    # print(file_list)

    # 엑셀 데이터 읽기
    # 데이터프레임 객체 리스트 생성 및 값 저장
    tmp_raw = []
    for file_name in file_list:
        tmp = pd.read_excel(file_name, header = 2, sheet_name = 0)
        tmp_raw.append(tmp)

    # 데이터 가공 단계
    # 데이터프레임 리스트 요소를 하나의 프레임으로 결합
    station_raw = pd.concat(tmp_raw, axis = 0, ignore_index = True)
    # print(station_raw.head())

    # 1. 프레임 재구성
    station = pd.DataFrame(data = { 'Oil_store' : station_raw['상호'], '주소' : station_raw['주소'],
        '가격' : station_raw['휘발유'], '셀프' : station_raw['셀프여부'], '상표' : station_raw['상표']
    })
    # print(station.head())

    # 2. 구 컬럼 추가 -> 구별 주유 가격을 조사하기 위함
    station['구'] = [eachAddress.split()[1] for eachAddress in station['주소']]
    # print(station.head())

    # 구 컬럼 데이터의 종류를 확인
    # print(station['구'].unique())

    # 3. 가격 데이터 가공 -> 가격 컬럼이 '-' 표시가 된 행을 전부 제외
    # print(station[station['가격'] == '-'])
    drop_list = list(station[station['가격'] == '-'].index)
    for idx in drop_list:
        station.drop([idx], axis = 0, inplace = True)
    # print(station[station['가격'] == '-'])

    # 4. 가격 컬럼 데이터를 숫자형으로 변환
    station['가격'] = [float(price) for price in station['가격']]

    # 5. pivot_table()을 이용하여 구별 가격 정보를 얻기
    gu_data = pd.pivot_table(data = station, index=["구"], values = ['가격'], aggfunc = np.mean)
    # print(gu_data.head())
    # print(gu_data.index)

    # 데이터 시각화
    visual_obj = Price_Visualize(dataFrame = station)
    # visual_obj.show_Box_Plus_Swarm()
    # visual_obj.show_Choropleth(dataFrame = gu_data)
    visual_obj.show_Pricemap_TopBottom10()
