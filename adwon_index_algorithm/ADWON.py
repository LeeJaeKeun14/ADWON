
from adwon_index_algorithm.input_data import *
from adwon_index_algorithm.data_analysis import *
from adwon_index_algorithm.prediction import *

class Algorithm():
    def __init__(self):
        pass
    
    def read_data(self):
        """
        데이터베이스의 데이터를 읽은 뒤 csv파일로 저장한다.
        읽은 테이블 raw_tv, raw_mo, raw_pc, spot_foot_traffics
        
        반환값 없음
        """
        
        read_data.new_read() # tv, 모바일, 인터넷, 옥외광고 데이터 read
        read_data.user_read() # 유져 1인당 노출 회수 데이터 read
        
        # 데이터 저장 위치 : "/adwon_index_algorithm/input_data/created_data/"
    
    def read_df(self):
        """
        저장된 csv파일을 읽은 뒤 데이터 프레임 형식으로 반환한다.
        raw_tv, raw_mo, raw_pc, spot_foot_traffics
        
        반환값 df
        """ 
        df = make_temp_data.read_df()
        return df
    
    def print_cost(self):
        """
        가격 기준 데이터 핸들링
        가격 기준으로 성별, 연령별 분류하여 노출된 유져수 카운팅
        
        반환값 가격 기준 데이터 핸들링
        """ 
        df = data_handling.print_cost()
        return df
    
    def handling(self, 연령 = "total"):
        """
        연령 = "(성별)_(연령)"으로 정렬
        예) 연령 = "전체_전체"
        예) 연령 = "남_10대"
        예) 연령 = "여_전체"
        예) 연령 = "전체_60대"
        
        시계열 기준 데이터 핸들링
        데이터를 1주 단위로 분류하여 각 광고별 노출 카운트 계산하여 핸들링을 진행한다.
        
        반환값 시계열 기준 데이터 핸들링
        """
        reg_df = data_handling.handling(연령)
        return reg_df
    
    def print_regression(self, 연령 = "total"):
        """
        연령 = "(성별)_(연령)"으로 정렬
        예) 연령 = "전체_전체"
        예) 연령 = "남_10대"
        예) 연령 = "여_전체"
        예) 연령 = "전체_60대"
        
        시계열 기준 데이터 핸들링을 기반으로 회귀분석을 진행
        진행한 후 summary 결과를 출력
        
        반환값 없음
        """
        data_handling.print_regression(연령)
        
        
    def regression(self,연령 = "total"):
        """
        연령 = "(성별)_(연령)"으로 정렬
        예) 연령 = "전체_전체"
        예) 연령 = "남_10대"
        예) 연령 = "여_전체"
        예) 연령 = "전체_60대"
        
        시계열 기준 데이터 핸들링을 기반으로 회귀분석을 진행
        진행한 후 회귀계수를 index용으로 반환
        
        반환값 |b1, b2, b3, b4|
        """
        p = data_handling.regression(연령)
        return p

    def p_regression(self,연령 = "total"):
        """
        연령 = "(성별)_(연령)"으로 정렬
        예) 연령 = "전체_전체"
        예) 연령 = "남_10대"
        예) 연령 = "여_전체"
        예) 연령 = "전체_60대"
        
        시계열 기준 데이터 핸들링을 기반으로 회귀분석을 진행
        진행한 후 회귀계수를 예측 알고리즘 용으로 반환
        
        반환값 |b0, b1, b2, b3, b4|
        """

        p = data_handling.p_regression(연령)
        return p
    
    def total_regression(self):
        """
        시계열 기준 데이터 핸들링을 기반으로 회귀분석을 진행
        진행한 후 회귀계수를 모든 연령, 모든 성별로 index용으로 반환
        
        반환값 df
        """
        
        df = data_handling.total_regression()
        return df
    
    def Mnb(self):
        """
        확률론적 생성모형을 이용하여 
        유져 한명의 다운로드 확률을 예측한 후 모형을 
        pickle 형태로 저장
        """
        data_handling.Mnb()
        
    def print_analysis(self):
        df = analysis.print_analysis()
        
        return df
    
    def print_handling(self):
        df = analysis.print_handling()
        
        return df
    
    def print_analysis2(self):
        df = analysis.print_analysis2()
        
        return df
    
    def print_TV(self):
        df = analysis.print_TV()
        
        return df
    
    def print_모바일(self):
        df = analysis.print_TV()
        
        return df
    
    def print_인터넷(self):
        df = analysis.print_TV()
        
        return df
    
    def print_옥외광고(self):
        df = analysis.print_TV()
        
        return df
