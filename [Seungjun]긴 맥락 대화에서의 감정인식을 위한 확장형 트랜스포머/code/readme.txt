Description

긴 맥락 대화에서의 감정인식을 위한 확장형 트랜스포머 소프트웨어

긴 맥락으로 구성된 대화에서 감정인식을 위해 발화 및 단어 간의 관계를 분석하는 어텐션을 분리하고 
입력 크기를 늘려 긴 대화에서 감정인식을 가능하게 하는 모델을 통해 학습 비용을 줄이면서 
긴 대화에서의 맥락 정보를 고려한다.

Environment

requirements.txt 참조

Files

requirements.txt - 환경
data.py - dataset load 코드
loss.py - loss 코드
model.py - model 코드
preprocessor.py - raw 데이터 전처리 코드, 
train.py - 모델 훈련 코드
test.py - 모델 평가 코드
util.py - 각종 라이브러리, 예) min-max, mel spectogram

Usage

0. requirements.txt 참고하여 환경 설정
1. python preprocessor --data_pth '데이터 경로' - 데이터 경로에 raw 데이터 전처리 생성
2. python train.py - 학습 진행
3. python test.py - 평가 진행