# 
## 1. 모델 미세조정
미세조정을 하기 위해 다음 코드를 터미널에 입력한다.


    sh scripts/run.sh


**명령어 입력 후 모델의 학습 진행은 google-bert, skt-bert, kc-bert 폴더의 train.log에서 확인 가능**

## 2. 미세조정된 모델 평가
평가를 위해서는 학습된 모델의 파라미터 path값을 인자로 넘겨줘야 합니다.
Ex)


    python nsmc_eval.py --model-path 'skt_bert/bert_2022.12.28_21.46.43/gpt2_step_1200.pth' --output-dir 'skt_bert' --tokenizer_name 'monologg/kobert' --model_name 'monologg/kobert'

원하는 모델 명, 토크나이저 이름, 저장위치, path 값을 /script/run_eval.sh에 입력 후에 밑의 명령어를 터미널에 입력한다.


    sh scirpt/run_eval.sh

**명령어 입력 후 모델의 평가 결과는 google-bert, skt-bert, kc-bert 폴더의 valid.log에서 확인 가능**

### 3. configs.py
위 py파일은 하이퍼 파라미터를 위한 파일이다.

### 4. /dataload/semti_data.py
NSMC의 데이터를 받아와 전처리(패딩, 어텐션마스킹)를 진행하는 파일이다.

### 5. Naver Sentiment Analysis

* Dataset : <https://github.com/e9t/nsmc>

| Model                                                                                               | Accuracy                                                        |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [BERT base multilingual cased](https://github.com/google-research/bert/blob/master/multilingual.md) | 86.2140                                                           |
| KoBERT                                                                                              | 89.4760 |
| KcBERT                                                         | **[89.7520]**                                                    |


감상문 AI 프로젝하면서 재밌었다.
좋은 기회 마련해주신 나승훈 교수님을 비롯한 조교들, 강사님들 감사합니다.
