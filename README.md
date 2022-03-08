# Kaggle-NBME
  1. Kaggle NBME Competition Code
  2. URL : https://www.kaggle.com/c/nbme-score-clinical-patient-notes/overview/evaluation

## 목표
  1. Kaggle NBME 대회 참가
  2. **Token Classification에 대해서 공부하고 모델 성능 고도화해보기**
  3. 모델 개발 뿐만 아니라 전처리 및 후처리를 잘 응용해보기

## 대회 개요
  1. 의사는 환자의 불편하거나 아픈 점에 대해서 기록을 한다.
  2. 각 환자에는 의학적인 의미를 가지는 특징이 존재한다. 
  3. 그렇다면 "기록 내에서 어떠한 부분을 참고해서 그러한 의학적인 의미를 나타낼 수 있는가?"에 대한 답을 자연어 처리 모델이 낼 수 있다.

## 모델이 해야하는 역할
  1. 환자의 기록 그리고 환자의 의학적인 의미를 입력으로 받는다.
  2. 환자의 의학적 의미와 관련된 부분을 환자의 기록 내에서 찾아야 한다.
  3. 단어를 찾는 것이 아니라 환자의 기록에서 "어디에서 어디까지" 즉 구간을 찾아야 한다.

## 구현 방향
  1. 환자의 기록과 환자의 의학적 의미에 해당되는 특징을 하나의 문서로 만든다.
  2. 모델이 문서를 입력으로 받아서 각 토큰별로 0,1로 구분할 수 있게 한다.(TokenClassification)
  3. "0이면 관련 없다, 1이면 관련 있다"으로 생각하고 이를 바탕으로 후처리를 해서 문서의 특정 부분을 검출할 수 있게한다.

## Baseline 구축
  1. offset_mapping을 통한 전처리하기
  2. AutoModelForTokenClassification, AutoConfig, AutoTokenizer을 이용한 대회에 맞는 모델 및 토크나이저 불러오기
  3. 토큰 단위의 결과를 후처리를 통해서 글자단위의 답으로 변경하기
  4. **결과 제출은 csv 파일이 아니라 ipynb로 작성해서 kaggle에 맞춰서 구현하기**

## 사용하는 모델
  1. Base Model : roberta-large 모델 기반의 tokenclassificaion 모델 사용

## Argument
|Argument|Description|Default|
|--------|-----------|-------|
|output_dir|model saving directory|exp|
|logging_dir|logging directory|log|
|dir_path|dataset directory|data|
|PLM|model name(huggingface)|roberta-large|
|model_type|custom model type|base|
|epochs|train epochs|5|
|lr|learning rate|3e-5|
|train_batch_size|train batch size|4|
|eval_batch_size|evaluation batch size|8|
|max_len|input max length|512|
|warmup_steps|warmup steps|200|
|weight_decay|weight decay|1e-3|
|gradient_accumulation_steps|gradient accumulation steps|2|
|eval_ratio|evaluation data ratio|0.2|
|save_steps|model save steps|500|
|logging_steps|logging steps|100|
|eval_steps|model evaluation steps|500|
|seed|random seed|42|
|dotenv_path|wandb & huggingface path|path.env|


## Terminal Command Example
  ```
  # training 
  python train.py --PLM roberta-large --dir_path ./data --output_dir ./exp --epochs 5 --lr 2e-5 --warmup_steps 500 --weight_decay 1e-3 --eval_ratio 0.2
  ```
