# RAGAS Metrics 분석 보고서


## 1. LLMContextRecall

 - LLMContextRecall은 사용자 질문에 대해 모델이 검색한 정보가 얼마나 정확히 참조 답변에 포함된 주장(claim)을 포함하고 있는지 평가
    - LLM을 사용하여 참조 답변(reference answer)의 각 주장이 검색된 컨텍스트(retrieved context)에 포함되어 있는지를 확인 
    - 이 metric은 0과 1 사이의 값을 가지며, 1에 가까울수록 검색된 정보가 참조 답변을 잘 포함하고 있음을 의미

### 예시 코드

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall

# 평가할 샘플 생성
sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",  # 사용자 질문
    response="The Eiffel Tower is located in Paris.",  # 모델의 응답
    reference="The Eiffel Tower is located in Paris.",  # 참조 답변
    retrieved_contexts=["Paris is the capital of France."],  # 검색된 컨텍스트
)

# LLM 기반의 Context Recall 계산
context_recall = LLMContextRecall()
await context_recall.single_turn_ascore(sample)
```

  - 예를 들어, "The Eiffel Tower is located in Paris."라는 참조 답변이 주어졌을 때, 모델이 "Paris is the capital of France."라는 검색 결과를 제시한 경우, 어느 정도 관련성이 있다고 평가

## 2. FactualCorrectness
  - FactualCorrectness는 모델의 응답이 얼마나 사실에 부합하는지 평가
    -  검색된 컨텍스트나 참조 정답과의 관련성을 확인해 모델이 부정확하거나 오해를 일으킬 수 있는 정보 제공 X

예시 코드
```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FactualCorrectness

sample = SingleTurnSample(
    user_input="What is the capital of France?",
    response="The capital of France is Paris.",
    reference="Paris is the capital of France."
)

factual_correctness = FactualCorrectness()
await factual_correctness.single_turn_ascore(sample)
```


## 3. NonLLMContextRecall
  - NonLLMContextRecall은 LLMContextRecall과 유사한 방식으로 컨텍스트 회수 능력을 평가하지만, LLM 대신 문자열 비교와 같은 간단한 기준 사용

예시 코드
```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

sample = SingleTurnSample(
    retrieved_contexts=["Paris is the capital of France."],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris."
    ]
)

context_recall = NonLLMContextRecall()
await context_recall.single_turn_ascore(sample)
```
  - 위 코드에서 NonLLMContextRecall은 검색된 컨텍스트가 참조 컨텍스트에 포함된 주장을 얼마나 잘 회수했는지 비교
