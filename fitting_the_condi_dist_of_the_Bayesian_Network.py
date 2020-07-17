import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.discretiser import Discretiser
from causalnex.network import BayesianNetwork
from causalnex.evaluation import classification_report, roc_auc
from causalnex.inference import InferenceEngine
import time


data = pd.read_csv('./student/student-por.csv', delimiter=';')
data.head(5)

drop_col = ['school', 'sex', 'age', 'Mjob', 'Fjob', 'reason', 'guardian']
data = data.drop(columns=drop_col)
data.head(5)

structure_data = data.copy()
non_numeric_columns = list(structure_data.select_dtypes(exclude=[np.number]).columns)

le = LabelEncoder()
for col in non_numeric_columns:
    structure_data[col] = le.fit_transform(structure_data[col])

structure_data.head(5)

# 이번엔 numeric값도 사용하기 위해 numeric을 categorical 데이터로 변경
discretised_data = data.copy()

data_vals = {col: data[col].unique() for col in data.columns}

# causalnex의 데이터 변환 map함수에 쓸 조건 dictionary 정의
failures_map = {v: 'no-failure' if v == [0]
            else 'have-failure' for v in data_vals['failures']}

studytime_map = {v: 'short-studytime' if v in [1,2]
                 else 'long-studytime' for v in data_vals['studytime']}

# 데이터를 dictionary와 mapping시키면 범주값으로 바뀜.
discretised_data["failures"] = discretised_data["failures"].map(failures_map)
discretised_data["studytime"] = discretised_data["studytime"].map(studytime_map)


# 범주값에 따라 레이블 숫자를 매겨준다.
discretised_data["absences"] = Discretiser(
    method="fixed",
    numeric_split_points=[1, 10]).transform(discretised_data["absences"].values)  # x<1: 0, 1<=x<10: 1, x>=10: 2

discretised_data["G1"] = Discretiser(
    method="fixed",
    numeric_split_points=[10]).transform(discretised_data["G1"].values)  # x<10: 0, x>=10: 1

discretised_data["G2"] = Discretiser(
    method="fixed",
    numeric_split_points=[10]).transform(discretised_data["G2"].values)

discretised_data["G3"] = Discretiser(
    method="fixed",
    numeric_split_points=[10]).transform(discretised_data["G3"].values)


# 다른 컬럼의 mapping dictionary (값이 범주값이기만 하면 되어서 꼭 str로 바꾸지 않아도 괜찮긴 하다)
absences_map = {0: "No-absence", 1: "Low-absence", 2: "High-absence"}

G1_map = {0: "Fail", 1: "Pass"}
G2_map = {0: "Fail", 1: "Pass"}
G3_map = {0: "Fail", 1: "Pass"}

discretised_data["absences"] = discretised_data["absences"].map(absences_map)
discretised_data["G1"] = discretised_data["G1"].map(G1_map)
discretised_data["G2"] = discretised_data["G2"].map(G2_map)
discretised_data["G3"] = discretised_data["G3"].map(G3_map)


# 데이터 분할
train, test = train_test_split(discretised_data, train_size=0.9, test_size=0.1, random_state=7)


# 데이터 구조 모델 (2~3분 소요)
start = time.time()

sm = from_pandas(structure_data)
sm.remove_edges_below_threshold(0.8)

sm = from_pandas(structure_data, tabu_edges=[("higher", "Medu")], w_threshold=0.8)
sm.add_edge("failures", "G1")
sm.remove_edge("Pstatus", "G1")
sm.remove_edge("address", "G1")

sm = sm.get_largest_subgraph()

end = time.time() - start
print(int(end))


# 베이지안 네트워크 모델 선언
bn = BayesianNetwork(sm)
bn = bn.fit_node_states(discretised_data)

# 조건부 확률 분포 (CPDS: Conditional Probability Distributions) 핏팅
bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

# 타겟 확인
print(bn.cpds["G1"])  # 시험 G1 성적 - Pass/Fail

# 타겟을 제외한 인풋(18번째 row) 확인
print(discretised_data.loc[18, discretised_data.columns != 'G1'])


# 예측
predictions = bn.predict(discretised_data, "G1")
print('The prediction is \'{prediction}\''.format(prediction=predictions.loc[18, 'G1_prediction']))
print('The ground truth is \'{truth}\''.format(truth=discretised_data.loc[18, 'G1']))

# 평가
classification_report(bn, test, "G1")

roc, auc = roc_auc(bn, test, "G1")
print(auc)


# 한계(Marginal) 확률 베이스라인 (위와 같음)
bn = bn.fit_cpds(discretised_data, method="BayesianEstimator", bayes_prior="K2")

# 모든 상태와 노드에 대해서 한계(Marginal) 우도(Likelihood) 계산
ie = InferenceEngine(bn)
marginals = ie.query()
print('Marginal Likelihood of Target: ', marginals["G1"])

# 실제 레이블 개수 분포를 세어서 계산한 우도와 비슷한지 확인
labels, counts = np.unique(discretised_data["G1"], return_counts=True)
list(zip(labels, counts))


# 학습시간 변수 각각의 경우(레이블)에 대해서 한계 확률 계산해보기
marginals_short = ie.query({"studytime": "short-studytime"})
marginals_long = ie.query({"studytime": "long-studytime"})
print("Marginal G1 | Short Studtyime", marginals_short["G1"])
print("Marginal G1 | Long Studytime", marginals_long["G1"])

"""
Marginal G1 | Short Studtyime {'Fail': 0.2776556433482524, 'Pass': 0.7223443566517477}
Marginal G1 | Long Studytime {'Fail': 0.15504850337837614, 'Pass': 0.8449514966216239}
=> 공부를 더 많이 한 경우 G1 시험 통과 확률이 더 높다.
"""

# Higher 변수(고등 교육 선호도)에 Intervention(개입, 조작, 조종) 수행
# -> 임의로 해당 변수의 분포를 통제(도메인 지식 개입))
# 여기서는 모두가 고등 교육을 선호할 것이다라고 가정하고 해당 변수의 분포를 강제로 변경
print("distribution before do", ie.query()["higher"])
ie.do_intervention("higher",
                   {'yes': 1.0,
                    'no': 0.0})
print("distribution after do", ie.query()["higher"])
"""
distribution before do {'no': 0.10752688172043011, 'yes': 0.8924731182795698}
distribution after do {'no': 0.0, 'yes': 0.9999999999999998}
=> higher 변수를 임의로 marginal까지 조정하여 강제로 변화시킨 결과
"""
# Intervention을 다시 되돌리고 싶을 때
ie.reset_do("higher")


# 변수에 Intervention을 수행한 후 확률 계산 변화
print("marginal G1", ie.query()["G1"])
ie.do_intervention("higher",
                   {'yes': 1.0,
                    'no': 0.0})
print("updated marginal G1", ie.query()["G1"])

"""
marginal G1 {'Fail': 0.25260687281677224, 'Pass': 0.7473931271832277}
updated marginal G1 {'Fail': 0.20682952942551894, 'Pass': 0.7931704705744809}
=> higher를 marginal까지 조정한 뒤 G1의 결과 변화 예상.
=> 고등 교육을 선호할 수록 시험에 통과할 확률이 높아진다라고 예측.
"""