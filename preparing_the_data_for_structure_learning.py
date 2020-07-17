import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
import time


# 여러 feature와 시험성적의 관계 데이터 셋 불러오기
data = pd.read_csv('./student/student-por.csv', delimiter=';')
data.head(5)

# 제거할 컬럼
drop_col = ['school', 'sex', 'age', 'Mjob', 'Fjob', 'reason', 'guardian']

# 컬럼 제거
data = data.drop(columns=drop_col)
data.head(5)

# 범주값 데이터만 사용
structure_data = data.copy()
non_numeric_columns = list(structure_data.select_dtypes(exclude=[np.number]).columns)

# 범주값에 따라 레이블숫자 붙여주기 (계산을 위해서는 범주레이블을 숫자로 변환해야함)
le = LabelEncoder()
for col in non_numeric_columns:
    structure_data[col] = le.fit_transform(structure_data[col])

structure_data.head(5)


# 앞의 structure 모델처럼 선언 후 도메인 지식으로 노드를 연결하지 않고
# 데이터에서 구조를 학습 (2~3분 소요)
start = time.time()

sm = from_pandas(structure_data)

# 엣지 가중치 0.8 이하는 제거한다. (변경 가능)
sm.remove_edges_below_threshold(0.8)

# 모델 구축시, 터부 엣지(제거 대상)를 설정할 수 있으며 제거 대상의 가중치 값도 설정 가능.
sm = from_pandas(structure_data, tabu_edges=[("higher", "Medu")], w_threshold=0.8)

# 데이터 기반 모델이어도 도메인 지식으로 엣지를 추가, 제거 가능함
sm.add_edge("failures", "G1")
sm.remove_edge("Pstatus", "G1")
sm.remove_edge("address", "G1")

# 생성된 서브그래프 중 가장 큰 것만 사용
sm = sm.get_largest_subgraph()

# 시각화
viz = plot_structure(sm, graph_attributes={"scale": "0.5"},
                     all_node_attributes=NODE_STYLE.WEAK,
                     all_edge_attributes=EDGE_STYLE.WEAK)

# 저장
viz.draw('res.png')

end = time.time() - start
print(int(end))