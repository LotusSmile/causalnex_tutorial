import warnings
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
# from IPython.display import Image
# import pygraphviz

warnings.filterwarnings("ignore")


# 도메인 지식으로 만드는 모델
sm = StructureModel()

# 도메인 지식으로 연결한 노드
causal_relationships = [('health', 'absences'), ('health', 'G1')]

# 모델에 엣지 추가
sm.add_edges_from(causal_relationships)


# 시각화
viz = plot_structure(sm, graph_attributes={"scale": "0.5"},
                     all_node_attributes=NODE_STYLE.WEAK,
                     all_edge_attributes=EDGE_STYLE.WEAK)

# Image(viz.draw(format='png'))

# 시각화한 객체를 파일로 저장
viz.draw('res.png')

