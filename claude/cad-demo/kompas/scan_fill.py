"""дыры скана (непросканированные места) закрываются до сборки по сечениям: открытый край даёт
разорванные контуры сечений, и стенка с карманами строится по обрывкам"""
import numpy as np, trimesh, networkx as nx
def fill_loops(m, max_span=0.5):
    """закрыть дыры скана: каждый замкнутый контур края — веер треугольников из его центра"""
    m = m.copy()
    b = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
    if len(b) == 0: return m, 0
    # ориентированные рёбра края (как в своих гранях) — чтобы новый веер шёл с правильной стороны
    E = m.edges[b]
    G = nx.DiGraph(); G.add_edges_from([tuple(e) for e in E])
    V = list(m.vertices); F = list(m.faces); nf = 0
    diag = np.linalg.norm(m.extents)
    for comp in nx.weakly_connected_components(G):
        sub = G.subgraph(comp)
        if any(sub.out_degree(v) != 1 or sub.in_degree(v) != 1 for v in sub): continue  # не простой контур
        cyc = [next(iter(comp))]
        while True:
            nxt = next(iter(sub.successors(cyc[-1])))
            if nxt == cyc[0]: break
            cyc.append(nxt)
        P = m.vertices[cyc]
        if np.ptp(P, axis=0).max() > max_span * diag: continue  # огромный край — не дыра, а открытая деталь
        c = len(V); V.append(P.mean(0))
        for i in range(len(cyc)):
            a, bb = cyc[i], cyc[(i + 1) % len(cyc)]
            F.append([bb, a, c]); nf += 1
    out = trimesh.Trimesh(np.array(V), np.array(F), process=True)
    return out, nf
