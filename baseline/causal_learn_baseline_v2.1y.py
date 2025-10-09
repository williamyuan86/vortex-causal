# causal_learn_benchmark_robust_v2_optimized.py
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import average_precision_score
import traceback
from sklearn.preprocessing import StandardScaler
# -------------------------------
# 动态导入 causal-learn
# -------------------------------
available = {}
pc = fci = rfci = ges = degs = None
DirectLiNGAM = ICALiNGAM = None
notears_linear = None
fisherz = gsq = None

try:
    from causallearn.search.ConstraintBased.PC import pc
    available['PC'] = True
except Exception as e:
    available['PC'] = False
    print("PC import failed:", e)

try:
    from causallearn.search.ConstraintBased.FCI import fci
    available['FCI'] = True
    try:
        from causallearn.utils.cit import fisherz, gsq
    except Exception:
        fisherz = gsq = None
except Exception as e:
    available['FCI'] = False
    print("FCI import failed:", e)

try:
    from causallearn.search.ScoreBased.GES import ges
    available['GES'] = True
except Exception as e:
    available['GES'] = False
    print("GES import failed:", e)

try:
    from causallearn.search.ConstraintBased.RFCI import rfci
    available['RFCI'] = True
except Exception:
    available['RFCI'] = False

try:
    from causallearn.search.ScoreBased.Deges import degs
    available['DEGES'] = True
except Exception:
    available['DEGES'] = False

try:
    from causallearn.search.FCMBased.lingam import DirectLiNGAM, ICALiNGAM
    available['DirectLiNGAM'] = True
    available['ICALiNGAM'] = True
except Exception:
    try:
        from causallearn.search.FCMBased.lingam import DirectLiNGAM
        available['DirectLiNGAM'] = True
        available['ICALiNGAM'] = False
    except Exception:
        available['DirectLiNGAM'] = False
        available['ICALiNGAM'] = False

try:
    from causallearn.search.ScoreBased.notears import notears_linear
    available['NOTEARS'] = True
except Exception:
    available['NOTEARS'] = False

print("可用算法：", {k: v for k, v in available.items() if v})

# -------------------------------
# 读取数据
# -------------------------------
from pgmpy.readwrite import BIFReader

file_path = "../data/sachs_data.csv"
bif_path  = "../data/sachs_true_graph.bif"

data = pd.read_csv(file_path)
reader = BIFReader(bif_path)
model = reader.get_model()

bif_nodes = [str(n).upper() for n in model.nodes()]
bif_edges = [(str(u).upper(), str(v).upper()) for (u, v) in model.edges()]

data_cols_upper = [c.upper() for c in data.columns]
if len(data.columns) == len(bif_nodes):
    try:
        col_order = [data_cols_upper.index(n) for n in bif_nodes]
        X_raw = data.values[:, col_order]
        print("成功将 data 列按 BIF 节点顺序对齐。")
    except ValueError:
        X_raw = data.values
        print("警告: data 列名无法按 BIF 节点名完全对齐，使用原始 data 顺序。")
else:
    mapping = []
    for n in bif_nodes:
        mapping.append(data_cols_upper.index(n) if n in data_cols_upper else None)
    if all(m is not None for m in mapping):
        X_raw = data.values[:, mapping]
        print("按 BIF 节点名成功选取并对齐 data 列。")
    else:
        X_raw = data.values
        print("警告: data 列数与 BIF 节点数不一致或无法完全匹配，使用原始 data。")

# ##################################################################
#  核心改动 1: 对数据进行标准化
#  这是提升很多因果发现算法性能的关键步骤
# ##################################################################
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print("\n数据已进行标准化处理 (零均值，单位方差)。")


true_graph = nx.DiGraph()
true_graph.add_nodes_from(bif_nodes)
true_graph.add_edges_from(bif_edges)
print("真实图节点数量:", true_graph.number_of_nodes())
print("真实图边数量:", true_graph.number_of_edges())


# -------------------------------
# 辅助函数 (保持不变, 写得很稳健)
# -------------------------------
def node_to_name(n, node_names):
    if n is None:
        return None
    if isinstance(n, (int, np.integer)):
        if 0 <= int(n) < len(node_names):
            return node_names[int(n)]
        return str(n)
    if isinstance(n, str):
        return n.upper()
    for attr in ('name', 'get_name', 'node_name', 'label'):
        if hasattr(n, attr):
            val = getattr(n, attr)
            if callable(val):
                try: val = val()
                except: pass
            if val is not None: return str(val).upper()
    for attr in ('index', 'idx', 'get_index', 'local_index', 'id'):
        if hasattr(n, attr):
            val = getattr(n, attr)
            if callable(val):
                try: val = val()
                except: pass
            try:
                ii = int(val)
                if 0 <= ii < len(node_names): return node_names[ii]
                else: return str(val).upper()
            except: return str(val).upper()
    return str(n).upper()

def get_graph_edge_iter(cg_result):
    if cg_result is None: return []
    if isinstance(cg_result, (list, tuple, set)):
        for el in cg_result:
            if isinstance(el, (list, tuple)) and len(el) > 0 and isinstance(next(iter(el)), (tuple, list)): return list(el)
            if isinstance(el, np.ndarray): return get_graph_edge_iter(el)
            if hasattr(el, "get_graph_edges") or hasattr(el, "edges"):
                cg_result = el; break
        else:
            try:
                if len(cg_result) > 0 and isinstance(next(iter(cg_result)), (tuple, list)): return list(cg_result)
            except Exception: pass
    if isinstance(cg_result, dict):
        for key in ('G', 'graph', 'causal_graph', 'causalGraph'):
            if key in cg_result: return get_graph_edge_iter(cg_result[key])
        for v in cg_result.values():
            if hasattr(v, "get_graph_edges") or hasattr(v, "edges") or isinstance(v, np.ndarray):
                return get_graph_edge_iter(v)
        return []
    if isinstance(cg_result, np.ndarray):
        return [(int(i), int(j)) for i, j in np.argwhere(np.abs(cg_result) > 0)]
    if hasattr(cg_result, "get_graph_edges"):
        try: return list(cg_result.get_graph_edges())
        except Exception: pass
    if hasattr(cg_result, "G"): return get_graph_edge_iter(cg_result.G)
    if hasattr(cg_result, "edges"):
        try: return list(cg_result.edges())
        except Exception: pass
    for attr in ('adj', 'adjacency', 'graph', 'edge_list', 'edge_list_'):
        if hasattr(cg_result, attr):
            try: return get_graph_edge_iter(getattr(cg_result, attr))
            except Exception: pass
    return []

def edges_to_nx_from_result(cg_result, node_names):
    G = nx.DiGraph(); G.add_nodes_from(node_names)
    edges = get_graph_edge_iter(cg_result)
    for e in edges:
        if isinstance(e, (tuple, list)) and len(e) >= 2: raw_u, raw_v = e[0], e[1]
        else:
            raw_u = getattr(e, "node1", None) or getattr(e, "tail", None) or getattr(e, "u", None) or getattr(e, "source", None)
            raw_v = getattr(e, "node2", None) or getattr(e, "head", None) or getattr(e, "v", None) or getattr(e, "target", None)
            if raw_u is None or raw_v is None:
                for a,b in (('from_node','to_node'), ('left','right'), ('x','y')):
                    raw_u = getattr(e, a, raw_u); raw_v = getattr(e, b, raw_v)
        u, v = node_to_name(raw_u, node_names), node_to_name(raw_v, node_names)
        if u in node_names and v in node_names: G.add_edge(u, v)
    return G

# -------------------------------
# 算法 wrappers (参数优化)
# -------------------------------
wrappers = {}

if available.get('PC'):
    def wrapper_pc(X, node_names):
        # 核心改动 2: 将 alpha 从 0.2 降至 0.05，使检验更严格，减少假阳性边
        res = pc(data=X, alpha=0.05, indep_test=fisherz, show_progress=False)
        return edges_to_nx_from_result(res.G, node_names)
    wrappers['PC'] = wrapper_pc

if available.get('FCI'):
    def wrapper_fci(X, node_names):
        # 核心改动 2: 同样使用更严格的 alpha
        res = fci(X, independence_test_method=fisherz, alpha=0.05)
        return edges_to_nx_from_result(res[0], node_names) # FCI 返回一个元组
    wrappers['FCI'] = wrapper_fci

if available.get('RFCI'):
    def wrapper_rfci(X, node_names):
        # 核心改动 2: 同样使用更严格的 alpha
        res = rfci(data=X, alpha=0.05)
        return edges_to_nx_from_result(res, node_names)
    wrappers['RFCI'] = wrapper_rfci

if available.get('GES'):
    def wrapper_ges(X, node_names):
        # 核心改动 3: 为高斯数据选择正确的分数函数
        # 'local_score_gauss' 是官方推荐的用于连续/高斯数据的分数函数
        record = ges(X, score_func='local_score_gauss', maxP=None, parameters=None)
        return edges_to_nx_from_result(record['G'], node_names)
    wrappers['GES'] = wrapper_ges

if available.get('DEGES'):
    def wrapper_deges(X, node_names):
        res = degs(X)
        return edges_to_nx_from_result(res, node_names)
    wrappers['DEGES'] = wrapper_deges

if available.get('DirectLiNGAM'):
    def wrapper_direct_lingam(X, node_names):
        m = DirectLiNGAM()
        m.fit(X)
        return edges_to_nx_from_result(m.adjacency_matrix_.T, node_names) # LiNGAM的邻接矩阵需要转置
    wrappers['DirectLiNGAM'] = wrapper_direct_lingam

if available.get('ICALiNGAM'):
    def wrapper_ica_lingam(X, node_names):
        m = ICALiNGAM()
        m.fit(X)
        return edges_to_nx_from_result(m.adjacency_matrix_.T, node_names) # LiNGAM的邻接矩阵需要转置
    wrappers['ICALiNGAM'] = wrapper_ica_lingam

if available.get('NOTEARS'):
    # 为 NOTEARS 单独定义一个 wrapper，因为它能返回带权重的矩阵，方便计算更有意义的 AUPR
    def wrapper_notears_func(X, node_names):
        # notears_linear 返回的是权重矩阵 W，我们可以直接用它来构建图
        W = notears_linear(X, lambda1=0.1, loss_type='l2')
        return W
    wrappers['NOTEARS'] = wrapper_notears_func

# -------------------------------
# 指标函数
# -------------------------------
def calculate_shd(true_graph, pred_graph):
    true_adj = nx.to_numpy_array(true_graph, nodelist=pred_graph.nodes())
    pred_adj = nx.to_numpy_array(pred_graph, nodelist=pred_graph.nodes())
    
    # SHD for DAGs: a_ij != b_ij
    diff = true_adj - pred_adj
    
    # Extra edges (FP)
    fp = np.sum(diff < 0)
    
    # Missing edges (FN)
    fn = np.sum(diff > 0)
    
    # Reversed edges
    rev = np.sum((pred_adj.T == 1) & (diff == 1))
    
    return fp + fn - rev

def calculate_aupr(true_adj, pred_scores):
    try:
        return average_precision_score(true_adj.flatten(), pred_scores.flatten())
    except Exception as e:
        print(f"AUPR 计算失败: {e}")
        return float('nan')

# -------------------------------
# 运行并记录
# -------------------------------
results = []
node_names = bif_nodes
true_adj_matrix = nx.to_numpy_array(true_graph, nodelist=node_names)

print("\n将运行以下算法：", list(wrappers.keys()))
for name, fn in wrappers.items():
    start = time.time()
    try:
        print(f"\n运行 {name} ...")
        
        # 特殊处理 NOTEARS
        if name == 'NOTEARS':
            W_pred = fn(X, node_names)
            # 使用一个小的阈值来确定边的存在，用于计算SHD
            pred_graph = edges_to_nx_from_result((np.abs(W_pred) > 0.1).astype(int), node_names)
            # 使用权重的绝对值作为分数来计算AUPR，这比用0/1矩阵更有意义
            aupr = calculate_aupr(true_adj_matrix, np.abs(W_pred))
        else:
            pred_graph = fn(X, node_names)
            pred_adj_matrix = nx.to_numpy_array(pred_graph, nodelist=node_names)
            # 对于其他算法，AUPR基于0/1矩阵计算，这是一个局限
            aupr = calculate_aupr(true_adj_matrix, pred_adj_matrix)
            
        elapsed = time.time() - start
        
        shd = calculate_shd(true_graph, pred_graph)

        results.append({
            "算法": name,
            "SHD": int(shd) if shd is not None else None,
            "AUPR": float(aupr) if not np.isnan(aupr) else None,
            "edges_pred": pred_graph.number_of_edges(),
            "time_s": round(elapsed, 3),
        })
        print(f"✅ {name} 完成: time={elapsed:.3f}s, SHD={shd}, AUPR={aupr:.4f}, edges={pred_graph.number_of_edges()}")

    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        print(f"❌ {name} 失败 after {elapsed:.3f}s: {e}")
        print(tb)
        results.append({
            "算法": name, "SHD": None, "AUPR": None, "edges_pred": None,
            "time_s": round(elapsed, 3), "error": str(e)
        })

# 保存结果
if results:
    df = pd.DataFrame(results)
    df.sort_values(by="SHD", inplace=True)
    df.to_csv("causal_learn_benchmark_results_v2_optimized.csv", index=False)
    print("\n结果已保存到 causal_learn_benchmark_results_v2_optimized.csv")
    print(df)
else:
    print("没有生成结果。")