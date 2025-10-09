# causal_learn_benchmark_robust.py
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import average_precision_score
import traceback

# -------------------------------
# 尝试动态导入 causal-learn 算法（可用就注册）
# -------------------------------
available = {}
pc = fci = rfci = ges = degs = None
DirectLiNGAM = ICALiNGAM = None
notears_linear = None

try:
    from causallearn.search.ConstraintBased.PC import pc
    available['PC'] = True
except Exception as e:
    available['PC'] = False
    print("PC import failed:", e)

try:
    from causallearn.search.ConstraintBased.FCI import fci
    available['FCI'] = True
except Exception as e:
    available['FCI'] = False
    print("FCI import failed:", e)

try:
    from causallearn.search.ScoreBased.GES import ges
    available['GES'] = True
except Exception as e:
    available['GES'] = False
    print("GES import failed:", e)

# 尝试其它（依据你系统可能不存在）
try:
    from causallearn.search.ConstraintBased.RFCI import rfci
    available['RFCI'] = True
except Exception as e:
    available['RFCI'] = False

try:
    from causallearn.search.ScoreBased.Deges import degs
    available['DEGES'] = True
except Exception as e:
    available['DEGES'] = False

try:
    from causallearn.search.FCMBased.lingam import DirectLiNGAM, ICALiNGAM
    available['DirectLiNGAM'] = True
    available['ICALiNGAM'] = True
except Exception as e:
    # maybe only DirectLiNGAM present
    try:
        from causallearn.search.FCMBased.lingam import DirectLiNGAM
        available['DirectLiNGAM'] = True
        available['ICALiNGAM'] = False
    except Exception:
        available['DirectLiNGAM'] = False
        available['ICALiNGAM'] = False

# NOTE: notears may not be present in your version
try:
    from causallearn.search.ScoreBased.notears import notears_linear
    available['NOTEARS'] = True
except Exception as e:
    available['NOTEARS'] = False

print("可用算法：", {k: v for k, v in available.items() if v})

# -------------------------------
# 读取数据（按你路径调整）
# -------------------------------
from pgmpy.readwrite import BIFReader

file_path = "../data/sachs_data.csv"      # <- 根据你的环境调整
bif_path  = "../data/sachs_true_graph.bif"  # <- 根据你的环境调整

data = pd.read_csv(file_path)
reader = BIFReader(bif_path)
model = reader.get_model()

# 把 BIF 的节点名统一成大写字符串（避免大小写/命名不一致）
bif_nodes = [str(n).upper() for n in model.nodes()]
bif_edges = [(str(u).upper(), str(v).upper()) for (u, v) in model.edges()]

# 如果 data 列名能和 bif_nodes 匹配（忽略大小写），则把 X 的列按 bif_nodes 排序
data_cols_upper = [c.upper() for c in data.columns]
if len(data.columns) == len(bif_nodes):
    try:
        col_order = [data_cols_upper.index(n) for n in bif_nodes]
        X = data.values[:, col_order]
        print("成功将 data 列按 BIF 节点顺序对齐。")
    except ValueError:
        # 无法按名字对齐，退回到原始顺序，并警告
        X = data.values
        print("警告: data 列名无法按 BIF 节点名完全对齐，使用原始 data 顺序。")
else:
    # 列数不等时尽量匹配共有列
    mapping = []
    for n in bif_nodes:
        if n in data_cols_upper:
            mapping.append(data_cols_upper.index(n))
        else:
            mapping.append(None)
    if all(m is not None for m in mapping):
        X = data.values[:, mapping]
        print("按 BIF 节点名成功选取并对齐 data 列。")
    else:
        X = data.values
        print("警告: data 列数与 BIF 节点数不一致，且无法完全按名字对齐。继续使用原始 data (列数=%d, BIF nodes=%d)." % (data.shape[1], len(bif_nodes)))

# 构造 true_graph（networkx.DiGraph）
true_graph = nx.DiGraph()
true_graph.add_nodes_from(bif_nodes)
true_graph.add_edges_from(bif_edges)
print("真实图边数量:", len(bif_edges))

# -------------------------------
# 工具：把各种算法返回的结构解析成 networkx.DiGraph
# -------------------------------
def node_to_name(n, node_names):
    """把各种 node 表示（int / str / GraphNode obj / 有 index 属性）转换为 node name 字符串（与 node_names 一致）"""
    if n is None:
        return None
    # 整数索引
    if isinstance(n, (int, np.integer)):
        if 0 <= int(n) < len(node_names):
            return node_names[int(n)]
        else:
            return str(n)
    # 字符串
    if isinstance(n, str):
        return n.upper()
    # 试探常见属性
    for attr in ('name', 'get_name', 'node_name', 'label'):
        if hasattr(n, attr):
            val = getattr(n, attr)
            if callable(val):
                try:
                    val = val()
                except:
                    pass
            if val is not None:
                return str(val).upper()
    # 试探索引属性
    for attr in ('index', 'idx', 'get_index', 'local_index', 'id'):
        if hasattr(n, attr):
            val = getattr(n, attr)
            if callable(val):
                try:
                    val = val()
                except:
                    pass
            try:
                ii = int(val)
                if 0 <= ii < len(node_names):
                    return node_names[ii]
                else:
                    return str(val)
            except:
                return str(val).upper()
    # 最后退回字符串表示
    return str(n).upper()

def get_graph_edge_iter(cg_result):
    """
    兼容多种返回类型:
      - CausalGraph 或 包含 .G 的 dict/object
      - dict 包含 'G' 键
      - adjacency matrix (np.ndarray)
      - networkx graph
      - list/tuple of (i,j)
      - object with get_graph_edges()
    返回一个 edge iterable（每个 element 可能是 tuple/list 或 Edge-object）
    """
    if cg_result is None:
        return []
    # 如果是 (tuple/list) 直接返回
    if isinstance(cg_result, (list, tuple, set)):
        # 有可能是 (Record, CausalGraph) 或直接 list of edges
        # 选择第一个看起来像 graph 的元素
        # 先尝试找到对象带 get_graph_edges 的元素
        for el in cg_result:
            if hasattr(el, "get_graph_edges") or hasattr(el, "edges") or isinstance(el, np.ndarray):
                cg_result = el
                break
        # 如果仍为序列且第一个元素是二元 tuple，则认为它是边集合
        if len(cg_result) > 0 and isinstance(next(iter(cg_result)), (tuple, list)):
            return list(cg_result)

    # 如果是 dict，尝试取可能的 graph 字段
    if isinstance(cg_result, dict):
        for key in ('G', 'graph', 'causal_graph', 'causalGraph'):
            if key in cg_result:
                return get_graph_edge_iter(cg_result[key])
        # 如果 dict 的值里有 get_graph_edges
        for v in cg_result.values():
            if hasattr(v, "get_graph_edges") or hasattr(v, "edges"):
                return get_graph_edge_iter(v)
        # 不能解析则返回空
        return []

    # 如果是 numpy 矩阵（adj matrix / weight）
    if isinstance(cg_result, np.ndarray):
        edges = []
        for i, j in np.argwhere(np.abs(cg_result) > 0):
            edges.append((int(i), int(j)))
        return edges

    # 如果有 get_graph_edges 方法
    if hasattr(cg_result, "get_graph_edges"):
        try:
            return list(cg_result.get_graph_edges())
        except Exception:
            return []

    # 如果是 object 且有 .G 属性（CausalGraph wrapper）
    if hasattr(cg_result, "G"):
        return get_graph_edge_iter(cg_result.G)

    # 如果是 networkx graph-like
    if hasattr(cg_result, "edges"):
        try:
            return list(cg_result.edges())
        except Exception:
            return []

    # 最后退回空
    return []

def edges_to_nx_from_result(cg_result, node_names):
    """把上面的 edge iterable 解析成 networkx.DiGraph，node_names 是目标节点名列表"""
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    edges = get_graph_edge_iter(cg_result)
    for e in edges:
        # e 可能是 tuple (i,j) 或 Edge-object
        if isinstance(e, (tuple, list)) and len(e) >= 2:
            raw_u, raw_v = e[0], e[1]
        else:
            # 尝试常见属性名
            raw_u = getattr(e, "node1", None) or getattr(e, "tail", None) or getattr(e, "u", None) or getattr(e, "source", None)
            raw_v = getattr(e, "node2", None) or getattr(e, "head", None) or getattr(e, "v", None) or getattr(e, "target", None)
            # 如果还是 None，尝试属性对
            if raw_u is None or raw_v is None:
                for a,b in (('from_node','to_node'), ('left','right'), ('x','y')):
                    raw_u = getattr(e, a, raw_u)
                    raw_v = getattr(e, b, raw_v)
        u = node_to_name(raw_u, node_names)
        v = node_to_name(raw_v, node_names)
        if u in node_names and v in node_names:
            G.add_edge(u, v)
    return G

# -------------------------------
# 各算法包装（仅在导入成功时注册）
# 每个 wrapper 返回 networkx.DiGraph
# -------------------------------
wrappers = {}

if available.get('PC'):
    def wrapper_pc(X, node_names):
        # pc 返回可能是 CausalGraph 或类似结构
        res = pc(data=X, alpha=0.05, show_progress=False)
        return edges_to_nx_from_result(res, node_names)
    wrappers['PC'] = wrapper_pc

if available.get('FCI'):
    from causallearn.utils.cit import fisherz
    def wrapper_fci(X, node_names):
        res = fci(X, fisherz, 0.05)
        return edges_to_nx_from_result(res, node_names)
    wrappers['FCI'] = wrapper_fci

if available.get('RFCI'):
    def wrapper_rfci(X, node_names):
        res = rfci(data=X, alpha=0.05)
        return edges_to_nx_from_result(res, node_names)
    wrappers['RFCI'] = wrapper_rfci

if available.get('GES'):
    def wrapper_ges(X, node_names):
        # 某些版本返回 dict / tuple / CausalGraph，edges_to_nx_from_result 会解析
        try:
            res = ges(X, score_func='local_score_BIC')
        except TypeError:
            # 不同版本的 ges 可能需要不同签名
            res = ges(X)
        return edges_to_nx_from_result(res, node_names)
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
        A = getattr(m, "adjacency_matrix_", None)
        if A is None:
            # ICALiNGAM/其它实现可能给出不同字段
            try:
                A = m.get_adjacency_matrix()
            except Exception:
                A = None
        G = nx.DiGraph(); G.add_nodes_from(node_names)
        if isinstance(A, np.ndarray):
            thresh = 1e-3
            for i, j in np.argwhere(np.abs(A) > thresh):
                u = node_names[j]; v = node_names[i]
                if u in node_names and v in node_names:
                    G.add_edge(u, v)
        return G
    wrappers['DirectLiNGAM'] = wrapper_direct_lingam

if available.get('ICALiNGAM'):
    def wrapper_icalingam(X, node_names):
        m = ICALiNGAM()
        m.fit(X)
        A = getattr(m, "adjacency_matrix_", None)
        G = nx.DiGraph(); G.add_nodes_from(node_names)
        if isinstance(A, np.ndarray):
            for i, j in np.argwhere(np.abs(A) > 1e-3):
                u = node_names[j]; v = node_names[i]
                if u in node_names and v in node_names:
                    G.add_edge(u, v)
        return G
    wrappers['ICALiNGAM'] = wrapper_icalingam

if available.get('NOTEARS'):
    def wrapper_notears(X, node_names):
        W = notears_linear(X, lambda1=0.1)
        G = nx.DiGraph(); G.add_nodes_from(node_names)
        if isinstance(W, np.ndarray):
            for i, j in np.argwhere(np.abs(W) > 1e-6):
                u = node_names[j]; v = node_names[i]
                if u in node_names and v in node_names:
                    G.add_edge(u, v)
        return G
    wrappers['NOTEARS'] = wrapper_notears

# -------------------------------
# 评估函数：SHD / AUPR
# -------------------------------
def calculate_shd(true_graph, pred_graph):
    te = set(true_graph.edges())
    pe = set(pred_graph.edges())
    fp = pe - te
    fn = te - pe
    rev = set((v, u) for (u, v) in pe) & te
    return len(fp) + len(fn) + len(rev)

def calculate_aupr(true_graph, pred_graph, node_names):
    # 确保两个矩阵顺序一致
    true_adj = nx.to_numpy_array(true_graph, nodelist=node_names)
    pred_adj = nx.to_numpy_array(pred_graph, nodelist=node_names)
    try:
        return average_precision_score(true_adj.flatten(), pred_adj.flatten())
    except Exception as e:
        print("AUPR 计算失败:", e)
        return float('nan')

# -------------------------------
# 运行所有可用算法并记录
# -------------------------------
results = []
node_names = bif_nodes  # 使用 BIF 中的节点顺序（已 upper）

print("\n将运行以下算法：", list(wrappers.keys()))
for name, fn in wrappers.items():
    start = time.time()
    try:
        print(f"\n运行 {name} ...")
        pred = fn(X, node_names)
        elapsed = time.time() - start
        if not isinstance(pred, nx.DiGraph):
            print(f"警告: {name} 返回类型不是 networkx.DiGraph，已尝试转换。类型={type(pred)}")
            pred = edges_to_nx_from_result(pred, node_names)
        shd = calculate_shd(true_graph, pred)
        aupr = calculate_aupr(true_graph, pred, node_names)
        results.append({
            "算法": name,
            "SHD": int(shd),
            "AUPR": float(aupr) if not np.isnan(aupr) else None,
            "edges_pred": pred.number_of_edges(),
            "time_s": round(elapsed, 3),
        })
        print(f"✅ {name} 完成: time={elapsed:.3f}s, SHD={shd}, AUPR={aupr:.4f}, edges={pred.number_of_edges()}")
    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        print(f"❌ {name} 失败 after {elapsed:.3f}s: {e}")
        print(tb)
        results.append({
            "算法": name,
            "SHD": None,
            "AUPR": None,
            "edges_pred": None,
            "time_s": round(elapsed, 3),
            "error": str(e)
        })

# 保存结果
if results:
    df = pd.DataFrame(results)
    df.to_csv("causal_learn_benchmark_results.csv", index=False)
    print("\n结果已保存到 causal_learn_benchmark_results.csv")
    print(df)
else:
    print("没有生成结果。")
