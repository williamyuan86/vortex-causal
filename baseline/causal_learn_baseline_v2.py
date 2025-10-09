# causal_learn_benchmark_robust_v2.py
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import average_precision_score
import traceback
from sklearn.preprocessing import StandardScaler
# -------------------------------
# 动态导入 causal-learn（尽量兼容 0.1.x）
# -------------------------------
available = {}
pc = fci = rfci = ges = degs = None
DirectLiNGAM = ICALiNGAM = None
notears_linear = None
fisherz = None

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
        from causallearn.utils.cit import fisherz
    except Exception:
        fisherz = None
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
# 读取数据（按你路径调整）
# -------------------------------
from pgmpy.readwrite import BIFReader

file_path = "../data/sachs_data.csv"      # <- 根据你的环境调整
bif_path  = "../data/sachs_true_graph.bif"  # <- 根据你的环境调整

data = pd.read_csv(file_path)
reader = BIFReader(bif_path)
model = reader.get_model()

# 统一 BIF 节点名为大写字符串
bif_nodes = [str(n).upper() for n in model.nodes()]
bif_edges = [(str(u).upper(), str(v).upper()) for (u, v) in model.edges()]

# 尝试把 CSV 列与 BIF 节点按名字对齐（忽略大小写）
data_cols_upper = [c.upper() for c in data.columns]
if len(data.columns) == len(bif_nodes):
    try:
        col_order = [data_cols_upper.index(n) for n in bif_nodes]
        X = data.values[:, col_order]
        print("成功将 data 列按 BIF 节点顺序对齐。")
    except ValueError:
        X = data.values
        print("警告: data 列名无法按 BIF 节点名完全对齐，使用原始 data 顺序。")
else:
    # 尝试部分匹配
    mapping = []
    for n in bif_nodes:
        mapping.append(data_cols_upper.index(n) if n in data_cols_upper else None)
    if all(m is not None for m in mapping):
        X = data.values[:, mapping]
        print("按 BIF 节点名成功选取并对齐 data 列。")
    else:
        X = data.values
        print("警告: data 列数与 BIF 节点数不一致或无法完全匹配，使用原始 data。")

true_graph = nx.DiGraph()
true_graph.add_nodes_from(bif_nodes)
true_graph.add_edges_from(bif_edges)
print("真实图边数量:", len(bif_edges))

# -------------------------------
# 辅助函数：把各种结构解析成 networkx 图
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
                try:
                    val = val()
                except:
                    pass
            if val is not None:
                return str(val).upper()
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
                    return str(val).upper()
            except:
                return str(val).upper()
    return str(n).upper()

def get_graph_edge_iter(cg_result):
    """
    更稳健地解析算法返回值，返回 edge iterable:
      - 支持 list/tuple of edges
      - 支持 np.ndarray (adj matrix)
      - 支持 dict 包含 'G' / 'graph' / values 带 graph 的情况
      - 支持 object 带 get_graph_edges(), edges(), .G etc.
      - 兼容复合返回值 (tuple 包含 GeneralGraph 等)
    """
    if cg_result is None:
        return []

    # 如果是 list/tuple/set，先尝试直接判断是否是边列表
    if isinstance(cg_result, (list, tuple, set)):
        # 先尝试找到明显的边列表或 ndarray 或 graph-like 元素
        for el in cg_result:
            # 如果 el 是 list/tuple 且第一元素是 tuple/list -> 很可能是边列表
            if isinstance(el, (list, tuple)) and len(el) > 0 and isinstance(next(iter(el)), (tuple, list)):
                return list(el)
            if isinstance(el, np.ndarray):
                return get_graph_edge_iter(el)
            if hasattr(el, "get_graph_edges") or hasattr(el, "edges"):
                # 把 cg_result 指向这个 graph-like 元素，后面统一解析
                cg_result = el
                break
        else:
            # 如果循环没有 break 且 cg_result 本身是边列表
            try:
                if len(cg_result) > 0 and isinstance(next(iter(cg_result)), (tuple, list)):
                    return list(cg_result)
            except Exception:
                pass
    # 如果是 dict，尝试从常见键中取出 graph
    if isinstance(cg_result, dict):
        for key in ('G', 'graph', 'causal_graph', 'causalGraph'):
            if key in cg_result:
                return get_graph_edge_iter(cg_result[key])
        for v in cg_result.values():
            if hasattr(v, "get_graph_edges") or hasattr(v, "edges") or isinstance(v, np.ndarray):
                return get_graph_edge_iter(v)
        return []

    # 如果是 ndarray，视作邻接矩阵
    if isinstance(cg_result, np.ndarray):
        edges = []
        for i, j in np.argwhere(np.abs(cg_result) > 0):
            edges.append((int(i), int(j)))
        return edges

    # 如果有 get_graph_edges 方法（多数 causal-learn Graph 对象会有）
    if hasattr(cg_result, "get_graph_edges"):
        try:
            return list(cg_result.get_graph_edges())
        except Exception:
            # 某些实现可能抛错，继续往下尝试
            pass

    # 如果有 .G 属性（wrapper），递归解析
    if hasattr(cg_result, "G"):
        return get_graph_edge_iter(cg_result.G)

    # 如果有 networkx 风格的 edges()
    if hasattr(cg_result, "edges"):
        try:
            return list(cg_result.edges())
        except Exception:
            pass

    # 尝试访问常见容器属性
    for attr in ('adj', 'adjacency', 'graph', 'edge_list', 'edge_list_'):
        if hasattr(cg_result, attr):
            try:
                return get_graph_edge_iter(getattr(cg_result, attr))
            except Exception:
                pass

    # 最终退回空
    return []

def edges_to_nx_from_result(cg_result, node_names):
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    edges = get_graph_edge_iter(cg_result)
    for e in edges:
        if isinstance(e, (tuple, list)) and len(e) >= 2:
            raw_u, raw_v = e[0], e[1]
        else:
            raw_u = getattr(e, "node1", None) or getattr(e, "tail", None) or getattr(e, "u", None) or getattr(e, "source", None)
            raw_v = getattr(e, "node2", None) or getattr(e, "head", None) or getattr(e, "v", None) or getattr(e, "target", None)
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
# 算法 wrappers（仅注册可用的）
# -------------------------------
wrappers = {}

if available.get('PC'):
    def wrapper_pc(X, node_names):
        res = pc(data=X, alpha=0.2, show_progress=False)
        return edges_to_nx_from_result(res, node_names)
    wrappers['PC'] = wrapper_pc

if available.get('FCI'):
    def wrapper_fci(X, node_names):
        # 针对 0.1.x 的 fci 签名： fci(X, fisherz, alpha)
        if fisherz is None:
            # 若 fisherz 未导入成功，尝试直接调用 fci(X, 0.05)（部分版本可能支持）
            try:
                res = fci(X, 0.05)
            except TypeError:
                # 最后退回显式尝试常见签名
                res = fci(X, None, 0.05)
        else:
            # 标准调用
            try:
                res = fci(X, fisherz, 0.05)
            except TypeError:
                # 若签名不同，尝试关键字参数
                try:
                    res = fci(dataset=X, independence_test_method=fisherz, alpha=0.2)
                except Exception:
                    # 抛出原始异常以便调试
                    raise
        return edges_to_nx_from_result(res, node_names)
    wrappers['FCI'] = wrapper_fci

if available.get('RFCI'):
    def wrapper_rfci(X, node_names):
        res = rfci(data=X, alpha=0.2)
        return edges_to_nx_from_result(res, node_names)
    wrappers['RFCI'] = wrapper_rfci

if available.get('GES'):
    def wrapper_ges(X, node_names):
        try:
            res = ges(X, score_func='local_score_marginal_general')
        except TypeError:
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
        G = nx.DiGraph(); G.add_nodes_from(node_names)
        if isinstance(A, np.ndarray):
            for i, j in np.argwhere(np.abs(A) > 1e-3):
                u = node_names[j]; v = node_names[i]
                if u in node_names and v in node_names:
                    G.add_edge(u, v)
        return G
    wrappers['DirectLiNGAM'] = wrapper_direct_lingam

if available.get('ICALiNGAM'):
    def wrapper_ica_lingam(X, node_names):
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
    wrappers['ICALiNGAM'] = wrapper_ica_lingam

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
# 指标函数
# -------------------------------
def calculate_shd(true_graph, pred_graph):
    te = set(true_graph.edges())
    pe = set(pred_graph.edges())
    fp = pe - te
    fn = te - pe
    rev = set((v, u) for (u, v) in pe) & te
    return len(fp) + len(fn) + len(rev)

def calculate_aupr(true_graph, pred_graph, node_names):
    true_adj = nx.to_numpy_array(true_graph, nodelist=node_names)
    pred_adj = nx.to_numpy_array(pred_graph, nodelist=node_names)
    try:
        return average_precision_score(true_adj.flatten(), pred_adj.flatten())
    except Exception as e:
        print("AUPR 计算失败:", e)
        return float('nan')

# -------------------------------
# 运行并记录
# -------------------------------
results = []
node_names = bif_nodes

print("\n将运行以下算法：", list(wrappers.keys()))
for name, fn in wrappers.items():
    start = time.time()
    try:
        print(f"\n运行 {name} ...")
        pred = fn(X, node_names)
        elapsed = time.time() - start
        if not isinstance(pred, nx.DiGraph):
            print(f"警告: {name} 返回类型不是 networkx.DiGraph，尝试转换。 类型={type(pred)}")
            pred = edges_to_nx_from_result(pred, node_names)
        shd = calculate_shd(true_graph, pred)
        aupr = calculate_aupr(true_graph, pred, node_names)
        results.append({
            "算法": name,
            "SHD": int(shd) if shd is not None else None,
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
    df.to_csv("causal_learn_benchmark_results_v2.csv", index=False)
    print("\n结果已保存到 causal_learn_benchmark_results_v2.csv")
    print(df)
else:
    print("没有生成结果。")
