"""
DSM Optimizer for PIM using NSGA-II
PIM用DSM最適化（NSGA-II使用）

Adapted from nsga_2_clean.py for Process Insight Modeler
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class PIMDSMData:
    """
    PIMアプリケーションのセッションデータからDSM最適化用データを構築
    """
    
    def __init__(
        self,
        adj_matrix_df: pd.DataFrame,
        nodes: List[str],
        idef0_nodes: Dict[str, Dict[str, Any]],
        param_mode: str = "fixed_default",
        llm_params: Optional[Dict[str, Any]] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            adj_matrix_df: 隣接行列（評価スコア）
            nodes: ノード名リスト
            idef0_nodes: IDEF0ノードデータ（カテゴリごと）
            param_mode: "fixed_default" | "llm_auto" | "manual_custom"
            llm_params: LLM評価結果
            custom_params: カスタムパラメータ（cost, structure, range, importance）
        """
        self.adj_matrix_df = adj_matrix_df
        self.nodes = nodes
        self.idef0_nodes = idef0_nodes
        self.param_mode = param_mode
        self.llm_params = llm_params or {}
        self.custom_params = custom_params or {}
        
        self._classify_nodes()
        self._build_dsm_data()
    
    def _classify_nodes(self) -> None:
        """ノードをFR（Output）とDP（Mechanism+Input）に分類"""
        from utils.idef0_classifier import classify_node_type, NodeType
        
        self.fr_indices = []  # Output nodes
        self.dp_indices = []  # Mechanism + Input nodes
        self.node_types = {}
        self.node_categories = {}
        
        for i, node_name in enumerate(self.nodes):
            node_type, category = classify_node_type(node_name, self.idef0_nodes)
            self.node_types[node_name] = node_type
            self.node_categories[node_name] = category
            
            if node_type == NodeType.OUTPUT:
                self.fr_indices.append(i)
            else:  # MECHANISM or INPUT
                self.dp_indices.append(i)
        
        self.fn_num = len(self.fr_indices)
        self.dp_num = len(self.dp_indices)
        
        logger.info(f"Classified nodes: {self.fn_num} FRs (Outputs), {self.dp_num} DPs (Mechanisms+Inputs)")
    
    def _build_dsm_data(self) -> None:
        """DSM最適化用のデータ構造を構築"""
        n = len(self.nodes)
        
        # 隣接行列をNumPy配列として取得
        self.original_matrix = self.adj_matrix_df.values.astype(float)
        
        # FRとDPの順序で並び替え（FR が最初、DP が後）
        reorder_indices = self.fr_indices + self.dp_indices
        self.original_matrix = self.original_matrix[reorder_indices, :][:, reorder_indices]
        self.reordered_nodes = [self.nodes[i] for i in reorder_indices]
        
        # パラメータの取得（param_modeに応じて）
        if self.param_mode == "llm_auto" and "parameters" in self.llm_params:
            # LLM評価結果から取得
            llm_param_data = self.llm_params["parameters"]
            
            # コスト（DP用）
            dp_cost = np.ones((1, n))
            for i, idx in enumerate(reorder_indices):
                node_name = self.nodes[idx]
                if node_name in llm_param_data and "cost" in llm_param_data[node_name]:
                    cost_value = llm_param_data[node_name]["cost"]
                    if cost_value is not None:
                        dp_cost[0, i] = float(cost_value)
            self.original_dp_cost = dp_cost
            
            # 構造グループ
            structures = []
            for idx in reorder_indices:
                node_name = self.nodes[idx]
                if node_name in llm_param_data and "structure" in llm_param_data[node_name]:
                    structures.append(llm_param_data[node_name]["structure"])
                else:
                    structures.append(self.node_categories.get(node_name, "default"))
            self.original_dp_structure = np.array([structures]).reshape(1, -1)
            
            # 変動範囲（DP用）
            dp_range = np.ones((1, n))
            for i, idx in enumerate(reorder_indices):
                node_name = self.nodes[idx]
                if node_name in llm_param_data and "range" in llm_param_data[node_name]:
                    range_value = llm_param_data[node_name]["range"]
                    if range_value is not None:
                        dp_range[0, i] = float(range_value)
            self.original_dp_range = dp_range
            
            # 重要度（FR用）
            fn_importance = np.ones((1, n))
            for i, idx in enumerate(reorder_indices):
                node_name = self.nodes[idx]
                if node_name in llm_param_data and "importance" in llm_param_data[node_name]:
                    importance_value = llm_param_data[node_name]["importance"]
                    if importance_value is not None:
                        fn_importance[0, i] = float(importance_value)
            self.original_fn_importance = fn_importance
            
        elif self.param_mode == "manual_custom":
            # 手動カスタム（将来実装）
            dp_cost = np.array([self.custom_params.get('cost', [1]*n)]).reshape(1, -1)[:, reorder_indices]
            self.original_dp_cost = dp_cost
            
            self.original_dp_structure = np.array([self.custom_params.get('structure', self.node_categories.values())]).reshape(1, -1)
            
            dp_range = np.array([self.custom_params.get('range', [1]*n)]).reshape(1, -1)[:, reorder_indices]
            self.original_dp_range = dp_range
            
            fn_importance = np.array([self.custom_params.get('importance', [1]*n)]).reshape(1, -1)[:, reorder_indices]
            self.original_fn_importance = fn_importance
            
        else:  # fixed_default
            # 固定デフォルト値
            self.original_dp_cost = np.ones((1, n))
            
            structures = []
            for idx in reorder_indices:
                node_name = self.nodes[idx]
                category = self.node_categories.get(node_name, "default")
                if idx in self.fr_indices:
                    structures.append(f"{category}_FR")
                else:
                    structures.append(f"{category}_DP")
            self.original_dp_structure = np.array([structures]).reshape(1, -1)
            
            self.original_dp_range = np.ones((1, n))
            self.original_fn_importance = np.ones((1, n))
        
        # ノード名
        self.original_node_name = np.array([self.reordered_nodes]).reshape(1, -1)
        
        # リクエスト（FR/DPの固定・変更指定、デフォルト=すべて変更可能）
        self.fn_request = np.array([["変更"] * n]).reshape(1, -1)
        self.dp_request = np.array([["変更"] * n]).reshape(1, -1)
    
    @property
    def om_size(self) -> int:
        """マトリクスサイズ"""
        return self.original_matrix.shape[0]
    
    def get_at_request(self) -> np.ndarray:
        """属性変更/固定マスク（簡素化版）
        
        FR（Output）: 常に保持（削除しない）
        DP（Mechanism/Input）: ランダム探索可能
        """
        at_request = np.empty((1, len(self.reordered_nodes)), dtype=object)
        
        for i in range(len(self.reordered_nodes)):
            if i < self.fn_num:
                at_request[0, i] = "保持"  # FRは削除しない
            else:
                at_request[0, i] = "探索"  # DPはランダム探索
        
        return at_request


# -----------------------------------------------------------------------------
# Checkpoint Management
# -----------------------------------------------------------------------------

def save_checkpoint(
    step: str,
    generation: int,
    population: list,
    pareto_front: list,
    params: Dict[str, Any],
    checkpoint_id: str
) -> Path:
    """チェックポイント保存"""
    checkpoint_file = CHECKPOINT_DIR / f"{checkpoint_id}_gen{generation}.pkl"
    
    checkpoint_data = {
        "step": step,
        "generation": generation,
        "population": population,
        "pareto_front": pareto_front,
        "params": params,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    logger.info(f"Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(checkpoint_id: str) -> Optional[Dict[str, Any]]:
    """最新のチェックポイントを復元"""
    checkpoint_files = list(CHECKPOINT_DIR.glob(f"{checkpoint_id}_gen*.pkl"))
    
    if not checkpoint_files:
        return None
    
    latest = max(checkpoint_files, key=lambda p: int(p.stem.split("gen")[1]))
    
    with open(latest, "rb") as f:
        checkpoint_data = pickle.load(f)
    
    logger.info(f"Checkpoint loaded: {latest}")
    return checkpoint_data


def clear_checkpoints(checkpoint_id: str) -> None:
    """チェックポイントファイル削除"""
    for f in CHECKPOINT_DIR.glob(f"{checkpoint_id}_gen*.pkl"):
        f.unlink()
    logger.info(f"Checkpoints cleared: {checkpoint_id}")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def remove_rows_and_columns(matrix: np.ndarray, indices: List[int]) -> np.ndarray:
    """行と列を削除"""
    mod = np.delete(matrix, indices, axis=0)
    mod = np.delete(mod, indices, axis=1)
    return mod


def remove_indices(arr: np.ndarray, indices: List[int]) -> np.ndarray:
    """列を削除"""
    return np.delete(arr, indices, axis=1)


def calculate_cost(dp_structure: np.ndarray, dp_cost: np.ndarray) -> float:
    """コスト計算（同一構造内の最大コストの合計）"""
    unique_structures = np.unique(dp_structure[dp_structure != "None"])
    max_costs = {s: 0.0 for s in unique_structures}
    for i, st in enumerate(dp_structure[0]):
        if st != "None":
            max_costs[st] = max(max_costs[st], float(dp_cost[0, i]))
    return float(sum(max_costs.values()))


def calculate_freedom_per_fn(
    matrix: np.ndarray,
    combo: List[int],
    fn_indices: List[int],
    dp_range: np.ndarray,
    fn_num: int,
) -> np.ndarray:
    """各FNの自由度比を計算"""
    sums_before = []
    for row in fn_indices:
        sums_before.append(
            sum(
                abs(matrix[row, col]) * dp_range[0, col]
                for col in range(matrix.shape[1])
                if matrix[row, col] != 0
            )
        )
    
    mod_matrix = remove_rows_and_columns(matrix, combo)
    mod_range = remove_indices(dp_range, combo)
    
    ratios = np.zeros(fn_num)
    for row in range(fn_num):
        sum_after = sum(
            abs(mod_matrix[row, col]) * mod_range[0, col]
            for col in range(mod_matrix.shape[1])
            if mod_matrix[row, col] != 0
        )
        ratios[row] = sum_after / sums_before[row] if sums_before[row] else 0.0
    return ratios.reshape(1, -1)


def calculate_freedom(package: dict) -> float:
    """総自由度を計算"""
    return float(np.sum(package["fn_freedom"]))


# -----------------------------------------------------------------------------
# STEP-1: Design Parameter Selection
# -----------------------------------------------------------------------------

class PIMStep1NSGA2:
    """STEP-1: 設計パラメータ選択の最適化"""
    
    def __init__(self, data: PIMDSMData):
        self.data = data
        self.at_request = data.get_at_request()
        random.seed(RANDOM_SEED)
        
        # DEAP creator登録（ユニーク名で重複回避）
        creator_name_fitness = f"PIMStep1Fitness_{id(self)}"
        creator_name_individual = f"PIMStep1Individual_{id(self)}"
        
        if not hasattr(creator, creator_name_fitness):
            creator.create(creator_name_fitness, base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, creator_name_individual):
            creator.create(creator_name_individual, list, fitness=getattr(creator, creator_name_fitness))
        
        self.fitness_class = getattr(creator, creator_name_fitness)
        self.individual_class = getattr(creator, creator_name_individual)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", partial(self._mutate, mutpb=0.05))
        self.toolbox.register("select", tools.selNSGA2)
    
    def _create_individual(self):
        """個体生成（バイナリベクトル）
        
        FR（Output）: 0（保持）
        DP（Mechanism/Input）: ランダム（0 or 1）
        """
        ind = []
        for i in range(len(self.data.reordered_nodes)):
            if i < self.data.fn_num:
                ind.append(0)  # FRは削除しない
            else:
                ind.append(random.randint(0, 1))  # DPはランダム
        return self.individual_class(ind)
    
    def _evaluate(self, individual: list[int]) -> Tuple[float, float]:
        """評価関数（コスト、自由度の逆数）"""
        indices = [idx for idx, val in enumerate(individual) if val == 1]
        pkg = self._make_package(indices)
        cost = calculate_cost(pkg["dp_structure"], pkg["dp_cost"])
        freedom = calculate_freedom(pkg)
        freedom_inv = 1.0 / freedom if freedom != 0 else float("inf")
        return cost, freedom_inv
    
    def _crossover(self, ind1, ind2):
        """交叉（一様交叉）"""
        tools.cxUniform(ind1, ind2, indpb=0.5)
        return ind1, ind2
    
    def _mutate(self, individual, mutpb: float):
        """突然変異
        
        FR（Output）: 常に0（保持）
        DP（Mechanism/Input）: mutpbの確率で反転
        """
        for i in range(len(individual)):
            if i < self.data.fn_num:
                individual[i] = 0  # FRは必ず0（保持）
            elif random.random() < mutpb:
                individual[i] = 1 - individual[i]  # DPのみ反転
        return (individual,)
    
    def _make_package(self, combo: List[int]) -> dict:
        """パッケージ生成（削除インデックスから派生データを作成）"""
        d = self.data
        fn_num = d.fn_num
        modified_matrix = remove_rows_and_columns(d.original_matrix, combo)
        pkg = {
            "matrix": modified_matrix.astype(float),
            "matrix_size": modified_matrix.shape[0],
            "dp_num": modified_matrix.shape[0] - fn_num,
            "dp_cost": remove_indices(d.original_dp_cost, combo).astype(float),
            "dp_structure": remove_indices(d.original_dp_structure, combo),
            "node_name": remove_indices(d.original_node_name, combo),
            "range": remove_indices(d.original_dp_range, combo).astype(float),
            "fn_importance": remove_indices(d.original_fn_importance, combo).astype(float),
            "fn_request": remove_indices(d.fn_request, combo),
            "fn_freedom": calculate_freedom_per_fn(
                d.original_matrix,
                combo,
                list(range(fn_num)),
                d.original_dp_range,
                fn_num,
            ),
        }
        return pkg
    
    def run(
        self, 
        n_pop: int = 300, 
        n_gen: int = 100,
        checkpoint_id: Optional[str] = None,
        save_every: int = 10,
        progress_callback: Optional[callable] = None
    ):
        """最適化実行（チェックポイント対応）
        
        Args:
            n_pop: 個体数
            n_gen: 世代数
            checkpoint_id: チェックポイントID（Noneの場合は保存しない）
            save_every: チェックポイント保存間隔（世代数）
            progress_callback: 進捗コールバック関数 callback(gen, pareto_size)
        """
        logger.info(f"Running STEP-1 NSGA-II: pop={n_pop}, gen={n_gen}")
        
        CXPB, MUTPB = 0.9, 0.05
        start_gen = 0
        
        # チェックポイントから復元
        if checkpoint_id:
            checkpoint = load_checkpoint(checkpoint_id)
            if checkpoint and checkpoint.get("step") == "step1":
                pop = checkpoint["population"]
                start_gen = checkpoint["generation"] + 1
                logger.info(f"Resumed from generation {start_gen}")
            else:
                pop = self.toolbox.population(n=n_pop)
        else:
            pop = self.toolbox.population(n=n_pop)
        
        # 初期評価
        if start_gen == 0:
            fitnesses = map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
        
        # 世代ループ
        for gen in range(start_gen, n_gen):
            # 選択
            offspring = self.toolbox.select(pop, n_pop)
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # 突然変異
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 評価
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 次世代
            pop[:] = offspring + pop
            pop[:] = self.toolbox.select(pop, n_pop)
            
            # パレートフロント
            pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            
            # 進捗コールバック
            if progress_callback:
                progress_callback(gen + 1, len(pareto_front))
            
            # チェックポイント保存
            if checkpoint_id and (gen + 1) % save_every == 0:
                save_checkpoint(
                    step="step1",
                    generation=gen,
                    population=pop,
                    pareto_front=pareto_front,
                    params={"n_pop": n_pop, "n_gen": n_gen},
                    checkpoint_id=checkpoint_id
                )
        
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        logger.info(f"STEP-1 completed: {len(pareto_front)} Pareto solutions found")
        
        # 最終チェックポイント削除
        if checkpoint_id:
            clear_checkpoints(checkpoint_id)
        
        return pareto_front


# -----------------------------------------------------------------------------
# STEP-2: Dependency Direction Optimization
# -----------------------------------------------------------------------------

def apply_options_to_all(matrix: np.ndarray) -> np.ndarray:
    """双方向リンクの一方をランダムにゼロ化"""
    size = matrix.shape[0]
    mod = matrix.copy()
    for i in range(size):
        for j in range(i + 1, size):
            if matrix[i, j] != 0 and matrix[j, i] != 0:
                if random.choice([True, False]):
                    mod[i, j] = 0
                else:
                    mod[j, i] = 0
    return mod


# -- Adjustment difficulty (α + γ patterns) --

def _find_alpha_pos(matrix: np.ndarray, fn_num: int):
    """αパターン検出（1つのDPが2つのFRに逆符号で影響）"""
    res = []
    size = matrix.shape[0]
    for i in range(fn_num, size):
        for j in range(fn_num):
            for k in range(j + 1, fn_num):
                if matrix[i, j] * matrix[i, k] < 0:
                    res.append((i, j, k))
    return res


def _trade_off_alpha(matrix, alpha_pos, dp_range, fn_importance):
    """αパターンのトレードオフ量計算"""
    total = 0.0
    for i, j, k in alpha_pos:
        w1j = abs(matrix[i, j]) * dp_range[0, i]
        w2j = sum(abs(matrix[j, p]) * dp_range[0, p] for p in range(matrix.shape[0]) if matrix[j, p] != 0)
        t_j = fn_importance[0, j] * max(1 - w2j / w1j, 0) if w1j else 0
        
        w1k = abs(matrix[i, k]) * dp_range[0, i]
        w2k = sum(abs(matrix[k, p]) * dp_range[0, p] for p in range(matrix.shape[0]) if matrix[k, p] != 0)
        t_k = fn_importance[0, k] * max(1 - w2k / w1k, 0) if w1k else 0
        
        total += t_j + t_k
    return total


def _find_gamma_pos(matrix: np.ndarray, fn_num: int):
    """γパターン検出（FR→DP→FRの連鎖で符号反転）"""
    res = []
    size = matrix.shape[0]
    for i in range(fn_num):
        for j in range(fn_num, size):
            if matrix[i, j] != 0:
                for k in range(fn_num):
                    if matrix[j, k] != 0 and np.sign(matrix[i, j]) != np.sign(matrix[j, k]):
                        res.append((i, j, k))
    return res


def _trade_off_gamma(matrix, gamma_pos, dp_range, fn_importance):
    """γパターンのトレードオフ量計算"""
    total = 0.0
    for i, j, k in gamma_pos:
        w1 = abs(matrix[j, k]) * dp_range[0, j]
        w2 = sum(abs(matrix[k, p]) * dp_range[0, p] for p in range(matrix.shape[0]) if matrix[k, p] != 0)
        t = fn_importance[0, k] * max(1 - w2 / w1, 0) if w1 else 0
        total += t
    return total


def calc_adjustment_difficulty(matrix, dp_range, fn_importance, fn_num):
    """調整困難度計算（α + γ）"""
    alpha_pos = _find_alpha_pos(matrix, fn_num)
    gamma_pos = _find_gamma_pos(matrix, fn_num)
    return _trade_off_alpha(matrix, alpha_pos, dp_range, fn_importance) + _trade_off_gamma(matrix, gamma_pos, dp_range, fn_importance)


# -- Conflict difficulty --

def _find_conflict_cols(matrix):
    """競合列検出（2つ以上の要素から影響を受ける列）"""
    return [i for i in range(matrix.shape[1]) if np.count_nonzero(matrix[:, i]) >= 2]


def _calc_conflict(matrix, conflict_cols, at_range, dp_link_num, dp_link_weight_product):
    """競合困難度計算"""
    total = 0.0
    for i in conflict_cols:
        rows_idx = np.nonzero(matrix[:, i])[0]
        num_nz = len(rows_idx)
        non_zero_elems = matrix[rows_idx, i]
        weight = dp_link_weight_product[0, i] or 1.0
        conf = abs(np.prod(non_zero_elems)) * weight * ((num_nz + dp_link_num[0, i]) ** 3) / at_range[0, i]
        total += conf
    return total


def calc_conflict_difficulty(matrix, at_range, dp_link_num, dp_link_weight_product):
    """競合困難度計算"""
    cols = _find_conflict_cols(matrix)
    return _calc_conflict(matrix, cols, at_range, dp_link_num, dp_link_weight_product)


# -- Loop difficulty --

def _extract_cycles(matrix):
    """閉路検出"""
    import networkx as nx
    G = nx.DiGraph(matrix)
    return list(nx.simple_cycles(G))


def _loop_factors(matrix, cycles, at_range):
    """ループ因子計算"""
    factors = []
    for cyc in cycles:
        rng_prod = np.prod([at_range[0, i] for i in cyc if at_range[0, i] != 0])
        seq_len = len(cyc)
        link_prod = np.prod([abs(matrix[cyc[i], cyc[(i + 1) % len(cyc)]]) for i in range(len(cyc))])
        factors.append((rng_prod, seq_len, link_prod))
    return factors


def _calc_loop_degree(loop_factors):
    """ループ困難度計算"""
    total = 0.0
    for rng_prod, loop_len, link_prod in loop_factors:
        total += (loop_len ** 2 * link_prod) / rng_prod if rng_prod else 0.0
    return total


def calc_loop_difficulty(matrix, at_range):
    """ループ困難度計算"""
    cycles = _extract_cycles(matrix)
    factors = _loop_factors(matrix, cycles, at_range)
    return _calc_loop_degree(factors)


class PIMStep2NSGA2:
    """STEP-2: 依存関係方向決定の最適化"""
    
    def __init__(self, data: PIMDSMData, fixed_indices: List[int]):
        self.data = data
        self.fixed_indices = fixed_indices
        self.pkg = self._make_package_step2(fixed_indices)
        self.fn_num = data.fn_num
        random.seed(RANDOM_SEED)
        
        creator_name_fitness = f"PIMStep2Fitness_{id(self)}"
        creator_name_individual = f"PIMStep2Individual_{id(self)}"
        
        if not hasattr(creator, creator_name_fitness):
            creator.create(creator_name_fitness, base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, creator_name_individual):
            creator.create(creator_name_individual, list, fitness=getattr(creator, creator_name_fitness))
        
        self.fitness_class = getattr(creator, creator_name_fitness)
        self.individual_class = getattr(creator, creator_name_individual)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "individual",
            tools.initIterate,
            self.individual_class,
            lambda: [apply_options_to_all(self.pkg["matrix"])],
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate, init_mat=self.pkg["matrix"], mutpb=0.05)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self._evaluate)
    
    def _make_package_step2(self, combo: List[int]):
        """STEP-2用パッケージ生成"""
        d = self.data
        mod_matrix = remove_rows_and_columns(d.original_matrix, combo)
        mod_range = remove_indices(d.original_dp_range, combo)
        
        def _link_weight_product():
            res = np.zeros((1, mod_matrix.shape[0]))
            columns_removed = set(combo)
            for i in range(d.fn_num, d.om_size):
                if i in columns_removed:
                    continue
                prod = 1
                for col in columns_removed:
                    if d.original_matrix[i, col] != 0:
                        prod *= d.original_matrix[i, col]
                res[0, i - len([c for c in combo if c < i])] = abs(prod)
            return res
        
        def _link_num():
            res = np.zeros((1, mod_matrix.shape[0]))
            columns_removed = set(combo)
            for i in range(d.fn_num, d.om_size):
                if i in columns_removed:
                    continue
                cnt = sum(
                    1
                    for col in columns_removed
                    if col < d.original_matrix.shape[1] and d.original_matrix[i, col] != 0
                )
                res[0, i - len([c for c in combo if c < i])] = cnt
            return res
        
        pkg = {
            "matrix": mod_matrix.astype(float),
            "range": mod_range.astype(float),
            "fn_importance": remove_indices(d.original_fn_importance, combo).astype(float),
            "dp_link_weight_product": _link_weight_product(),
            "dp_link_num": _link_num(),
            "matrix_size": mod_matrix.shape[0],
            "node_name": remove_indices(d.original_node_name, combo),
        }
        return pkg
    
    @staticmethod
    def _mutate(individual, init_mat: np.ndarray, mutpb: float):
        """突然変異（リンク方向反転）"""
        mat = individual[0]
        size = mat.shape[0]
        for i in range(size):
            for j in range(i + 1, size):
                if init_mat[i, j] != 0 and init_mat[j, i] != 0 and random.random() < mutpb:
                    mat[i, j], mat[j, i] = mat[j, i], mat[i, j]
        return (individual,)
    
    @staticmethod
    def _crossover(ind1, ind2, cx_prob: float = 0.5):
        """交叉（ブロック交叉）"""
        if random.random() < cx_prob:
            mat1, mat2 = ind1[0].copy(), ind2[0].copy()
            size = mat1.shape[0]
            cx_point = random.randint(1, size - 1)
            mat1[:cx_point, :cx_point], mat2[:cx_point, :cx_point] = (
                mat2[:cx_point, :cx_point],
                mat1[:cx_point, :cx_point],
            )
            ind1[0], ind2[0] = mat1, mat2
        return ind1, ind2
    
    def _evaluate(self, individual):
        """評価関数（調整困難度、競合困難度、ループ困難度）"""
        mat = individual[0]
        d = self.pkg
        adj = calc_adjustment_difficulty(mat, d["range"], d["fn_importance"], self.fn_num)
        conf = calc_conflict_difficulty(mat, d["range"], d["dp_link_num"], d["dp_link_weight_product"])
        loop = calc_loop_difficulty(mat, d["range"])
        return adj, conf, loop
    
    def run(
        self,
        n_pop: int = 300,
        n_gen: int = 100,
        checkpoint_id: Optional[str] = None,
        save_every: int = 10,
        progress_callback: Optional[callable] = None
    ):
        """最適化実行（チェックポイント対応）
        
        Args:
            n_pop: 個体数
            n_gen: 世代数
            checkpoint_id: チェックポイントID（Noneの場合は保存しない）
            save_every: チェックポイント保存間隔（世代数）
            progress_callback: 進捗コールバック関数 callback(gen, pareto_size)
        """
        logger.info(f"Running STEP-2 NSGA-II: pop={n_pop}, gen={n_gen}")
        
        CXPB, MUTPB = 0.9, 0.05
        MU, LAMBDA = n_pop // 3, 2 * n_pop // 3
        start_gen = 0
        
        # チェックポイントから復元
        if checkpoint_id:
            checkpoint = load_checkpoint(checkpoint_id)
            if checkpoint and checkpoint.get("step") == "step2":
                pop = checkpoint["population"]
                start_gen = checkpoint["generation"] + 1
                logger.info(f"Resumed from generation {start_gen}")
            else:
                pop = self.toolbox.population(n=n_pop)
        else:
            pop = self.toolbox.population(n=n_pop)
        
        # 初期評価
        if start_gen == 0:
            fitnesses = map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
        
        # 世代ループ
        for gen in range(start_gen, n_gen):
            # 選択
            offspring = self.toolbox.select(pop, LAMBDA)
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # 突然変異
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 評価
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 次世代（μ + λ選択）
            pop[:] = self.toolbox.select(pop + offspring, MU)
            
            # パレートフロント
            pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            
            # 進捗コールバック
            if progress_callback:
                progress_callback(gen + 1, len(pareto_front))
            
            # チェックポイント保存
            if checkpoint_id and (gen + 1) % save_every == 0:
                save_checkpoint(
                    step="step2",
                    generation=gen,
                    population=pop,
                    pareto_front=pareto_front,
                    params={"n_pop": n_pop, "n_gen": n_gen},
                    checkpoint_id=checkpoint_id
                )
        
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        logger.info(f"STEP-2 completed: {len(pareto_front)} Pareto solutions found")
        
        # 最終チェックポイント削除
        if checkpoint_id:
            clear_checkpoints(checkpoint_id)
        
        return pareto_front
