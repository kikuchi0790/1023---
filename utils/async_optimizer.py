"""
Asynchronous NSGA-II Optimizer Wrapper
非同期NSGA-II最適化ラッパー

Streamlitサーバーのクラッシュを防ぐため、最適化を別プロセスで実行
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROGRESS_DIR = Path("progress")
PROGRESS_DIR.mkdir(exist_ok=True)


class AsyncOptimizer:
    """非同期最適化実行クラス"""
    
    def __init__(self, optimizer_id: str):
        self.optimizer_id = optimizer_id
        self.progress_file = PROGRESS_DIR / f"{optimizer_id}_progress.json"
        self.result_file = PROGRESS_DIR / f"{optimizer_id}_result.json"
        self.process: Optional[mp.Process] = None
    
    def start_step1(
        self,
        dsm_data,
        n_pop: int,
        n_gen: int,
        checkpoint_id: str
    ):
        """STEP-1を非同期実行"""
        if self.is_running():
            logger.warning("Optimizer already running")
            return
        
        self._clear_progress()
        
        self.process = mp.Process(
            target=_run_step1_worker,
            args=(dsm_data, n_pop, n_gen, checkpoint_id, self.progress_file, self.result_file)
        )
        self.process.start()
        logger.info(f"STEP-1 started (PID: {self.process.pid})")
    
    def start_step2(
        self,
        dsm_data,
        removed_indices,
        n_pop: int,
        n_gen: int,
        checkpoint_id: str
    ):
        """STEP-2を非同期実行"""
        if self.is_running():
            logger.warning("Optimizer already running")
            return
        
        self._clear_progress()
        
        self.process = mp.Process(
            target=_run_step2_worker,
            args=(dsm_data, removed_indices, n_pop, n_gen, checkpoint_id, 
                  self.progress_file, self.result_file)
        )
        self.process.start()
        logger.info(f"STEP-2 started (PID: {self.process.pid})")
    
    def is_running(self) -> bool:
        """実行中かチェック"""
        return self.process is not None and self.process.is_alive()
    
    def get_progress(self) -> Optional[Dict[str, Any]]:
        """進捗取得"""
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """結果取得"""
        if not self.result_file.exists():
            return None
        
        try:
            with open(self.result_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def wait(self, timeout: Optional[float] = None):
        """完了まで待機"""
        if self.process:
            self.process.join(timeout=timeout)
    
    def terminate(self):
        """強制終了"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            logger.info("Optimizer terminated")
    
    def _clear_progress(self):
        """進捗ファイル削除"""
        if self.progress_file.exists():
            self.progress_file.unlink()
        if self.result_file.exists():
            self.result_file.unlink()


def _run_step1_worker(
    dsm_data,
    n_pop: int,
    n_gen: int,
    checkpoint_id: str,
    progress_file: Path,
    result_file: Path
):
    """STEP-1ワーカープロセス"""
    from utils.dsm_optimizer import PIMStep1NSGA2
    
    def progress_callback(gen: int, pareto_size: int):
        progress = {
            "step": "step1",
            "generation": gen,
            "total_generations": n_gen,
            "pareto_size": pareto_size,
            "progress_pct": gen / n_gen * 100
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f)
    
    try:
        step1 = PIMStep1NSGA2(dsm_data)
        pareto_front = step1.run(
            n_pop=n_pop,
            n_gen=n_gen,
            checkpoint_id=checkpoint_id,
            save_every=10,
            progress_callback=progress_callback
        )
        
        # 結果を保存（シリアライズ可能な形式に変換）
        step1_results = []
        for ind in pareto_front:
            cost, freedom_inv = ind.fitness.values
            removed_indices = [i for i, val in enumerate(ind) if val == 1]
            removed_nodes = [dsm_data.reordered_nodes[i] for i in removed_indices]
            step1_results.append({
                'individual': list(ind),
                'cost': cost,
                'freedom_inv': freedom_inv,
                'freedom': 1/freedom_inv if freedom_inv != float('inf') else 0,
                'removed_count': len(removed_nodes),
                'removed_nodes': removed_nodes
            })
        
        result = {
            "status": "completed",
            "step": "step1",
            "results": step1_results,
            "pareto_size": len(pareto_front)
        }
        
    except Exception as e:
        logger.error(f"STEP-1 error: {e}", exc_info=True)
        result = {
            "status": "error",
            "step": "step1",
            "error": str(e)
        }
    
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)


def _run_step2_worker(
    dsm_data,
    removed_indices,
    n_pop: int,
    n_gen: int,
    checkpoint_id: str,
    progress_file: Path,
    result_file: Path
):
    """STEP-2ワーカープロセス"""
    from utils.dsm_optimizer import PIMStep2NSGA2
    import numpy as np
    
    def progress_callback(gen: int, pareto_size: int):
        progress = {
            "step": "step2",
            "generation": gen,
            "total_generations": n_gen,
            "pareto_size": pareto_size,
            "progress_pct": gen / n_gen * 100
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f)
    
    try:
        step2 = PIMStep2NSGA2(dsm_data, removed_indices)
        pareto_front = step2.run(
            n_pop=n_pop,
            n_gen=n_gen,
            checkpoint_id=checkpoint_id,
            save_every=10,
            progress_callback=progress_callback
        )
        
        # 結果を保存（シリアライズ可能な形式に変換）
        step2_results = []
        for ind in pareto_front:
            adj, conf, loop = ind.fitness.values
            matrix = ind[0]
            
            # NumPy配列をリストに変換
            matrix_list = matrix.tolist() if isinstance(matrix, np.ndarray) else matrix
            
            step2_results.append({
                'matrix': matrix_list,
                'adjustment': adj,
                'conflict': conf,
                'loop': loop
            })
        
        result = {
            "status": "completed",
            "step": "step2",
            "results": step2_results,
            "pareto_size": len(pareto_front),
            "node_names": step2.pkg["node_name"].tolist()
        }
        
    except Exception as e:
        logger.error(f"STEP-2 error: {e}", exc_info=True)
        result = {
            "status": "error",
            "step": "step2",
            "error": str(e)
        }
    
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
