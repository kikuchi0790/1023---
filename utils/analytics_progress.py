"""
Advanced Analytics Progress Tracking System
高度な分析用の統一された進捗追跡システム

各分析での進捗表示を統一化し、コードの重複を削減する。
"""

from typing import List, Tuple, Callable, Optional
from contextlib import contextmanager
import time
import streamlit as st


class AnalyticsProgressTracker:
    """
    高度な分析用の統一された進捗追跡システム
    
    機能:
    - 進捗バー表示
    - 推定残り時間
    - ステージ管理（複数ステップの分析）
    - 完了メッセージ
    - エラーハンドリング
    """
    
    def __init__(
        self,
        analysis_name: str,
        total_steps: int = 100,
        show_eta: bool = True
    ):
        """
        Args:
            analysis_name: 分析名
            total_steps: 総ステップ数
            show_eta: 残り時間を表示するかどうか
        """
        self.analysis_name = analysis_name
        self.total_steps = total_steps
        self.show_eta = show_eta
        
        # UI要素
        self.start_msg = st.empty()
        self.progress_bar = st.progress(0.0)
        self.progress_text = st.empty()
        self.stage_text = st.empty()
        
        self.start_time = time.time()
        self.current_step = 0
        self.stages = []
        
        # 開始メッセージ
        self.start_msg.info(f"✨ {analysis_name}を開始しました")
    
    def update(
        self,
        current: int,
        message: str = "",
        stage: str = None
    ):
        """
        進捗を更新
        
        Args:
            current: 現在のステップ（0-total_steps）
            message: 表示メッセージ
            stage: 現在のステージ名（オプション）
        """
        self.current_step = current
        progress_pct = current / self.total_steps if self.total_steps > 0 else 0
        
        # 進捗バー更新
        self.progress_bar.progress(min(progress_pct, 1.0))
        
        # メッセージ構築
        if self.show_eta and current > 0:
            elapsed = time.time() - self.start_time
            eta_seconds = (elapsed / current) * (self.total_steps - current)
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)
            
            msg = f"{message} | 進捗: {progress_pct*100:.1f}% | 残り時間: {eta_min}分{eta_sec}秒"
        else:
            msg = f"{message} | 進捗: {progress_pct*100:.1f}%"
        
        self.progress_text.text(msg)
        
        # ステージ表示
        if stage:
            self.stage_text.markdown(f"**現在のステージ:** {stage}")
    
    def set_stage(self, stage_name: str, step_range: Tuple[int, int]):
        """
        マルチステージ分析のステージを設定
        
        Args:
            stage_name: ステージ名
            step_range: (開始ステップ, 終了ステップ)
        """
        self.stages.append({
            "name": stage_name,
            "start": step_range[0],
            "end": step_range[1]
        })
    
    def complete(self, computation_time: float = None):
        """分析完了"""
        self.progress_bar.progress(1.0)
        self.progress_text.empty()
        self.stage_text.empty()
        
        if computation_time is None:
            computation_time = time.time() - self.start_time
        
        minutes = int(computation_time // 60)
        seconds = int(computation_time % 60)
        
        self.start_msg.success(
            f"✅ {self.analysis_name}が完了しました（計算時間: {minutes}分{seconds}秒）"
        )
    
    def error(self, error_message: str):
        """エラー表示"""
        self.progress_bar.empty()
        self.progress_text.empty()
        self.stage_text.empty()
        
        self.start_msg.error(f"❌ {self.analysis_name}でエラーが発生しました: {error_message}")
    
    def cleanup(self):
        """UI要素をクリーンアップ"""
        self.progress_bar.empty()
        self.progress_text.empty()
        self.stage_text.empty()


@contextmanager
def track_progress(analysis_name: str, total_steps: int = 100, show_eta: bool = True):
    """
    進捗追跡のコンテキストマネージャー
    
    使用例:
        with track_progress("Shapley Value分析", total_steps=1000) as tracker:
            for i in range(1000):
                # 分析処理
                tracker.update(i, f"サンプル{i}処理中...")
    
    Args:
        analysis_name: 分析名
        total_steps: 総ステップ数
        show_eta: 残り時間を表示するかどうか
    
    Yields:
        AnalyticsProgressTracker: 進捗トラッカーインスタンス
    """
    tracker = AnalyticsProgressTracker(analysis_name, total_steps, show_eta)
    
    try:
        yield tracker
        tracker.complete()
    except Exception as e:
        tracker.error(str(e))
        raise


class MultiStageProgress:
    """
    複数ステージを持つ分析の進捗管理
    
    例: Transfer Entropyの場合
    - Stage 1: ランダムウォーク (0-30%)
    - Stage 2: 離散化 (30-35%)
    - Stage 3: TE計算 (35-85%)
    - Stage 4: 統計的フィルタリング (85-95%)
    - Stage 5: 可視化 (95-100%)
    """
    
    def __init__(self, analysis_name: str, stages: List[Tuple[str, float]]):
        """
        Args:
            analysis_name: 分析名
            stages: [(ステージ名, 割合), ...] 割合の合計は1.0
        
        Example:
            progress = MultiStageProgress("Transfer Entropy分析", [
                ("ランダムウォーク", 0.3),
                ("離散化", 0.05),
                ("TE計算", 0.5),
                ("フィルタリング", 0.1),
                ("可視化", 0.05)
            ])
        """
        self.tracker = AnalyticsProgressTracker(analysis_name, total_steps=100)
        self.stages = stages
        self.current_stage_idx = 0
        
        # 累積割合を計算
        self.cumulative_progress = [0]
        for _, pct in stages:
            self.cumulative_progress.append(self.cumulative_progress[-1] + pct)
    
    def start_stage(self, stage_idx: int):
        """
        ステージを開始
        
        Args:
            stage_idx: ステージインデックス（0から始まる）
        """
        self.current_stage_idx = stage_idx
        stage_name = self.stages[stage_idx][0]
        
        self.tracker.update(
            int(self.cumulative_progress[stage_idx] * 100),
            f"{stage_name}を開始...",
            stage=stage_name
        )
    
    def update_stage(self, progress_in_stage: float, message: str = ""):
        """
        現在のステージ内での進捗を更新
        
        Args:
            progress_in_stage: ステージ内の進捗（0.0-1.0）
            message: メッセージ
        """
        stage_start = self.cumulative_progress[self.current_stage_idx]
        stage_pct = self.stages[self.current_stage_idx][1]
        
        overall_progress = stage_start + progress_in_stage * stage_pct
        
        self.tracker.update(
            int(overall_progress * 100),
            message,
            stage=self.stages[self.current_stage_idx][0]
        )
    
    def complete(self):
        """分析完了"""
        self.tracker.complete()
    
    def error(self, error_message: str):
        """エラー表示"""
        self.tracker.error(error_message)


def create_simple_callback(tracker: AnalyticsProgressTracker) -> Callable[[int, int], None]:
    """
    既存の分析関数のprogress_callbackとして使えるシンプルなコールバックを生成
    
    Args:
        tracker: AnalyticsProgressTrackerインスタンス
    
    Returns:
        progress_callback関数
    
    Example:
        tracker = AnalyticsProgressTracker("Shapley Value分析", total_steps=1000)
        callback = create_simple_callback(tracker)
        result = analyzer.compute_shapley_values(progress_callback=callback)
        tracker.complete()
    """
    def callback(current: int, total: int):
        tracker.update(current, f"サンプル{current}/{total}処理中...")
    
    return callback
