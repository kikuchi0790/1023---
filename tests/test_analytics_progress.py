"""
Test cases for Analytics Progress Tracking System
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from utils.analytics_progress import (
    AnalyticsProgressTracker,
    track_progress,
    MultiStageProgress,
    create_simple_callback
)


class TestAnalyticsProgressTracker:
    """AnalyticsProgressTrackerのテスト"""
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_initialization(self, mock_progress, mock_empty):
        """初期化テスト"""
        tracker = AnalyticsProgressTracker("Test Analysis", total_steps=100)
        
        assert tracker.analysis_name == "Test Analysis"
        assert tracker.total_steps == 100
        assert tracker.show_eta == True
        assert tracker.current_step == 0
        
        # UI要素が初期化されている
        assert mock_progress.called
        assert mock_empty.call_count >= 3  # start_msg, progress_text, stage_text
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_update_without_eta(self, mock_progress, mock_empty):
        """ETA表示なしの更新テスト"""
        tracker = AnalyticsProgressTracker("Test", total_steps=100, show_eta=False)
        
        mock_text = MagicMock()
        tracker.progress_text = mock_text
        mock_bar = MagicMock()
        tracker.progress_bar = mock_bar
        
        tracker.update(50, "テストメッセージ")
        
        # 進捗バーが更新されている
        mock_bar.progress.assert_called_once_with(0.5)
        
        # メッセージが更新されている（ETAなし）
        call_args = mock_text.text.call_args[0][0]
        assert "テストメッセージ" in call_args
        assert "進捗: 50.0%" in call_args
        assert "残り時間" not in call_args
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_update_with_eta(self, mock_progress, mock_empty):
        """ETA表示ありの更新テスト"""
        tracker = AnalyticsProgressTracker("Test", total_steps=100, show_eta=True)
        
        mock_text = MagicMock()
        tracker.progress_text = mock_text
        mock_bar = MagicMock()
        tracker.progress_bar = mock_bar
        
        # 少し時間を経過させるためにsleepを挟む
        time.sleep(0.1)
        tracker.update(10, "テストメッセージ")
        
        # メッセージにETAが含まれている
        call_args = mock_text.text.call_args[0][0]
        assert "残り時間" in call_args
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_update_with_stage(self, mock_progress, mock_empty):
        """ステージ表示付き更新テスト"""
        tracker = AnalyticsProgressTracker("Test", total_steps=100)
        
        mock_stage = MagicMock()
        tracker.stage_text = mock_stage
        mock_bar = MagicMock()
        tracker.progress_bar = mock_bar
        
        tracker.update(50, "メッセージ", stage="ステージ1")
        
        # ステージが表示されている
        call_args = mock_stage.markdown.call_args[0][0]
        assert "ステージ1" in call_args
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_complete(self, mock_progress, mock_empty):
        """完了処理のテスト"""
        tracker = AnalyticsProgressTracker("Test Analysis", total_steps=100)
        
        mock_msg = MagicMock()
        tracker.start_msg = mock_msg
        mock_bar = MagicMock()
        tracker.progress_bar = mock_bar
        
        tracker.complete(computation_time=65.5)
        
        # 進捗バーが100%
        mock_bar.progress.assert_called_with(1.0)
        
        # 成功メッセージが表示されている
        call_args = mock_msg.success.call_args[0][0]
        assert "完了しました" in call_args
        assert "1分5秒" in call_args  # 65.5秒 = 1分5秒
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_error(self, mock_progress, mock_empty):
        """エラー処理のテスト"""
        tracker = AnalyticsProgressTracker("Test Analysis", total_steps=100)
        
        mock_msg = MagicMock()
        tracker.start_msg = mock_msg
        
        tracker.error("テストエラー")
        
        # エラーメッセージが表示されている
        call_args = mock_msg.error.call_args[0][0]
        assert "エラーが発生しました" in call_args
        assert "テストエラー" in call_args


class TestTrackProgressContextManager:
    """track_progressコンテキストマネージャーのテスト"""
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_normal_execution(self, mock_progress, mock_empty):
        """正常実行のテスト"""
        with track_progress("Test", total_steps=10) as tracker:
            assert tracker.analysis_name == "Test"
            tracker.update(5, "中間処理")
        
        # 完了メッセージが呼ばれている（completeが自動で呼ばれる）
        # （実際にはmockなので詳細は省略）
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_exception_handling(self, mock_progress, mock_empty):
        """例外発生時のテスト"""
        with pytest.raises(ValueError):
            with track_progress("Test", total_steps=10) as tracker:
                raise ValueError("テスト例外")
        
        # エラーが適切に伝播する


class TestMultiStageProgress:
    """MultiStageProgressのテスト"""
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_initialization(self, mock_progress, mock_empty):
        """初期化テスト"""
        stages = [
            ("ステージ1", 0.3),
            ("ステージ2", 0.5),
            ("ステージ3", 0.2)
        ]
        
        progress = MultiStageProgress("Test", stages)
        
        assert len(progress.stages) == 3
        assert progress.cumulative_progress == [0, 0.3, 0.8, 1.0]
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_start_stage(self, mock_progress, mock_empty):
        """ステージ開始のテスト"""
        stages = [
            ("ステージ1", 0.5),
            ("ステージ2", 0.5)
        ]
        
        progress = MultiStageProgress("Test", stages)
        
        mock_tracker = MagicMock()
        progress.tracker = mock_tracker
        
        progress.start_stage(0)
        
        # update が呼ばれている（0%から）
        assert mock_tracker.update.called
        call_args = mock_tracker.update.call_args
        assert call_args[0][0] == 0  # 0%
        assert "ステージ1" in str(call_args)
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_update_stage(self, mock_progress, mock_empty):
        """ステージ内更新のテスト"""
        stages = [
            ("ステージ1", 0.5),
            ("ステージ2", 0.5)
        ]
        
        progress = MultiStageProgress("Test", stages)
        progress.current_stage_idx = 0
        
        mock_tracker = MagicMock()
        progress.tracker = mock_tracker
        
        # ステージ1の50%進捗 → 全体の25%
        progress.update_stage(0.5, "テストメッセージ")
        
        call_args = mock_tracker.update.call_args
        assert call_args[0][0] == 25  # 0% + 50% * 0.5 = 25%
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_complete(self, mock_progress, mock_empty):
        """完了処理のテスト"""
        stages = [("ステージ1", 1.0)]
        
        progress = MultiStageProgress("Test", stages)
        
        mock_tracker = MagicMock()
        progress.tracker = mock_tracker
        
        progress.complete()
        
        # tracker.completeが呼ばれている
        assert mock_tracker.complete.called


class TestCreateSimpleCallback:
    """create_simple_callbackのテスト"""
    
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    def test_callback_creation(self, mock_progress, mock_empty):
        """コールバック生成のテスト"""
        tracker = AnalyticsProgressTracker("Test", total_steps=100)
        
        mock_tracker_update = MagicMock()
        tracker.update = mock_tracker_update
        
        callback = create_simple_callback(tracker)
        
        # コールバックを呼び出す
        callback(50, 100)
        
        # tracker.updateが適切に呼ばれている
        assert mock_tracker_update.called
        call_args = mock_tracker_update.call_args
        assert call_args[0][0] == 50  # current
        assert "50/100" in call_args[0][1]  # メッセージ


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
