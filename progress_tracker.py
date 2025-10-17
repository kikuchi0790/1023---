#!/usr/bin/env python3
"""
Progress Tracker for Process Insight Modeler
進捗管理とレポート生成のための自動化ツール
"""

import json
import argparse
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class TaskStatus(Enum):
    """タスクステータス定義"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class PhaseStatus(Enum):
    """フェーズステータス定義"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"


@dataclass
class Task:
    """タスクデータモデル"""
    id: str
    name: str
    status: str
    assigned_date: Optional[str]
    completed_date: Optional[str]
    estimated_hours: float
    actual_hours: float


@dataclass
class Phase:
    """フェーズデータモデル"""
    id: str
    name: str
    status: str
    start_date: str
    end_date: str
    progress_percentage: float
    tasks: List[Task]


class ProgressTracker:
    """進捗管理クラス"""
    
    def __init__(self, progress_file: str = "PROGRESS.json"):
        """
        初期化
        
        Args:
            progress_file: 進捗データファイルパス
        """
        self.progress_file = Path(progress_file)
        self.data = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """進捗データの読み込み"""
        if not self.progress_file.exists():
            raise FileNotFoundError(f"進捗ファイルが見つかりません: {self.progress_file}")
        
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_progress(self) -> None:
        """進捗データの保存"""
        self.data['metrics']['last_updated'] = datetime.now().isoformat()
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          actual_hours: Optional[float] = None) -> bool:
        """
        タスクステータスの更新
        
        Args:
            task_id: タスクID
            status: 新しいステータス
            actual_hours: 実際の作業時間
        
        Returns:
            更新成功の場合True
        """
        for phase in self.data['phases']:
            for task in phase['tasks']:
                if task['id'] == task_id:
                    task['status'] = status.value
                    
                    if status == TaskStatus.IN_PROGRESS and not task['assigned_date']:
                        task['assigned_date'] = date.today().isoformat()
                    elif status == TaskStatus.COMPLETED:
                        task['completed_date'] = date.today().isoformat()
                    
                    if actual_hours is not None:
                        task['actual_hours'] = actual_hours
                    
                    self._update_metrics()
                    self._save_progress()
                    return True
        
        print(f"警告: タスクID {task_id} が見つかりません")
        return False
    
    def _update_metrics(self) -> None:
        """メトリクスの更新"""
        total_tasks = 0
        completed_tasks = 0
        in_progress_tasks = 0
        not_started_tasks = 0
        total_estimated_hours = 0
        total_actual_hours = 0
        
        for phase in self.data['phases']:
            phase_completed = 0
            phase_total = len(phase['tasks'])
            
            for task in phase['tasks']:
                total_tasks += 1
                total_estimated_hours += task['estimated_hours']
                total_actual_hours += task['actual_hours']
                
                if task['status'] == TaskStatus.COMPLETED.value:
                    completed_tasks += 1
                    phase_completed += 1
                elif task['status'] == TaskStatus.IN_PROGRESS.value:
                    in_progress_tasks += 1
                elif task['status'] == TaskStatus.NOT_STARTED.value:
                    not_started_tasks += 1
            
            # フェーズ進捗率の更新
            phase['progress_percentage'] = (phase_completed / phase_total * 100) if phase_total > 0 else 0
            
            # フェーズステータスの更新
            if phase['progress_percentage'] == 0:
                phase['status'] = PhaseStatus.NOT_STARTED.value
            elif phase['progress_percentage'] == 100:
                phase['status'] = PhaseStatus.COMPLETED.value
            else:
                phase['status'] = PhaseStatus.IN_PROGRESS.value
        
        # メトリクスの更新
        metrics = self.data['metrics']
        metrics['total_tasks'] = total_tasks
        metrics['completed_tasks'] = completed_tasks
        metrics['in_progress_tasks'] = in_progress_tasks
        metrics['not_started_tasks'] = not_started_tasks
        metrics['total_estimated_hours'] = total_estimated_hours
        metrics['total_actual_hours'] = total_actual_hours
        metrics['overall_progress_percentage'] = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    def generate_report(self, format: str = "text") -> str:
        """
        進捗レポートの生成
        
        Args:
            format: レポート形式 ("text", "markdown", "json")
        
        Returns:
            レポート文字列
        """
        if format == "json":
            return json.dumps(self.data, ensure_ascii=False, indent=2)
        
        report = []
        
        if format == "markdown":
            report.append("# 進捗レポート")
            report.append(f"\n## プロジェクト: {self.data['project']['name']}")
            report.append(f"**更新日時**: {self.data['metrics']['last_updated']}")
            report.append(f"\n### 📊 全体進捗")
        else:
            report.append("=" * 60)
            report.append(f"進捗レポート - {self.data['project']['name']}")
            report.append(f"更新日時: {self.data['metrics']['last_updated']}")
            report.append("=" * 60)
            report.append("\n全体進捗:")
        
        metrics = self.data['metrics']
        progress_bar = self._create_progress_bar(metrics['overall_progress_percentage'])
        
        report.append(f"進捗率: {metrics['overall_progress_percentage']:.1f}% {progress_bar}")
        report.append(f"完了タスク: {metrics['completed_tasks']}/{metrics['total_tasks']}")
        report.append(f"進行中: {metrics['in_progress_tasks']}, 未着手: {metrics['not_started_tasks']}")
        report.append(f"作業時間: {metrics['total_actual_hours']:.1f}/{metrics['total_estimated_hours']:.1f} 時間")
        
        if format == "markdown":
            report.append("\n### 📋 フェーズ別進捗")
        else:
            report.append("\nフェーズ別進捗:")
        
        for phase in self.data['phases']:
            progress_bar = self._create_progress_bar(phase['progress_percentage'])
            
            if format == "markdown":
                status_emoji = self._get_status_emoji(phase['status'])
                report.append(f"\n#### {status_emoji} {phase['name']}")
                report.append(f"- 期間: {phase['start_date']} 〜 {phase['end_date']}")
                report.append(f"- 進捗: {phase['progress_percentage']:.1f}% {progress_bar}")
            else:
                report.append(f"\n  {phase['name']} ({phase['status']})")
                report.append(f"    期間: {phase['start_date']} 〜 {phase['end_date']}")
                report.append(f"    進捗: {phase['progress_percentage']:.1f}% {progress_bar}")
            
            # タスク詳細
            for task in phase['tasks']:
                status_mark = self._get_status_mark(task['status'])
                
                if format == "markdown":
                    report.append(f"  - {status_mark} {task['name']}")
                else:
                    report.append(f"      {status_mark} {task['name']}")
        
        # マイルストーン
        if format == "markdown":
            report.append("\n### 🎯 マイルストーン")
        else:
            report.append("\nマイルストーン:")
        
        for milestone in self.data['milestones']:
            status_mark = "✅" if milestone['status'] == "completed" else "⏳"
            
            if format == "markdown":
                report.append(f"- {status_mark} **{milestone['name']}** - {milestone['target_date']}")
            else:
                report.append(f"  {status_mark} {milestone['name']} - {milestone['target_date']}")
        
        return "\n".join(report)
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """プログレスバーの作成"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def _get_status_mark(self, status: str) -> str:
        """ステータスマークの取得"""
        marks = {
            TaskStatus.COMPLETED.value: "✅",
            TaskStatus.IN_PROGRESS.value: "🚀",
            TaskStatus.NOT_STARTED.value: "⭕",
            TaskStatus.BLOCKED.value: "🚨"
        }
        return marks.get(status, "❓")
    
    def _get_status_emoji(self, status: str) -> str:
        """フェーズステータス絵文字の取得"""
        emojis = {
            PhaseStatus.COMPLETED.value: "✅",
            PhaseStatus.IN_PROGRESS.value: "🚀",
            PhaseStatus.NOT_STARTED.value: "📋",
            PhaseStatus.DELAYED.value: "⚠️"
        }
        return emojis.get(status, "❓")
    
    def check_quality_metrics(self) -> Dict[str, Any]:
        """
        品質メトリクスのチェック
        
        Returns:
            品質メトリクス辞書
        """
        # ここでは仮の実装
        # 実際には各種テストツールと連携
        quality = {
            "test_coverage": 0,  # pytest-covから取得
            "pylint_score": 0,   # pylintから取得
            "mypy_errors": 0,    # mypyから取得
            "black_formatted": False,  # blackチェック
            "checks_passed": []
        }
        
        # TODO: 実際のツールと統合
        
        return quality
    
    def calculate_velocity(self) -> float:
        """
        開発速度の計算（時間/タスク）
        
        Returns:
            平均開発速度
        """
        completed = [
            task for phase in self.data['phases']
            for task in phase['tasks']
            if task['status'] == TaskStatus.COMPLETED.value and task['actual_hours'] > 0
        ]
        
        if not completed:
            return 0
        
        total_hours = sum(task['actual_hours'] for task in completed)
        return total_hours / len(completed)
    
    def estimate_completion(self) -> Optional[str]:
        """
        完了予定日の推定
        
        Returns:
            推定完了日（ISO形式）
        """
        velocity = self.calculate_velocity()
        if velocity == 0:
            return None
        
        remaining_tasks = [
            task for phase in self.data['phases']
            for task in phase['tasks']
            if task['status'] != TaskStatus.COMPLETED.value
        ]
        
        if not remaining_tasks:
            return date.today().isoformat()
        
        estimated_hours = sum(task['estimated_hours'] for task in remaining_tasks)
        days_needed = estimated_hours / 8  # 1日8時間と仮定
        
        from datetime import timedelta
        completion_date = date.today() + timedelta(days=days_needed)
        
        return completion_date.isoformat()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Process Insight Modeler 進捗管理ツール")
    parser.add_argument("--report", action="store_true", help="進捗レポートを生成")
    parser.add_argument("--format", choices=["text", "markdown", "json"], 
                       default="text", help="レポート形式")
    parser.add_argument("--update-task", metavar="TASK_ID", help="タスクステータスを更新")
    parser.add_argument("--status", choices=["not_started", "in_progress", "completed", "blocked"],
                       help="新しいステータス")
    parser.add_argument("--hours", type=float, help="実際の作業時間")
    parser.add_argument("--quality-check", action="store_true", help="品質メトリクスをチェック")
    parser.add_argument("--estimate", action="store_true", help="完了予定日を推定")
    
    args = parser.parse_args()
    
    try:
        tracker = ProgressTracker()
        
        if args.update_task:
            if not args.status:
                print("エラー: --statusを指定してください")
                return 1
            
            status = TaskStatus[args.status.upper()]
            success = tracker.update_task_status(args.update_task, status, args.hours)
            
            if success:
                print(f"タスク {args.update_task} を {args.status} に更新しました")
            else:
                print("タスクの更新に失敗しました")
                return 1
        
        elif args.quality_check:
            quality = tracker.check_quality_metrics()
            print("品質メトリクス:")
            for key, value in quality.items():
                print(f"  {key}: {value}")
        
        elif args.estimate:
            completion = tracker.estimate_completion()
            if completion:
                print(f"推定完了日: {completion}")
            else:
                print("完了日を推定するには、完了タスクのデータが必要です")
        
        elif args.report:
            report = tracker.generate_report(format=args.format)
            print(report)
        
        else:
            # デフォルト: 簡易レポート表示
            report = tracker.generate_report(format="text")
            print(report)
        
        return 0
        
    except Exception as e:
        print(f"エラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())