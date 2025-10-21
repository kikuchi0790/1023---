"""
サンプルデータ定義
簡単な製造プロセス（コーヒー製造）を使用
"""

SAMPLE_PROCESS_NAME = "コーヒー豆の焙煎・包装プロセス"

SAMPLE_PROCESS_DESCRIPTION = """
生のコーヒー豆を焙煎し、品質検査を行い、包装して出荷するプロセス。
主な工程は、焙煎、冷却、品質検査、包装の4つです。
各工程で温度管理、風味評価、異物混入防止などの品質管理が重要です。
"""

EXPECTED_CATEGORIES = [
    "焙煎工程",
    "冷却工程", 
    "品質検査工程",
    "包装工程"
]

EXPECTED_MIN_NODES = 12

EXPECTED_MAX_NODES = 20

EXPECTED_IDEF0_STRUCTURE = {
    "inputs": ["生コーヒー豆", "包装材料"],
    "mechanisms": ["焙煎機", "冷却装置", "検査装置", "包装機"],
    "outputs": ["焙煎済みコーヒー豆", "包装済み製品"]
}
