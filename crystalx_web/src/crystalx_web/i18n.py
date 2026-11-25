"""Interface text in Chinese, Japanese and English.

Only the interface's own words live here. Diagnostic messages from the core and from the programs
are shown verbatim (in English) as evidence, under a label that says so: translating a program's
error text loses exactly the detail a crystallographer needs to search for.
"""

from __future__ import annotations

__all__ = [
    "LANGS",
    "DEFAULT_LANG",
    "LANG_CHOICES",
    "t",
    "phase_label",
    "stage_label",
    "stage_state",
    "model_label",
    "refinement_label",
    "readiness_label",
    "alert_meaning",
    "issue_message",
    "atom_columns",
    "runs_columns",
    "native_label",
    "blocker_label",
]

LANGS = ("zh", "ja", "en")
DEFAULT_LANG = "zh"
LANG_CHOICES = [("中文", "zh"), ("日本語", "ja"), ("English", "en")]

# key: (zh, ja, en)
_T: dict[str, tuple[str, str, str]] = {
    # ---------------------------------------------------------------- page
    "title": ("CrystalX 晶体结构初解", "CrystalX 結晶構造初期解析", "CrystalX preliminary structure solution"),
    "header": (
        "# CrystalX 晶体结构初解 · 本地 Web 应用\n上传 INS/HKL → 诊断与规划 → SHELXT 初解 → 有限的 SHELXL 初步精修 → PLATON 诊断 → 三维模型与结果导出。\n\n本应用只在本机运行（仅监听 127.0.0.1），不向外部发送任何数据。得到的是**初步、未经验证**的模型：可以查看和导出，但不能据此排序、接受或宣称已验证。",
        "# CrystalX 結晶構造初期解析 · ローカル Web アプリ\nINS/HKL のアップロード → 診断と計画 → SHELXT 初期解 → 制限付き SHELXL 予備精密化 → PLATON 診断 → 3D モデルと結果のエクスポート。\n\nこのアプリはこのコンピュータ上だけで動作し（127.0.0.1 のみで待ち受け）、外部へデータを送信しません。得られるのは**予備的で未検証**のモデルです。閲覧とエクスポートはできますが、順位付け・採用・検証済みとすることはできません。",
        "# CrystalX preliminary structure solution · local Web app\nUpload INS/HKL → diagnosis and plan → SHELXT initial solution → bounded SHELXL preliminary refinement → PLATON diagnostics → 3-D model and export.\n\nThis app runs only on this computer (it listens on 127.0.0.1 only) and sends no data anywhere. The result is a **preliminary, unvalidated** model: it can be inspected and exported, never ranked, accepted or called validated.",
    ),
    "language": ("界面语言", "表示言語", "Interface language"),
    "mock_banner": (
        "🧪 <b>模拟演示模式</b>：只接受内置的合成演示数据；SHELXT、SHELXL、PLATON 均由替身程序代替，所有数字都是占位值，<b>不是真实结果</b>。",
        "🧪 <b>模擬デモモード</b>：内蔵の合成デモデータのみ受け付けます。SHELXT・SHELXL・PLATON は代替プログラムで、数値はすべてプレースホルダーであり<b>実際の結果ではありません</b>。",
        "🧪 <b>Mock demo mode</b>: only the built-in synthetic demo case is accepted; SHELXT, SHELXL and PLATON are stand-ins and every number is a placeholder — <b>not a real result</b>.",
    ),
    "caps_accordion": ("能力状态：代码就绪程度与本机程序（静态检查）", "機能状態：コードの完成度とローカルプログラム（静的検査）", "Capabilities: code readiness and local programs (static check)"),
    "recheck": ("重新检查", "再確認", "Check again"),
    "sec_input": ("### 1 · 输入与化学先验", "### 1 · 入力と化学的事前情報", "### 1 · Input and chemical prior"),
    "sec_plan": ("### 2 · 诊断与规划", "### 2 · 診断と計画", "### 2 · Diagnosis and plan"),
    "sec_run": ("### 3 · 运行", "### 3 · 実行", "### 3 · Run"),
    "sec_results": ("### 4 · 结果", "### 4 · 結果", "### 4 · Results"),
    "ins_file": ("INS / RES 指令文件", "INS / RES 命令ファイル", "INS / RES instruction file"),
    "hkl_file": ("HKL 衍射数据", "HKL 回折データ", "HKL reflection data"),
    "demo_button": ("载入内置合成演示数据（虚构结构，配合模拟工具链）", "内蔵の合成デモデータを読み込む（架空の構造、模擬ツールチェーン用）", "Load the built-in synthetic demo (fictional structure, mock toolchain)"),
    "mode": ("运行模式", "実行モード", "Run mode"),
    "mode_real": ("本机真实程序", "ローカルの実プログラム", "Real local programs"),
    "mode_mock": ("模拟演示（仅限内置合成数据）", "模擬デモ（内蔵合成データのみ）", "Mock demo (built-in synthetic data only)"),
    "metals": ("金属（逗号分隔）", "金属（カンマ区切り）", "Metals (comma-separated)"),
    "metals_ph": ("例如 Zn", "例：Zn", "e.g. Zn"),
    "elements": ("其他已知元素（逗号分隔）", "その他の既知元素（カンマ区切り）", "Other known elements (comma-separated)"),
    "elements_ph": ("例如 I, S, F", "例：I, S, F", "e.g. I, S, F"),
    "peptide": ("肽类体系（默认视为对映纯，启用手性检查）", "ペプチド系（既定で鏡像異性的に純粋とみなし、キラリティ検査を有効化）", "Peptide-like (treated as enantiopure; enables the chirality check)"),
    "racemic": ("外消旋 / 内消旋制备", "ラセミ体 / メソ体", "Racemic / meso preparation"),
    "composition": ("组成覆盖（可选）：替换 SHELXT 输入副本中的 SFAC/UNIT，逐条记录", "組成の上書き（任意）：SHELXT 入力コピーの SFAC/UNIT を置き換え、すべて記録", "Composition override (optional): replaces SFAC/UNIT in the SHELXT input copy; every change is recorded"),
    "composition_ph": ("例如 C 40 H 52 N 4 O 8 Zn 2", "例：C 40 H 52 N 4 O 8 Zn 2", "e.g. C 40 H 52 N 4 O 8 Zn 2"),
    "limits_accordion": ("运行限额", "実行上限", "Run limits"),
    "shelxt_timeout": ("SHELXT 时限（秒）", "SHELXT 制限時間（秒）", "SHELXT time limit (s)"),
    "shelxl_cycles": ("SHELXL 各向同性循环数", "SHELXL 等方性サイクル数", "SHELXL isotropic cycles"),
    "shelxl_timeout": ("SHELXL 时限（秒）", "SHELXL 制限時間（秒）", "SHELXL time limit (s)"),
    "platon_timeout": ("PLATON 时限（秒）", "PLATON 制限時間（秒）", "PLATON time limit (s)"),
    "cpu_note": ("CPU 时间上限按“时限 × 逻辑 CPU 数”逐个程序设定（SHELXT 为多线程）。", "CPU 時間の上限はプログラムごとに「制限時間 × 論理 CPU 数」です（SHELXT はマルチスレッド）。", "CPU time is capped per program at time limit × logical CPUs (SHELXT is multi-threaded)."),
    "btn_preview": ("诊断与规划", "診断と計画", "Diagnose and plan"),
    "btn_run": ("开始运行", "実行開始", "Start run"),
    "btn_cancel": ("取消运行", "実行を取消", "Cancel run"),
    "plan_hint": ("_上传文件后点击“诊断与规划”，或直接开始运行。_", "_ファイルをアップロードして「診断と計画」を押すか、そのまま実行を開始してください。_", "_Upload the files, then press “Diagnose and plan” or start the run directly._"),
    "no_run": ("_尚无运行。_", "_まだ実行はありません。_", "_No run yet._"),
    "log_box": ("当前程序输出（实时）", "現在のプログラム出力（リアルタイム）", "Current program output (live)"),
    "events_box": ("事件日志", "イベントログ", "Event log"),
    "tab_viewer": ("三维模型", "3D モデル", "3-D model"),
    "tab_shelxt": ("SHELXT 初解", "SHELXT 初期解", "SHELXT solution"),
    "tab_shelxl": ("SHELXL 初步精修", "SHELXL 予備精密化", "SHELXL preliminary refinement"),
    "tab_platon": ("PLATON 诊断", "PLATON 診断", "PLATON diagnostics"),
    "tab_downloads": ("下载", "ダウンロード", "Downloads"),
    "view_choice": ("显示的模型", "表示するモデル", "Model shown"),
    "view_refined": ("初步精修模型", "予備精密化モデル", "Preliminary refined model"),
    "view_solution": ("SHELXT 初解", "SHELXT 初期解", "SHELXT solution"),
    "atom_table": ("位点表（三维视图的文字版，可排序）", "サイト表（3D 表示のテキスト版、並べ替え可）", "Site table (text version of the 3-D view; sortable)"),
    "downloads": ("本次运行的文件（仅限本次运行记录的产物）", "この実行のファイル（この実行が記録した成果物のみ）", "Files of this run (only artefacts this run recorded)"),
    "btn_bundle": ("导出完整运行记录（zip）", "実行記録一式をエクスポート（zip）", "Export the complete run record (zip)"),
    "bundle_file": ("运行记录压缩包", "実行記録アーカイブ", "Run record archive"),
    "history": ("历史运行", "実行履歴", "Run history"),
    "run_id": ("运行 ID", "実行 ID", "Run ID"),
    "btn_load": ("载入该运行", "この実行を読み込む", "Load this run"),
    "btn_refresh": ("刷新列表", "一覧を更新", "Refresh list"),
    "empty_viewer": ("暂无可显示的模型。", "表示できるモデルはまだありません。", "No model to display yet."),
    # ---------------------------------------------------------------- messages
    "need_ins": ("请上传 INS/RES 指令文件。", "INS/RES 命令ファイルをアップロードしてください。", "Please upload an INS/RES instruction file."),
    "need_hkl": ("请上传 HKL 衍射数据文件。", "HKL 回折データファイルをアップロードしてください。", "Please upload an HKL reflection file."),
    "bad_composition": ("组成覆盖：格式应为“元素 数量”成对出现（{detail}）。", "組成の上書き：「元素 数」の組で入力してください（{detail}）。", "Composition override: use element/count pairs ({detail})."),
    "bad_limits": ("运行限额无效：{detail}", "実行上限が無効です：{detail}", "Invalid run limits: {detail}"),
    "no_active_run": ("_没有正在进行的运行。_", "_実行中のものはありません。_", "_No active run._"),
    "cancel_requested": ("已请求取消：正在终止程序的整个进程组，已写出的日志会保留。", "取消を要求しました：プログラムのプロセスグループ全体を終了します。書き出し済みのログは保持されます。", "Cancellation requested: the program's whole process group is being stopped; logs written so far are kept."),
    "cancel_too_late": ("该运行已不在进行中。", "この実行はもう進行していません。", "This run is no longer active."),
    "run_not_found": ("⛔ 找不到该运行 ID。", "⛔ この実行 ID は見つかりません。", "⛔ No run with that ID."),
    "run_ended_no_output": ("（运行已结束，没有程序输出。）", "（実行は終了しました。プログラム出力はありません。）", "(The run has ended; no program output.)"),
    "run_ended_log": ("（运行已结束；各程序的完整输出见“下载”）", "（実行終了。各プログラムの完全な出力は「ダウンロード」にあります）", "(run ended; full program output is under “Downloads”)"),
    "need_finished_run": ("_请先完成一次运行。_", "_先に実行を完了してください。_", "_Finish a run first._"),
    "export_failed": ("_无法导出：{detail}_", "_エクスポートできません：{detail}_", "_Cannot export: {detail}_"),
    "bundle_ready": ("已生成运行记录压缩包（包含输入副本、日志、程序输出与全部记录）。", "実行記録アーカイブを作成しました（入力のコピー、ログ、プログラム出力、全記録を含む）。", "Run record archive created (input copies, logs, program outputs and the full record)."),
    "progress": ("{done}/{total} 步已结束", "{done}/{total} ステップ終了", "{done}/{total} steps finished"),
    "progress_aria": ("已结束的步骤", "終了したステップ", "Finished steps"),
    # ---------------------------------------------------------------- render: shared
    "unknown": ("未知", "不明", "unknown"),
    "none": ("（无）", "（なし）", "(none)"),
    "original": ("（原文）", "（原文）", "(original text)"),
    "warnings": ("提醒", "注意", "Notes"),
    "seconds": ("秒", "秒", "s"),
    # capabilities
    "core_version": ("核心版本", "コアバージョン", "Core version"),
    "tool_source": ("工具来源", "ツールの由来", "Tools"),
    "src_mock": ("模拟演示工具链", "模擬デモツールチェーン", "mock demo toolchain"),
    "src_real": ("本机真实程序", "ローカルの実プログラム", "real local programs"),
    "host": ("主机", "ホスト", "Host"),
    "rosetta_na": ("不适用", "該当なし", "not applicable"),
    "rosetta_marker": ("检测到标记文件（不等于可以启动）", "マーカーファイルあり（起動できるとは限らない）", "marker file present (not proof it can launch)"),
    "rosetta_absent": ("未检测到", "検出されず", "not detected"),
    "tools_static": ("**程序可用性（静态检查，不是启动测试）**", "**プログラムの利用可否（静的検査であり起動テストではない）**", "**Program availability (static check, not a launch test)**"),
    "col_program": ("程序", "プログラム", "Program"),
    "col_static": ("静态检查", "静的検査", "Static check"),
    "col_diagnostic": ("诊断", "診断", "Diagnostic"),
    "col_arch": ("架构", "アーキテクチャ", "Arch"),
    "col_path": ("路径", "パス", "Path"),
    "col_note": ("说明", "説明", "Note"),
    "pass": ("✅ 通过", "✅ 合格", "✅ pass"),
    "unusable": ("⛔ 不可用", "⛔ 利用不可", "⛔ unusable"),
    "readiness_title": ("**代码就绪程度**", "**コードの完成度**", "**Code readiness**"),
    "col_component": ("组件", "コンポーネント", "Component"),
    "col_status": ("状态", "状態", "Status"),
    "inference": (
        "**CrystalX 推理**：本版本未接入工作流（实验性 Apple 芯片后端{pkg}，PyTorch {torch}；仅用合成权重验证过软件路径）。元素标签来自 SHELXT/SHELXL 的峰高，未经独立验证；详见“原子指认”页。",
        "**CrystalX 推論**：このバージョンではワークフローに組み込まれていません（実験的な Apple シリコンバックエンド{pkg}、PyTorch {torch}。合成重みでソフトウェア経路を確認しただけ）。元素ラベルは SHELXT/SHELXL のピーク高さに由来し、独立には検証されていません。「原子の帰属」タブを参照してください。",
        "**CrystalX inference**: not part of the workflow in this release (experimental Apple Silicon backend {pkg}, PyTorch {torch}; software path verified with synthetic weights only). Element labels come from SHELXT/SHELXL peak heights and are not independently verified; see the Atom assignment tab.",
    ),
    "installed": ("已安装", "インストール済み", "installed"),
    "not_installed": ("未安装", "未インストール", "not installed"),
    # diagnosis
    "not_diagnosed": ("_尚未诊断。_", "_まだ診断していません。_", "_Not diagnosed yet._"),
    "d_title": ("标题", "タイトル", "Title"),
    "d_cell": ("晶胞", "格子", "Cell"),
    "d_cell_missing": ("未知（文件无 CELL）", "不明（ファイルに CELL がない）", "unknown (no CELL record)"),
    "d_wavelength": ("波长", "波長", "Wavelength"),
    "d_assumed": ("，假定值", "、仮定値", ", assumed"),
    "d_symmetry": ("对称性", "対称性", "Symmetry"),
    "d_chirality": ("手性相容性", "キラリティとの整合性", "chirality compatibility"),
    "d_composition": ("组成", "組成", "Composition"),
    "d_trusted": ("可信", "信頼できる", "trusted"),
    "d_placeholder": ("疑似占位值", "仮の値の疑い", "looks like a placeholder"),
    "d_override": ("覆盖为", "上書き", "override"),
    "d_reflections": ("衍射点", "反射", "Reflections"),
    "d_count": ("{n} 条", "{n} 本", "{n}"),
    "d_anomalous": ("反常信号", "異常分散", "anomalous"),
    "d_friedel": ("Friedel 对 {n}", "Friedel 対 {n}", "{n} Friedel pairs"),
    "d_resolution": ("分辨率", "分解能", "resolution"),
    "d_mean_ios": ("平均 I/σ", "平均 I/σ", "mean I/σ"),
    "d_completeness": ("完整度", "完全性", "Completeness"),
    "d_not_computed": ("未计算", "未計算", "not computed"),
    "d_handedness": ("手性假设", "キラリティの仮定", "Handedness"),
    "d_enantiopure": ("对映纯", "鏡像異性的に純粋", "enantiopure"),
    "d_not_enantiopure": ("不假定对映纯", "鏡像異性的純度は仮定しない", "enantiopurity not assumed"),
    # preconditions and plan
    "pre_title": ("前置检查", "前提条件の検査", "Preconditions"),
    "pre_ok": ("✅ 可以开始", "✅ 開始可能", "✅ may start"),
    "pre_refused": ("⛔ 拒绝开始", "⛔ 開始拒否", "⛔ refused"),
    "pre_block": ("阻止", "阻止", "block"),
    "pre_note": ("提示", "注意", "note"),
    "pre_evidence": ("证据", "根拠", "evidence"),
    "pre_remedy": ("建议", "対処", "remedy"),
    "pre_override": ("覆盖方式", "上書き方法", "override"),
    "plan_routes": ("**初解路线**", "**初期解の経路**", "**Initial-solution routes**"),
    "col_route": ("路线", "経路", "Route"),
    "col_execute": ("执行", "実行", "Runs"),
    "col_readiness": ("就绪程度", "完成度", "Readiness"),
    "col_priority": ("优先级", "優先度", "Priority"),
    "col_reasons": ("理由", "理由", "Reasons"),
    "will_run": ("▶ 将运行", "▶ 実行する", "▶ will run"),
    "not_run": ("— 不运行", "— 実行しない", "— not run"),
    "plan_excluded": ("**未采用的路线**", "**採用しない経路**", "**Routes not used**"),
    "plan_downstream": ("**后续有限步骤**", "**後続の制限付きステップ**", "**Bounded follow-up steps**"),
    "time_limit": ("时限 {s} 秒", "制限 {s} 秒", "limit {s} s"),
    "plan_limits": (
        "资源限制：SHELXT CPU {cpu} 秒，单文件 ≤ {mib} MiB，打开文件 ≤ {files}，nice +{nice}；内存不设限（macOS 不执行 RLIMIT_AS）",
        "資源制限：SHELXT CPU {cpu} 秒、単一ファイル ≤ {mib} MiB、オープンファイル ≤ {files}、nice +{nice}。メモリは無制限（macOS は RLIMIT_AS を強制しない）",
        "Limits: SHELXT CPU {cpu} s, largest file ≤ {mib} MiB, open files ≤ {files}, nice +{nice}; memory unlimited (macOS does not enforce RLIMIT_AS)",
    ),
    # status
    "s_run": ("运行", "実行", "Run"),
    "s_mock": ("🧪 **模拟演示（非真实结果）**", "🧪 **模擬デモ（実際の結果ではない）**", "🧪 **mock demo (not a real result)**"),
    "s_state": ("状态", "状態", "State"),
    "col_step": ("步骤", "ステップ", "Step"),
    "col_start": ("开始", "開始", "Start"),
    "col_end": ("结束", "終了", "End"),
    "e_phase": ("运行状态", "実行状態", "run"),
    "e_exec": ("执行", "実行", "exec"),
    "e_exit": ("结束", "終了", "exit"),
    "e_staged": ("已复制输入 {role}（sha256 {digest}…）", "入力 {role} をコピー（sha256 {digest}…）", "input {role} copied (sha256 {digest}…)"),
    # results
    "no_results": ("_尚无结果。_", "_まだ結果はありません。_", "_No results yet._"),
    "r_mock": (
        "> 🧪 **模拟演示**：输入是程序生成的虚构数据，SHELXT/SHELXL/PLATON 均为替身程序。以下数字是占位值，**不是真实结果**。",
        "> 🧪 **模擬デモ**：入力はプログラムが生成した架空のデータで、SHELXT/SHELXL/PLATON は代替プログラムです。以下の数値はプレースホルダーであり、**実際の結果ではありません**。",
        "> 🧪 **Mock demo**: the input is generated, fictional data and SHELXT/SHELXL/PLATON are stand-ins. The numbers below are placeholders, **not real results**.",
    ),
    "col_fact": ("事实", "事実", "Fact"),
    "col_conclusion": ("结论", "結論", "Conclusion"),
    "f_process": ("过程完成情况", "処理の完了状況", "Process"),
    "f_model": ("可用模型", "利用できるモデル", "Model available"),
    "f_refinement": ("精修状态", "精密化の状態", "Refinement"),
    "f_eligibility": ("科学资格", "科学的適格性", "Scientific eligibility"),
    "f_not_evaluated": ("**未评估，不可排序或称为已验证**", "**未評価：順位付けや検証済みとすることはできない**", "**not evaluated — cannot be ranked or called validated**"),
    "no_solution": ("_没有 SHELXT 解。_", "_SHELXT の解はありません。_", "_No SHELXT solution._"),
    "col_solution": ("解", "解", "Solution"),
    "col_sg": ("空间群（SHELXT）", "空間群（SHELXT）", "Space group (SHELXT)"),
    "col_atoms": ("原子数", "原子数", "Atoms"),
    "col_elements": ("元素", "元素", "Elements"),
    "col_chirality": ("手性相容性", "キラリティとの整合性", "Chirality"),
    "primary": ("（首选）", "（第一候補）", " (primary)"),
    "sg_unstated": ("未注明", "記載なし", "not stated"),
    "concerns": ("**需要注意**", "**注意点**", "**Concerns**"),
    "lxt_rows": ("_.lxt 表格读取到 {n} 行；格式{v}核对，仅供参考。_", "_.lxt 表から {n} 行を読み取りました。形式は{v}。参考情報です。_", "_{n} rows read from the .lxt table; format {v}; indicative only._"),
    "verified": ("已", "照合済み", "verified"),
    "unverified": ("未", "未照合", "unverified"),
    "no_refinement": ("_未进行精修。_", "_精密化は行われていません。_", "_No refinement._"),
    "tool_unavailable": ("工具不可用", "ツール利用不可", "Tool unavailable"),
    "refinement_not_done": ("精修未进行", "精密化は行われなかった", "Refinement not run"),
    "failed": ("失败", "失敗", "Failed"),
    "ref_value": ("数值", "値", "Value"),
    "ref_from_lst": ("来自 .lst", ".lst より", "from .lst"),
    "ref_from_res": ("来自 .res", ".res より", "from .res"),
    "ref_r1obs": ("R1（Fo > 4σ）", "R1（Fo > 4σ）", "R1 (Fo > 4σ)"),
    "ref_r1all": ("R1（全部）", "R1（全データ）", "R1 (all)"),
    "ref_mean_shift": ("平均 位移/su", "平均 シフト/su", "mean shift/su"),
    "ref_max_shift": ("最大 位移/su", "最大 シフト/su", "max shift/su"),
    "ref_peak": ("最高残余峰 (e/Å³)", "最大残差ピーク (e/Å³)", "highest peak (e/Å³)"),
    "ref_hole": ("最深空穴 (e/Å³)", "最深ホール (e/Å³)", "deepest hole (e/Å³)"),
    "ref_params": ("精修参数", "精密化パラメータ", "parameters"),
    "ref_cycles": ("已运行循环", "実行サイクル", "cycles run"),
    "ref_settings": ("**设置** {cycles} 个各向同性循环，时限 {timeout} 秒", "**設定** 等方性 {cycles} サイクル、制限 {timeout} 秒", "**Settings** {cycles} isotropic cycles, limit {timeout} s"),
    "ref_diag": ("**程序诊断**", "**プログラム診断**", "**Program diagnostic**"),
    "ref_unknown_note": (
        "未能解析的数值显示为“未知”，不会以 0 代替。.lst 格式{v}与本机真实 SHELXL 输出核对。",
        "解析できなかった値は「不明」と表示し、0 で置き換えません。.lst の形式はこのマシンの実際の SHELXL 出力と{v}。",
        "Values that could not be parsed are shown as unknown, never as 0. The .lst format is {v} against real SHELXL output on this machine.",
    ),
    "ref_verified_yes": ("已", "照合済み", "verified"),
    "ref_verified_no": ("尚未", "未照合", "not yet verified"),
    "ref_disagree": ("**.lst 与 .res 不一致**", "**.lst と .res の不一致**", "**.lst and .res disagree**"),
    "ref_changes": ("**对精修输入所做的修改（逐条记录）**", "**精密化入力への変更（すべて記録）**", "**Changes made to the refinement input (each recorded)**"),
    "no_platon": ("_未进行 PLATON 诊断。_", "_PLATON 診断は行われていません。_", "_No PLATON diagnostics._"),
    "skipped": ("跳过", "スキップ", "Skipped"),
    "blocked": ("被阻止", "阻止", "Blocked"),
    "p_program": ("程序", "プログラム", "Program"),
    "col_level": ("级别", "レベル", "Level"),
    "col_code": ("代码", "コード", "Code"),
    "col_type": ("类型", "種類", "Type"),
    "col_text": ("内容", "内容", "Text"),
    "p_no_alerts": ("_报告中没有可识别的 ALERT 行。_", "_レポートに識別できる ALERT 行はありません。_", "_No recognisable ALERT lines in the report._"),
    "p_unrecognised": ("**含 ALERT 但未能识别的行**", "**ALERT を含むが識別できない行**", "**Lines mentioning ALERT that were not recognised**"),
    "p_levels": ("级别含义", "レベルの意味", "Levels"),
    "p_format": (".chk 格式{v}与本机真实 PLATON 输出核对。", ".chk の形式はこのマシンの実際の PLATON 出力と{v}。", "The .chk format is {v} against real PLATON output on this machine."),
    # atoms
    "kind_atom": ("原子", "原子", "atom"),
    "kind_unknown": ("未知位点", "未知サイト", "unknown site"),
    "kind_q": ("Q 峰（差值电子密度）", "Q ピーク（差電子密度）", "Q peak (difference density)"),
}

_PHASE = {
    "queued": ("排队中", "待機中", "queued"),
    "running": ("运行中", "実行中", "running"),
    "completed": ("已完成", "完了", "completed"),
    "failed": ("失败", "失敗", "failed"),
    "cancelled": ("已取消", "取消済み", "cancelled"),
    "blocked": ("已阻止（未启动程序）", "阻止（プログラム未起動）", "blocked (no program started)"),
    "interrupted": ("已中断（服务曾停止）", "中断（サービス停止）", "interrupted (service stopped)"),
    "unknown": ("未知", "不明", "unknown"),
}

_STAGE = {
    "stage_inputs": ("复制输入", "入力のコピー", "Copy inputs"),
    "diagnose": ("数据诊断", "データ診断", "Diagnose data"),
    "preconditions": ("前置检查", "前提条件の検査", "Preconditions"),
    "plan": ("求解规划", "解法の計画", "Plan"),
    "shelxt": ("SHELXT 初解", "SHELXT 初期解", "SHELXT solution"),
    "shelxl": ("SHELXL 初步精修", "SHELXL 予備精密化", "SHELXL refinement"),
    "platon": ("PLATON 诊断", "PLATON 診断", "PLATON diagnostics"),
    "viewer": ("三维模型", "3D モデル", "3-D model"),
}

_STAGE_STATE = {
    "pending": ("○", ("等待", "待機", "pending")),
    "running": ("◐", ("进行中", "実行中", "running")),
    "succeeded": ("●", ("完成", "完了", "done")),
    "failed": ("✕", ("失败", "失敗", "failed")),
    "skipped": ("–", ("跳过", "スキップ", "skipped")),
    "blocked": ("⊘", ("被阻止", "阻止", "blocked")),
    "tool_unavailable": ("⊘", ("工具不可用", "ツール利用不可", "tool unavailable")),
    "cancelled": ("■", ("已取消", "取消済み", "cancelled")),
}

_MODEL = {
    "none": ("无模型", "モデルなし", "no model"),
    "solution_model": ("SHELXT 初解模型（未精修）", "SHELXT 初期解モデル（未精密化）", "SHELXT solution model (unrefined)"),
    "preliminary_refined_model": ("初步精修模型（未验证）", "予備精密化モデル（未検証）", "preliminary refined model (unvalidated)"),
}

_REFINEMENT = {
    "not_run": ("未运行", "未実行", "not run"),
    "failed": ("失败", "失敗", "failed"),
    "errors_reported": (
        "SHELXL 报告了错误标记行，未提供精修模型（待真实输出核对哪些属致命错误）",
        "SHELXL がエラー行を報告したため精密化モデルは提供しない（どれが致命的かは実出力で要確認）",
        "SHELXL reported error-marked lines; no refined model is offered (which are fatal awaits real output)",
    ),
    "ran_convergence_not_assessed": (
        "已运行；收敛未评估（显示原始平均与最大 位移/su）",
        "実行済み。収束は未評価（平均・最大 シフト/su の生の値を表示）",
        "ran; convergence not assessed (raw mean and maximum shift/su shown)",
    ),
}

_READINESS = {
    "production": ("已实现并测试", "実装・テスト済み", "implemented and tested"),
    "implemented": ("已实现（输出格式待真实运行核对）", "実装済み（出力形式は実際の実行で要照合）", "implemented (output format awaits a real run)"),
    "plan_only": ("仅规划（不执行）", "計画のみ（実行しない）", "plan only (never executed)"),
    "not_integrated": ("未集成", "未統合", "not integrated"),
    "not_evaluated": ("未评估", "未評価", "not evaluated"),
    "no_backend": ("无后端", "バックエンドなし", "no backend"),
}

_ISSUE = {
    "missing_file": ("文件不存在", "ファイルが存在しません", "file does not exist"),
    "wrong_extension": ("文件扩展名不对", "拡張子が正しくありません", "wrong file extension"),
    "unknown_element": ("不是有效的元素符号", "有効な元素記号ではありません", "not an element symbol"),
    "non_positive_count": ("数量必须为正数", "数は正でなければなりません", "count must be positive"),
    "duplicate_element": ("元素重复", "元素が重複しています", "element listed twice"),
    "mock_requires_demo": (
        "模拟演示只能使用内置的合成演示数据；真实数据需要真实程序",
        "模擬デモは内蔵の合成デモデータでのみ実行できます。実データには実プログラムが必要です",
        "the mock demo only runs on the built-in synthetic demo case; real data needs real programs",
    ),
    "unreadable_input": ("输入文件无法解析", "入力ファイルを解析できません", "the input file could not be read"),
}

_FIELD = {
    "ins": ("INS/RES 文件", "INS/RES ファイル", "INS/RES file"),
    "hkl": ("HKL 文件", "HKL ファイル", "HKL file"),
    "metals": ("金属", "金属", "metals"),
    "elements": ("其他元素", "その他の元素", "other elements"),
    "composition": ("组成覆盖", "組成の上書き", "composition override"),
    "mode": ("运行模式", "実行モード", "run mode"),
    "input": ("输入文件", "入力ファイル", "input file"),
}

_ALERT = {
    "A": ("A：很可能是严重问题，需要解决或解释", "A：重大な問題の可能性が高い。解決または説明が必要", "A: most likely a serious problem; resolve or explain"),
    "B": ("B：潜在的严重问题，需要仔细考虑", "B：重大な問題の可能性。慎重に検討", "B: a potentially serious problem; consider carefully"),
    "C": ("C：请检查，确认不是遗漏或疏忽", "C：見落としや不注意でないか確認", "C: check that it is not an omission or oversight"),
    "G": ("G：一般信息，确认不是意外情况", "G：一般情報。想定外でないか確認", "G: general information; check it is not unexpected"),
}

_ATOM_COLUMNS = (
    ["标签", "元素", "类型", "x", "y", "z", "占有率", "Uiso/Ueq (Å²)", "峰高"],
    ["ラベル", "元素", "種類", "x", "y", "z", "占有率", "Uiso/Ueq (Å²)", "ピーク高"],
    ["Label", "Element", "Kind", "x", "y", "z", "Occupancy", "Uiso/Ueq (Å²)", "Peak height"],
)

_RUNS_COLUMNS = (
    ["运行 ID", "标签", "模式", "状态", "说明", "创建时间"],
    ["実行 ID", "ラベル", "モード", "状態", "説明", "作成日時"],
    ["Run ID", "Label", "Mode", "State", "Message", "Created"],
)


def _index(lang: str) -> int:
    return LANGS.index(lang) if lang in LANGS else 0


def _pick(table: dict[str, tuple[str, str, str]], key: str, lang: str) -> str:
    row = table.get(key)
    return row[_index(lang)] if row else key


def t(key: str, lang: str = DEFAULT_LANG, **values: object) -> str:
    """One interface string. Unknown keys come back unchanged so a gap is visible, not fatal."""
    text = _pick(_T, key, lang)
    return text.format(**values) if values else text


def phase_label(phase: str, lang: str = DEFAULT_LANG) -> str:
    return _pick(_PHASE, phase, lang)


def stage_label(name: str, lang: str = DEFAULT_LANG) -> str:
    return _pick(_STAGE, name, lang)


def stage_state(state: str, lang: str = DEFAULT_LANG) -> tuple[str, str]:
    icon, labels = _STAGE_STATE.get(state, ("?", (state, state, state)))
    return icon, labels[_index(lang)]


def model_label(value: str, lang: str = DEFAULT_LANG) -> str:
    return _pick(_MODEL, value, lang)


def refinement_label(value: str, lang: str = DEFAULT_LANG) -> str:
    return _pick(_REFINEMENT, value, lang)


def readiness_label(value: str, lang: str = DEFAULT_LANG) -> str:
    return _pick(_READINESS, value, lang)


def alert_meaning(level: str, lang: str = DEFAULT_LANG) -> str:
    return _pick(_ALERT, level, lang)


def issue_message(code: str, field: str, detail: str, lang: str = DEFAULT_LANG) -> str:
    sep = "：" if lang in ("zh", "ja") else ": "
    left, right = ("（", "）") if lang in ("zh", "ja") else (" (", ")")
    return f"{_pick(_FIELD, field, lang)}{sep}{_pick(_ISSUE, code, lang)}{left}{detail}{right}"


def atom_columns(lang: str = DEFAULT_LANG) -> list[str]:
    return list(_ATOM_COLUMNS[_index(lang)])


def runs_columns(lang: str = DEFAULT_LANG) -> list[str]:
    return list(_RUNS_COLUMNS[_index(lang)])


# ------------------------------------------------------------------ native tools and atom assignment
_T.update({
    "native_accordion": ("原生工具状态与阻断原因（静态检查）", "ネイティブツールの状態と阻害要因（静的検査）", "Native tools: status and blockers (static check)"),
    "strict_native": (
        "严格原生模式：拒绝仅含 x86_64 代码的程序，不经 Rosetta 2 运行",
        "厳格ネイティブモード：x86_64 専用のプログラムを拒否し、Rosetta 2 を使わない",
        "Strict native mode: refuse x86_64-only programs; never run through Rosetta 2",
    ),
    "native_policy_strict": ("**策略**：严格原生（Apple 芯片上不使用 Rosetta 2）", "**ポリシー**：厳格ネイティブ（Apple シリコンで Rosetta 2 を使わない）", "**Policy**: strict native (no Rosetta 2 on Apple Silicon)"),
    "native_policy_translation": ("**策略**：允许转译（x86_64 程序需经 Rosetta 2，且 Rosetta 必须已安装并可用）", "**ポリシー**：変換を許可（x86_64 プログラムは Rosetta 2 経由。Rosetta が導入済みで動作する必要があります）", "**Policy**: translation allowed (x86_64 programs need a working Rosetta 2)"),
    "native_host": ("**本机**", "**このマシン**", "**Host**"),
    "native_translated": ("（当前 Python 进程经转译运行）", "（この Python プロセスは変換実行中）", " (this Python process runs translated)"),
    "native_rosetta_markers": ("Rosetta 标记文件：{n} 个（仅为标记，不代表 x86_64 程序能启动）", "Rosetta マーカーファイル：{n} 件（マーカーのみ。x86_64 プログラムが起動する証拠ではありません）", "Rosetta marker files: {n} (markers only, not proof that x86_64 programs launch)"),
    "native_ready": ("✅ 流水线所需程序均通过**静态检查**（只读文件；尚未证明可以启动或计算正确）。", "✅ パイプラインに必要なプログラムはすべて**静的検査**に合格（ファイルを読んだだけで、起動や計算の正しさは未確認）。", "✅ Every program the pipeline needs passes the **static check** (files read only; launching and correct computation are not yet shown)."),
    "native_config_problems": ("⚠️ 工具配置需要检查：", "⚠️ ツール設定を確認してください：", "⚠️ Review the tool configuration:"),
    "native_blocked": ("⛔ **已阻断（静态检查）**：{tools}。请先处理这些问题再进行真实运行；能否启动与计算是否正确仍未验证，不会以模拟结果替代。", "⛔ **ブロック中（静的検査）**：{tools}。実行前にこれらの問題を解消してください。起動と計算の正しさは未確認で、模擬結果で代用しません。", "⛔ **Blocked by static inspection**: {tools}. Resolve these blockers before a real run. Startup and correct computation remain unverified; mock results are never substituted."),
    "native_unavailable": ("无法生成原生工具清单：{detail}", "ネイティブツール一覧を作成できません：{detail}", "The native tool inventory is unavailable: {detail}"),
    "native_missing_pkg": (
        "未安装 crystalx-native-tools（请运行 `sh crystalx_web/scripts/bootstrap_web.sh`）",
        "crystalx-native-tools が未インストールです（`sh crystalx_web/scripts/bootstrap_web.sh` を実行してください）",
        "crystalx-native-tools is not installed (run `sh crystalx_web/scripts/bootstrap_web.sh`)",
    ),
    "col_tool": ("程序", "プログラム", "Program"),
    "col_role": ("用途", "用途", "Role"),
    "col_static_check": ("静态检查", "静的検査", "Static check"),
    "col_launch": ("能否启动", "起動", "Launch"),
    "col_validated": ("实际验证", "実地検証", "Validated"),
    "col_selected": ("将使用的文件", "使用するファイル", "File a run would use"),
    "col_native": ("架构", "アーキテクチャ", "Architecture"),
    "native_blockers": ("**阻断原因与下一步**（技术信息保留原文）", "**阻害要因と次の手順**（技術情報は原文のまま）", "**Blockers and next steps** (technical detail kept verbatim)"),
    "native_checkdef": ("**check.def**：{found}；版本匹配：{match}（启发式）", "**check.def**：{found}；バージョン一致：{match}（ヒューリスティック）", "**check.def**: {found}; version match: {match} (heuristic)"),
    "native_checkdef_none": ("未找到（PLATON 文档说明找不到时会在工作目录自动生成）", "見つかりません（PLATON の文書では、見つからない場合は作業ディレクトリに自動生成されます）", "none found (PLATON is documented to create one in its working directory)"),
    "native_evidence": ("官方 arm64 版本：{build}；源码：{source}", "公式 arm64 版：{build}；ソース：{source}", "Official arm64 build: {build}; source: {source}"),
    "native_note": (
        "> 以上仅为静态检查：只读取文件，不运行任何程序。“能否启动”只能由显式授权的启动探测确认（`crystalx-tools probe <程序> --authorize-launch`，网页不会自动执行）；“实际验证”只能由经授权的真实运行及其解析结果确认。包装程序无法把 x86_64 代码变成 arm64 代码。",
        "> 以上は静的検査のみです。ファイルを読むだけで、プログラムは一切実行しません。「起動」は明示的に許可した起動プローブ（`crystalx-tools probe <プログラム> --authorize-launch`、Web からは自動実行しません）でのみ確認でき、「実地検証」は許可された実データでの実行と出力の解析でのみ確認できます。ラッパーで x86_64 コードを arm64 コードに変えることはできません。",
        "> Static check only: files are read, no program is run. Whether a program **starts** is shown only by the explicitly authorised launch probe (`crystalx-tools probe <program> --authorize-launch`; the Web page never runs it); whether it is **validated** only by an authorised real run with parsed output. A wrapper cannot turn x86_64 code into arm64 code.",
    ),
    "tab_assign": ("原子指认", "原子の帰属", "Atom assignment"),
    "assign_traditional": (
        "**传统流程（可用，需本机 SHELX）**：SHELXT 依据电子密度峰高给出元素，仅作为**起始猜测**；有限的 SHELXL 精修沿用这些元素并精修位置。元素指认未经独立验证。",
        "**従来の流れ（利用可能、ローカルの SHELX が必要）**：SHELXT は電子密度のピーク高さから元素を割り当てますが、これは**初期の推定**にすぎません。制限付き SHELXL 精密化はその元素のまま位置を精密化します。元素の帰属は独立に検証されていません。",
        "**Traditional route (available with local SHELX)**: SHELXT assigns elements from electron-density peak heights — a **starting guess** only; the bounded SHELXL refinement keeps those elements and refines positions. The assignment is not independently verified.",
    ),
    "assign_ai": (
        "**AI 元素指认（CrystalX）：不可用。** {engine}。在正式权重、晶体学预处理与集成验收完成之前，本界面不显示 AI 指认结果，也不显示原子置信度。",
        "**AI による元素帰属（CrystalX）：利用できません。** {engine}。正式な重み・結晶学的前処理・統合の受け入れ確認が終わるまで、この画面は AI の帰属結果も原子の信頼度も表示しません。",
        "**AI element assignment (CrystalX): not available.** {engine}. Until trained weights, crystallographic preprocessing and integration acceptance are done, this page shows no AI assignment and no atom confidence.",
    ),
    "engine_absent": ("Apple 芯片推理后端未安装", "Apple シリコン推論バックエンドは未インストール", "The Apple Silicon inference backend is not installed"),
    "engine_present": (
        "已安装实验性 Apple 芯片推理后端 {version}：仅用合成权重验证了 CPU/MPS 软件路径，尚未接入工作流",
        "実験的な Apple シリコン推論バックエンド {version} がインストール済み：合成重みで CPU/MPS のソフトウェア経路を確認しただけで、ワークフローには未統合",
        "Experimental Apple Silicon inference backend {version} is installed: its CPU/MPS software path was verified with synthetic weights only, and it is not integrated",
    ),
    "assign_elements": ("**SHELXT 初解（解 {letter}）的元素**：{elements}", "**SHELXT 初期解（解 {letter}）の元素**：{elements}", "**Elements in SHELXT's solution {letter}**: {elements}"),
    "assign_refined": ("**精修模型的元素计数**：{counts}", "**精密化モデルの元素数**：{counts}", "**Element counts in the refined model**: {counts}"),
    "assign_no_run": ("尚无运行结果。", "まだ実行結果がありません。", "No run results yet."),
})

_NATIVE_LABELS: dict[str, tuple[str, str, str]] = {
    "ok": ("通过", "合格", "ok"),
    "blocked": ("阻断", "ブロック", "blocked"),
    "not_found": ("未找到", "見つからない", "not found"),
    "not_tested": ("未测试", "未テスト", "not tested"),
    "launched": ("已启动", "起動した", "started"),
    "running_when_stopped": ("已启动（超时后停止）", "起動した（時間切れで停止）", "started (stopped at the limit)"),
    "refused": ("被系统拒绝", "OS が拒否", "refused by the OS"),
    "library_failure": ("缺少库，无法启动", "ライブラリ不足で起動不可", "missing library at start"),
    "killed_at_start": ("启动即被终止", "起動直後に強制終了", "killed at start"),
    "not_verified": ("未验证", "未検証", "not verified"),
    "native_arm64": ("arm64 原生", "arm64 ネイティブ", "native arm64"),
    "universal_with_arm64": ("通用（含 arm64）", "ユニバーサル（arm64 含む）", "universal (has arm64)"),
    "x86_64_only": ("仅 x86_64", "x86_64 のみ", "x86_64 only"),
    "native_for_host": ("本机原生", "このマシンでネイティブ", "native for this host"),
    "script": ("脚本", "スクリプト", "script"),
    "not_macho": ("无法识别", "不明な形式", "unrecognised"),
    "foreign_format": ("macOS 程序（当前不在 macOS 上）", "macOS 用プログラム（現在 macOS ではない）", "macOS program (not on macOS now)"),
    "unknown": ("未知（无可读的权威来源）", "不明（読める公式情報なし）", "unknown (no readable authoritative source)"),
    "built_locally_from_source": ("由用户从源码自行编译", "利用者がソースからビルド", "built locally from source"),
    "distributed_as_executables": ("现行版以可执行程序形式分发（是否另有源码取得途径未确认）", "現行版は実行ファイルとして配布（別のソース入手方法の有無は未確認）", "current versions distributed as executables (other source access not established)"),
    "available_free_of_charge": ("免费提供", "無償で提供", "available free of charge"),
    "match": ("一致", "一致", "match"),
    "mismatch": ("不一致", "不一致", "mismatch"),
}

_BLOCKER_LABELS: dict[str, tuple[str, str, str]] = {
    "translation_refused": ("仅 x86_64：严格原生模式拒绝经 Rosetta 2 运行", "x86_64 のみ：厳格ネイティブモードは Rosetta 2 経由の実行を拒否", "x86_64 only: strict native mode refuses Rosetta 2"),
    "rosetta_missing": ("仅 x86_64，且本机没有可用的 Rosetta 2", "x86_64 のみで、Rosetta 2 が使えない", "x86_64 only and Rosetta 2 is not available"),
    "library_unavailable": ("依赖的动态库缺失或架构不符，程序无法启动", "依存ライブラリが無いかアーキテクチャ不一致で起動不可", "a linked library is missing or has the wrong architecture"),
    "unsigned_arm64": ("arm64 代码没有签名，macOS 会在启动时终止它", "arm64 コードに署名がなく、macOS が起動時に終了させる", "unsigned arm64 code is killed by macOS at launch"),
    "requires_newer_macos": ("为更新的 macOS 构建", "より新しい macOS 向けのビルド", "built for a newer macOS"),
    "native_unverified": ("脚本或无法识别的文件：严格原生模式无法确认它实际运行什么", "スクリプトまたは識別できないファイル：厳格ネイティブモードでは実際に何が動くか確認できない", "script or unrecognised file: strict native mode cannot verify what it runs"),
    "not_an_executable": ("不是可执行主程序（例如动态库）", "実行可能なメインプログラムではない（ライブラリなど）", "not a main executable (e.g. a library)"),
    "wrong_platform": ("为其他 Apple 平台构建，不是 macOS", "macOS ではなく別の Apple プラットフォーム向け", "built for another Apple platform, not macOS"),
    "malformed_binary": ("文件结构损坏或不完整", "ファイル構造が壊れているか不完全", "damaged or incomplete file structure"),
    "not_found": ("未找到", "見つからない", "not found"),
    "not_executable": ("没有可执行权限", "実行権限がない", "not executable"),
    "quarantined": ("带有 macOS 隔离属性", "macOS の隔離属性付き", "carries the macOS quarantine flag"),
    "arch_mismatch": ("架构与本机不符", "アーキテクチャがこのマシンと不一致", "wrong architecture for this host"),
}


def native_label(value: str, lang: str = DEFAULT_LANG) -> str:
    return _pick(_NATIVE_LABELS, value, lang)


def blocker_label(code: str, lang: str = DEFAULT_LANG) -> str:
    return _pick(_BLOCKER_LABELS, code, lang)


# ---------------------------------------------------------------- desktop start: first use, stopping
_T.update({
    "quick_title": ("### 快速开始 · 合成演示（MOCK）", "### クイックスタート · 合成デモ（MOCK）", "### Quick start · synthetic demo (MOCK)"),
    "quick_text": (
        "点击下方按钮即可走完整个流程：载入内置的**虚构**晶体结构，由替身程序代替 SHELXT、SHELXL、PLATON，依次显示诊断、初解、精修、校验、三维模型和导出。**不需要** SHELX、PLATON、XQuartz 或模型权重，也不联网。所有数字都是占位值，**不是真实结果**；模拟模式不接受你上传的真实数据。",
        "下のボタンで一連の流れを体験できます：内蔵の**架空**の結晶構造を読み込み、SHELXT・SHELXL・PLATON の代わりに代替プログラムが動き、診断・初期解・精密化・検証・3D モデル・エクスポートを順に表示します。SHELX、PLATON、XQuartz、モデルの重みは**不要**で、ネットワークも使いません。数値はすべてプレースホルダーで**実際の結果ではありません**。模擬モードはアップロードされた実データを受け付けません。",
        "The button below walks through the whole flow: it loads the built-in **fictional** structure and runs stand-ins for SHELXT, SHELXL and PLATON, showing diagnosis, initial solution, refinement, checks, the 3-D model and export. SHELX, PLATON, XQuartz and model weights are **not needed**, and nothing goes to the network. Every number is a placeholder — **not a real result** — and mock mode does not accept your own data.",
    ),
    "quick_button": ("▶ 一键运行合成演示（MOCK）", "▶ 合成デモを実行（MOCK）", "▶ Run the synthetic demo (MOCK)"),
    "quick_real_ready": (
        "**真实计算**：工具静态检查通过（尚未启动或验证任何程序）。要计算你自己的数据，请上传 INS/HKL 并选择“本机真实程序”；严格原生检查保持开启。",
        "**実計算**：ツールの静的チェックは通過しています（プログラムはまだ起動も検証もしていません）。自分のデータを計算するには INS/HKL をアップロードし「ローカルの実プログラム」を選びます。厳格ネイティブチェックは有効のままです。",
        "**Real computations**: the tools pass the static check (no program has been started or validated yet). To compute your own data, upload INS/HKL and choose \"Real local programs\"; strict native checks stay on.",
    ),
    "quick_real_blocked": (
        "**真实计算**：目前被阻断（{tools}）。原因和下一步见下方“原生工具状态与阻断原因”；严格原生检查保持开启，不会为了显示可用而自动放宽。",
        "**実計算**：現在ブロックされています（{tools}）。理由と次の手順は下の「ネイティブツールの状態と阻害要因」を参照してください。厳格ネイティブチェックは有効のままで、使えるように見せるために自動で緩めることはありません。",
        "**Real computations**: currently blocked ({tools}). Reasons and next steps are under \"Native tools: status and blockers\" below; strict native checks stay on and are never relaxed just to look ready.",
    ),
    "quick_real_unknown": ("**真实计算**：无法读取工具状态（{detail}）。", "**実計算**：ツールの状態を読み取れません（{detail}）。",
                           "**Real computations**: tool status unavailable ({detail})."),
    "stop_server": ("停止本地服务", "ローカルサービスを停止", "Stop local server"),
    "stop_confirm": ("确认停止（进行中的运行会被取消，记录保留）", "停止を確定（実行中の処理は取り消され、記録は残ります）",
                     "Confirm stop (an active run is cancelled; records are kept)"),
    "stop_keep": ("继续运行", "続ける", "Keep running"),
    "stop_done": (
        "本地服务正在停止。可以关闭此页面；运行记录保留在 `{runs}`。需要时再次双击启动器即可。",
        "ローカルサービスを停止しています。このページは閉じて構いません。実行記録は `{runs}` に残ります。必要なときはランチャーをもう一度ダブルクリックしてください。",
        "The local server is stopping. You can close this page; run records stay in `{runs}`. Double-click the launcher again whenever you need it.",
    ),
    "stop_unavailable": ("此页面不是由桌面启动器启动的，无法从这里停止。", "このページはデスクトップランチャーから起動されていないため、ここからは停止できません。",
                         "This page was not started by the desktop launcher, so it cannot be stopped from here."),
})
