"""Text for the iterative-reconstruction tab, in Chinese, Japanese and English.

Backend codes (stop codes, blockers, policy field names) are always shown verbatim next to these
words. The explanations say what a code means for the run; none of them turns a numerical stop
into a scientific result.
"""

from __future__ import annotations

from crystalx_web.i18n import DEFAULT_LANG, LANGS
from crystalx_core.app.reconstruction_stops import STOP_CODES

__all__ = ["rt", "field_label", "choice_label", "stop_explanation", "stop_codes_without_text", "history_columns",
           "run_columns", "decision_label", "DECISIONS"]


def _i(lang: str) -> int:
    return LANGS.index(lang) if lang in LANGS else LANGS.index(DEFAULT_LANG)


# key: (zh, ja, en)
_T: dict[str, tuple[str, str, str]] = {
    "device_line": ("推理设备：请求 {requested}，本次运行实际使用 {used}（{reason}）",
                    "推論デバイス：指定 {requested}、この実行で実際に使用 {used}（{reason}）",
                    "Inference device: requested {requested}, used in this run {used} ({reason})"),
    "device_blocked_line": ("设备选择被阻止：{code}", "デバイス選択がブロックされた：{code}", "Device selection blocked: {code}"),
    "heavy_prep": ("CrystalX 指认方式", "CrystalX 割り当て方式", "CrystalX assignment recipe"),
    "preview_heavy_stage": ("将运行“初解重原子阶段”：在 SHELXT 初解上按上游方式指认一次，再做一次各向异性 L.S. 16 精修；不进入迭代循环。",
                            "「初期重原子段階」を実行：SHELXT 初期解に上流の方式で 1 回割り当て、異方性 L.S. 16 精密化を 1 回行う。反復ループには入らない。",
                            "Runs the initial heavy stage: one upstream-style assignment on the SHELXT solution, then one anisotropic L.S. 16 refinement; no iterative rounds."),
    "stages_head": ("**固定阶段**", "**固定段階**", "**Fixed stages**"),
    "stage_line": ("- {stage}：SHELXL {cycles} 轮，各向异性 {aniso}；R1 {r1}，wR2 {wr2}，GooF {goof}，参数 {params}",
                   "- {stage}：SHELXL {cycles} サイクル、異方性 {aniso}；R1 {r1}、wR2 {wr2}、GooF {goof}、パラメータ {params}",
                   "- {stage}: SHELXL {cycles} cycles, anisotropic {aniso}; R1 {r1}, wR2 {wr2}, GooF {goof}, parameters {params}"),
    "heavy_stage_note": ("  这只是已发表 CrystalX 方法的重原子部分：还没有加氢阶段和权重精修，与上游代码的逐一对照也尚未完成。",
                         "  これは公開された CrystalX 手法の重原子部分のみ：水素段階と重み精密化はまだなく、上流コードとの一対一の照合も未完了。",
                         "  This is only the heavy-atom part of the published CrystalX method: no hydrogen stage or weighting yet, and parity with the upstream code is not yet verified."),
    "tab_solve": ("单次求解", "単発解析", "One-shot solve"),
    "tab_recon": ("迭代重构（预训练指认 + SHELXL 循环）", "反復再構成（事前学習割り当て＋SHELXL ループ）",
                  "Iterative reconstruction (pretrained assignment + SHELXL rounds)"),
    "recon_intro": (
        "### 迭代重构\n初解（三选一）→ 预训练 CrystalX 指认元素 → 多轮 SHELXL 精修，按数值规则停止或跑满固定轮数。"
        "结果是**未经验证的数值候选**：可以查看、下载，再单独做本机 checkCIF/PLATON 诊断。"
        "数值稳定不代表结构正确；本页不接受专家结构或参考答案作为求解输入。",
        "### 反復再構成\n初期解（3 方式から 1 つ）→ 事前学習済み CrystalX による元素割り当て → SHELXL 精密化の反復。数値規則で停止するか、"
        "指定ラウンド数で終了します。結果は**未検証の数値候補**で、閲覧・ダウンロードと、別途ローカル checkCIF/PLATON 診断ができます。"
        "数値的な安定は構造が正しいことを意味しません。このページは専門家の構造や参照解を求解入力として受け付けません。",
        "### Iterative reconstruction\nInitial route (one of three) → pretrained CrystalX element assignment → repeated SHELXL "
        "rounds, stopped by the numerical rule or after a fixed number of rounds. The result is an **unvalidated numerical "
        "candidate**: inspect and download it, then run local checkCIF/PLATON diagnostics separately. Numerical stability "
        "does not mean the structure is correct; this page accepts no expert structure or reference answer as a solve input.",
    ),
    "recon_unavailable": (
        "⛔ 这次启动没有开发用的仓库检出（需要 Git 检出里被忽略的 `runs/`、本机 SHELX 和预训练权重），迭代重构不可用。"
        "请用仓库里的 `CrystalX Web Demo (developer).command` 启动。单次求解不受影响。",
        "⛔ この起動には開発用のリポジトリのチェックアウトがありません（Git チェックアウト内の無視された `runs/`、ローカルの SHELX、"
        "事前学習済みの重みが必要）。反復再構成は使えません。リポジトリの `CrystalX Web Demo (developer).command` から起動してください。単発解析には影響しません。",
        "⛔ This start has no developer checkout (iterative reconstruction needs an ignored `runs/` inside a Git checkout, "
        "local SHELX programs and the pretrained weights), so it is unavailable here. Start from the repository's "
        "`CrystalX Web Demo (developer).command`. The one-shot solve is unaffected.",
    ),
    "service_error": ("⛔ 无法启动重构服务：{detail}", "⛔ 再構成サービスを開始できません：{detail}",
                      "⛔ The reconstruction service could not start: {detail}"),
    "sec_recon_input": ("#### 1 · 输入与初解路线", "#### 1 · 入力と初期解ルート", "#### 1 · Inputs and initial route"),
    "ins_original": ("原始 INS 指令文件（不含原子）", "元の INS 命令ファイル（原子なし）", "Original INS instruction file (no atoms)"),
    "hkl_measured": ("测量的 HKL 衍射数据", "測定 HKL 回折データ", "Measured HKL reflection data"),
    "strategy": ("初解路线（每次运行只选一条，失败不会自动换路线）", "初期解ルート（1 回の実行で 1 つ。失敗しても自動では切り替えない）",
                 "Initial route (one per run; a failed route is never swapped for another)"),
    "small_solver": ("小分子求解程序", "小分子ソルバー", "Small-molecule solver"),
    "small_method": ("SHELXS 方法", "SHELXS の方法", "SHELXS method"),
    "trials": ("初解尝试次数", "初期解の試行回数", "Initial search trials"),
    "find_sites_small": ("SHELXD FIND 数（所有者给定，必填于 SHELXD）", "SHELXD の FIND 数（所有者が指定、SHELXD では必須）",
                         "SHELXD FIND count (owner-supplied; required for SHELXD)"),
    "plop": ("SHELXD PLOP 计数（逗号分隔，最多 10 个）", "SHELXD PLOP カウント（カンマ区切り、最大 10）",
             "SHELXD PLOP counts (comma-separated, at most 10)"),
    "route_timeout": ("初解超时（秒）", "初期解のタイムアウト（秒）", "Initial-route timeout (s)"),
    "protein_head": (
        "**蛋白质实验定相（SAD/MAD）**：必须显式声明蛋白质、实验类型、空间群、重原子和数据集角色。普通小分子数据不会被自动当作 SAD/MAD。"
        "MTZ 需要先转换。输出的 `*_fa.res` 只是重原子亚结构，不是完整蛋白模型。",
        "**タンパク質の実験的位相決定（SAD/MAD）**：タンパク質であること、実験の種類、空間群、重原子、データセットの役割を明示する必要があります。"
        "通常の小分子データが自動で SAD/MAD 扱いされることはありません。MTZ は事前に変換が必要です。出力 `*_fa.res` は重原子部分構造で、完全なタンパク質モデルではありません。",
        "**Protein experimental phasing (SAD/MAD)**: protein status, experiment type, space group, heavy element and dataset "
        "roles must be declared explicitly. An ordinary small-molecule upload is never treated as SAD/MAD. MTZ needs "
        "conversion first. The `*_fa.res` output is a heavy-atom substructure, not a complete protein model.",
    ),
    "protein_confirmed": ("我确认这是蛋白质数据", "タンパク質データであることを確認", "I confirm this is protein data"),
    "experiment": ("实验类型", "実験の種類", "Experiment"),
    "space_group": ("空间群（声明值）", "空間群（宣言値）", "Space group (declared)"),
    "protein_find_sites": ("亚结构位点数（FIND）", "部分構造サイト数（FIND）", "Substructure sites (FIND)"),
    "heavy_element": ("重原子元素", "重原子元素", "Heavy element"),
    "solvent_fraction": ("溶剂比例（0–1 之间）", "溶媒の割合（0〜1）", "Solvent fraction (between 0 and 1)"),
    "dataset_role": ("数据集 {n} 角色", "データセット {n} の役割", "Dataset {n} role"),
    "dataset_file": ("数据集 {n} 文件", "データセット {n} のファイル", "Dataset {n} file"),
    "dataset_format": ("数据集 {n} 格式", "データセット {n} の形式", "Dataset {n} format"),
    "dataset_friedel": ("数据集 {n} 保留了 Friedel 对", "データセット {n} は Friedel 対を保持", "Dataset {n} keeps Friedel pairs"),
    "trace_cycles": ("SHELXE 追踪循环数", "SHELXE トレースサイクル数", "SHELXE trace cycles"),
    "density_cycles": ("SHELXE 密度修饰循环数", "SHELXE 密度修正サイクル数", "SHELXE density-modification cycles"),
    "heavy_atoms_in_native": ("Native 数据中含重原子", "Native データに重原子を含む", "Heavy atoms present in native"),
    "preferred_hand": ("SHELXE 手性选择", "SHELXE のハンド選択", "SHELXE hand"),
    "metal_acc": ("可选：仅金属锚定（默认关闭）", "任意：金属のみのアンカー（既定はオフ）", "Optional: metal-only anchoring (off by default)"),
    "metal_note": (
        "开启后：保留初解中选定的金属位点，从派生副本里删除其他原子，并在之后的 SHELXL 轮次中固定这些金属的坐标和元素。"
        "这不是已验证的骨架；相位每轮由部分模型重新计算，不是保留实验相位。金属不在初解里时会显示阻断原因，不会用导师坐标或答案补上。",
        "オンにすると：初期解で選んだ金属サイトを残し、派生コピーから他の原子を削除し、以降の SHELXL ラウンドでその金属の座標と元素を固定します。"
        "検証済みの骨格ではありません。位相は毎ラウンド部分モデルから再計算され、実験位相を保持するものではありません。金属が初期解に無い場合は阻害理由を表示し、指導者の座標や答えで補うことはしません。",
        "When on: the selected metal sites of the initial model are kept, every other atom is removed from a derivative "
        "copy, and the metals' coordinates and elements are fixed in later SHELXL rounds. This is not a validated "
        "scaffold; phases are recomputed from the partial model each round, not retained experimental phases. If no "
        "requested metal is in the initial model, the blocker is shown; no mentor coordinates or answer fill the gap.",
    ),
    "metal_only": ("仅金属锚定", "金属のみのアンカー", "Metal-only anchoring"),
    "anchor_elements": ("锚定金属（逗号分隔；留空 = 后端定义的金属集合）", "アンカー金属（カンマ区切り。空欄 = バックエンド定義の金属集合）",
                        "Anchor metals (comma-separated; blank = the backend's metal set)"),
    "candidate_elements": ("候选元素（可选，逗号分隔，至少两个非氢元素）", "候補元素（任意、カンマ区切り、非水素元素 2 つ以上）",
                           "Candidate elements (optional, comma-separated, at least two non-hydrogen)"),
    "candidate_ph": ("留空 = 按下方来源；这是化学先验，不是 AI 识别", "空欄 = 下の取得元に従う。化学的事前情報であり AI 認識ではない",
                     "blank = the source below; a chemical prior, not AI recognition"),
    "candidate_source": ("候选元素来源（留空时）", "候補元素の取得元（空欄時）", "Candidate-element source (when blank)"),
    "rounds_acc": ("轮数、容差与时间", "ラウンド数・許容値・時間", "Rounds, tolerances and time"),
    "protocol_acc": ("指认协议与停止规则", "割り当てプロトコルと停止規則", "Assignment protocol and stopping"),
    "safeguards_acc": ("拒绝后的退让与锚点检查", "棄却後の後退とアンカー確認", "Rejection backoff and anchor checks"),
    "refinement_acc": ("精修方法与稳定性（默认保持全矩阵 L.S.）", "精密化の方法と安定性（既定は全行列 L.S.）",
                       "Refinement method and stability (default: full-matrix L.S.)"),
    "peaks_acc": ("高级：新峰与拒绝规则", "詳細：新ピークと棄却規則", "Advanced: new peaks and rejection"),
    "adaptive_acc": ("实验性：自适应策略（默认全部关闭）", "実験的：適応ポリシー（既定はすべてオフ）", "Experimental: adaptive policy (all off by default)"),
    "adaptive_note": (
        "这些开关和阈值来自后端的 `AdaptivePolicy`，原样传给后端。开发用的冻结配置不会自动成为默认值。",
        "これらのスイッチと閾値はバックエンドの `AdaptivePolicy` のもので、そのまま渡されます。開発用の凍結設定が自動で既定値になることはありません。",
        "These switches and thresholds are the backend's `AdaptivePolicy`, passed through unchanged. A frozen development "
        "configuration never silently becomes the default.",
    ),
    "other_acc": ("其他后端字段", "その他のバックエンド項目", "Other backend fields"),
    "device": ("推理设备（运行中不回退）", "推論デバイス（実行中のフォールバックなし）", "Inference device (no runtime fallback)"),
    "label_field": ("运行标签（可选）", "実行ラベル（任意）", "Run label (optional)"),
    "btn_recon_preview": ("预检（不运行程序）", "事前確認（プログラムは実行しない）", "Preview (runs nothing)"),
    "btn_recon_start": ("开始重构", "再構成を開始", "Start reconstruction"),
    "btn_recon_cancel": ("请求取消", "取り消しを要求", "Request cancel"),
    "sec_recon_status": ("#### 2 · 预检、进度与结果", "#### 2 · 事前確認・進捗・結果", "#### 2 · Preview, progress and result"),
    "history_label": ("每轮记录（后端原始数值；R1 为分数）", "ラウンドごとの記録（バックエンドの値。R1 は分率）",
                      "Per-round record (backend values; R1 as a fraction)"),
    "viewer_tab": ("候选模型 3D", "候補モデル 3D", "Candidate 3-D"),
    "downloads": ("下载封存的候选文件（私有副本）", "封印済み候補ファイルのダウンロード（非公開コピー）", "Sealed candidate files (private copies)"),
    "btn_export": ("准备下载与 3D 视图", "ダウンロードと 3D 表示を準備", "Prepare downloads and 3-D view"),
    "export_failed": ("⛔ 无法读取封存的候选：{detail}", "⛔ 封印済み候補を読み取れません：{detail}", "⛔ The sealed candidate could not be read: {detail}"),
    "export_none": ("这次运行没有封存的候选。", "この実行には封印済み候補がありません。", "This run has no sealed candidate."),
    "checkcif_acc": ("单独的本机 checkCIF / PLATON 诊断", "別途のローカル checkCIF / PLATON 診断", "Separate local checkCIF / PLATON diagnostics"),
    "allow_builtin": ("允许 PLATON 使用其内置 check.def", "PLATON の組み込み check.def の使用を許可", "Allow PLATON's built-in check.def"),
    "checkdef_file": ("或提供 check.def 文件", "または check.def ファイルを指定", "Or provide a check.def file"),
    "platon_timeout": ("PLATON 超时（秒，≤ 3600）", "PLATON のタイムアウト（秒、≤ 3600）", "PLATON timeout (s, ≤ 3600)"),
    "btn_checkcif": ("运行本机 checkCIF / PLATON", "ローカル checkCIF / PLATON を実行", "Run local checkCIF / PLATON"),
    "runs_acc": ("重构运行记录", "再構成の実行記録", "Reconstruction runs"),
    "btn_runs_refresh": ("刷新", "更新", "Refresh"),
    "load_run": ("运行编号", "実行 ID", "Run id"),
    "btn_load": ("载入", "読み込む", "Load"),
    "cancel_requested": ("已送出取消请求；清理完成前状态可能仍显示运行中。", "取り消し要求を送りました。後片付けが終わるまで実行中と表示されることがあります。",
                         "Cancel request delivered; the state may show running until cleanup finishes."),
    "cancel_not_active": ("这个运行不在本页面的活动任务里，无法取消（可能已结束或属于其他进程）。",
                          "この実行はこのページの実行中タスクにないため取り消せません（終了済みか別プロセスの可能性）。",
                          "This run is not an active job of this page, so it cannot be cancelled (it may have ended or belong to another process)."),
    "no_selection": ("先运行或载入一个重构运行。", "先に再構成を実行するか読み込んでください。", "Run or load a reconstruction first."),
    "recon_mode": ("运行模式", "実行モード", "Run mode"),
    "recon_mock_banner": (
        "🧪 <b>合成演示（MOCK）</b>：只接受内置的虚构数据；初解、元素指认和 SHELXL 都由替身程序代替，不读取权重，不运行真实程序。"
        "所有数字都是占位值，<b>不是真实结果</b>。",
        "🧪 <b>合成デモ（MOCK）</b>：内蔵の架空データのみ受け付けます。初期解・元素割り当て・SHELXL はすべて代替プログラムで、重みは読み込まず、実プログラムも実行しません。"
        "数値はすべてプレースホルダーで、<b>実際の結果ではありません</b>。",
        "🧪 <b>Synthetic demo (MOCK)</b>: only the built-in fictional data is accepted; the initial solution, element assignment and "
        "SHELXL are stand-ins, no weights are loaded and no real program runs. Every number is a placeholder — <b>not a real result</b>.",
    ),
    "btn_recon_demo": ("载入内置合成数据（虚构结构）", "内蔵の合成データを読み込む（架空の構造）", "Load the built-in synthetic data (fictional structure)"),
    "mock_result_note": ("🧪 **合成演示（MOCK）**：替身程序的输出，数字不是真实结果。", "🧪 **合成デモ（MOCK）**：代替プログラムの出力で、数値は実際の結果ではありません。",
                         "🧪 **Synthetic demo (MOCK)**: stand-in output; the numbers are not real results."),
    "demo_unavailable": ("这次启动没有可用的合成演示。", "この起動では合成デモを利用できません。", "The synthetic demo is not available in this start."),
    "contract_mismatch": (
        "⚠️ 后端接口约定版本是 {backend}，本页面按 {page} 编写。字段和停止码可能对不上，请先让前端核对后再使用。",
        "⚠️ バックエンドのインターフェース規約は {backend} ですが、このページは {page} に合わせて作られています。項目や停止コードが一致しない可能性があります。フロントエンドの確認後に使用してください。",
        "⚠️ The backend contract is {backend}; this page was written against {page}. Fields and stop codes may not match; have the frontend reviewed before relying on it.",
    ),
    "rejections_line": ("被拒的不同提议 {distinct} 个；同一组动作再次被拒 {repeated} 次；跳过的重复提议或动作 {skipped} 次",
                        "棄却された異なる提案 {distinct} 件、同じ操作の再棄却 {repeated} 回、スキップした重複提案・操作 {skipped} 回",
                        "{distinct} distinct proposals rejected; {repeated} repeats of a rejected action set; {skipped} duplicate proposals or actions skipped"),
    "stop_detail_line": ("详情（后端原文）：{detail}", "詳細（バックエンドの原文）：{detail}", "Detail (backend text): {detail}"),
    # ------------------------------------------------------------- adapter output
    "delta_unavailable": ("—（无前一轮）", "—（前ラウンドなし）", "— (no previous round)"),
    "preview_hint": ("点“预检”查看所选路线需要的程序和阻断原因；预检不运行任何程序。",
                     "「事前確認」で選んだルートに必要なプログラムと阻害要因を表示します。プログラムは実行しません。",
                     "Use Preview to see the programs the selected route needs and any blockers; it runs nothing."),
    "preview_title": ("**预检结果**", "**事前確認の結果**", "**Preview**"),
    "preview_strategy": ("路线：{strategy}", "ルート：{strategy}", "Route: {strategy}"),
    "preview_steps": ("步骤：{steps}", "手順：{steps}", "Steps: {steps}"),
    "preview_tools_head": ("所需程序（静态检查，未启动）：", "必要なプログラム（静的検査、未起動）：", "Required programs (static check, not started):"),
    "preview_blocked": ("⛔ **被阻断**，原因（后端代码）：", "⛔ **ブロック**、理由（バックエンドのコード）：", "⛔ **Blocked**, reasons (backend codes):"),
    "preview_ready_static": ("✅ 静态检查通过。这不是启动测试，也不代表推理或科学结果。", "✅ 静的検査は通過。起動テストでも推論・科学的結果でもありません。",
                             "✅ Static check passed. This is not a launch test, nor an inference or scientific result."),
    "preview_warnings": ("提示：", "注意：", "Warnings:"),
    "preview_scope": ("checkCIF：单独进行，尚未评估。专家参考：未使用。", "checkCIF：別途実施、未評価。専門家の参照：未使用。",
                      "checkCIF: separate, not evaluated. Expert reference: not used."),
    "phase_queued": ("排队中", "待機中", "queued"),
    "phase_running": ("运行中", "実行中", "running"),
    "phase_completed": ("正常结束（含义见停止码）", "正常終了（意味は停止コードを参照）", "ended normally (see the stop code)"),
    "phase_blocked": ("被阻断", "ブロック", "blocked"),
    "phase_failed": ("未完成（失败或安全上限）", "未完了（失敗または安全上限）", "not completed (failure or safety limit)"),
    "phase_cancelled": ("已取消", "取り消し", "cancelled"),
    "phase_interrupted": ("已中断（所属进程不在了）", "中断（担当プロセスが存在しない）", "interrupted (owning process is gone)"),
    "no_recon_run": ("还没有重构运行。", "再構成の実行はまだありません。", "No reconstruction run yet."),
    "status_line": ("运行 {run_id} · {phase} · 第 {round} 轮 · 推理 {inference} 次 · 程序调用 {tools} 次 · 连续稳定 {stable}",
                    "実行 {run_id} · {phase} · 第 {round} ラウンド · 推論 {inference} 回 · プログラム呼び出し {tools} 回 · 連続安定 {stable}",
                    "Run {run_id} · {phase} · round {round} · {inference} inference calls · {tools} program calls · {stable} stable in a row"),
    "stop_line": ("停止码 {code}：{meaning}", "停止コード {code}：{meaning}", "Stop code {code}: {meaning}"),
    "message_line": ("后端消息：{message}", "バックエンドのメッセージ：{message}", "Backend message: {message}"),
    "blockers_head": ("阻断原因（后端代码）：", "阻害要因（バックエンドのコード）：", "Blockers (backend codes):"),
    "interrupted_note": ("所属进程已不在；不会自动重放或续跑。", "担当プロセスがありません。自動の再実行や再開はありません。",
                         "The owning process is gone; nothing is replayed or resumed automatically."),
    "not_validated_note": ("_数值候选，未经科学验证；checkCIF 需单独运行。_", "_数値候補で、科学的には未検証です。checkCIF は別途実行してください。_",
                           "_Numerical candidate, not scientifically validated; run checkCIF separately._"),
    "yes": ("是", "はい", "yes"),
    "no": ("否", "いいえ", "no"),
    "label_fixed_rounds": ("固定轮数已完成", "固定ラウンド数を完了", "fixed rounds completed"),
    "label_numerically_stable": ("数值稳定的候选", "数値的に安定した候補", "numerically stable candidate"),
    "label_unfinished": ("未达到停止条件时保留的候选", "停止条件に達しないまま保持された候補", "candidate kept without meeting a stop condition"),
    "summary_pending": ("运行结束后在这里显示摘要。", "実行が終わるとここに要約を表示します。", "The summary appears here when the run ends."),
    "summary_title": ("**运行摘要**", "**実行の要約**", "**Run summary**"),
    "summary_phase": ("状态：{phase} · 停止码 {code}：{meaning}", "状態：{phase} · 停止コード {code}：{meaning}", "State: {phase} · stop code {code}: {meaning}"),
    "candidate_available": ("候选：{label} · 封印 {seal}… · 候选文件 {sha}…", "候補：{label} · 封印 {seal}… · 候補ファイル {sha}…",
                            "Candidate: {label} · seal {seal}… · candidate file {sha}…"),
    "candidate_from_unfinished_run": ("⚠️ 这次运行没有满足停止条件，但保留了已封存的候选。它不是收敛结果。",
                                      "⚠️ この実行は停止条件を満たしていませんが、封印済み候補は保持されています。収束した結果ではありません。",
                                      "⚠️ This run did not meet a stop condition, but its sealed candidate is kept. It is not a converged result."),
    "candidate_missing": ("没有产生候选文件。", "候補ファイルはありません。", "No candidate file was produced."),
    "initial_line": ("初解状态 {status} · 程序调用 {calls} 次", "初期解の状態 {status} · プログラム呼び出し {calls} 回", "Initial route {status} · {calls} program calls"),
    "initial_reuse_line": ("初解来自已固定哈希的先前自动结果（不是退款，也不是续跑）。", "初期解はハッシュ固定済みの以前の自動結果から（返金でも再開でもない）。",
                           "The initial model came from a hash-pinned earlier automatic result (not a refund or a resume)."),
    "waivers_line": ("按所有者 2026-11-10 的决定暂缓的科学关口：{gates}", "所有者の 2026-11-10 の決定により保留した科学ゲート：{gates}",
                     "Scientific gates deferred by the owner's 2026-11-10 decision: {gates}"),
    "nonpositive_u_warning": ("⚠️ 第 {round} 轮：Uiso ≤ 0 的位点按原值保留、只用于几何推理，未获科学认可：{sites}",
                              "⚠️ 第 {round} ラウンド：Uiso ≤ 0 のサイトを値を変えずに保持し、幾何のみの推論に使用（科学的には未承認）：{sites}",
                              "⚠️ Round {round}: sites with Uiso ≤ 0 kept unchanged for geometry-only inference, not scientifically approved: {sites}"),
    "error_line": ("后端异常类型：{error}（详情见运行目录）", "バックエンドの例外の種類：{error}（詳細は実行ディレクトリ）",
                   "Backend exception type: {error} (details in the run directory)"),
    "viewer_source": ("AI 指认 + SHELXL 精修的候选（未经验证）", "AI 割り当て＋SHELXL 精密化の候補（未検証）",
                      "AI-assigned, SHELXL-refined candidate (unvalidated)"),
    "viewer_element_note": ("元素由预训练 CrystalX 模型指认后经 SHELXL 精修，未经独立验证。",
                            "元素は事前学習済み CrystalX モデルが割り当て、SHELXL で精密化したもので、独立には検証されていません。",
                            "Elements were assigned by the pretrained CrystalX model and refined by SHELXL; they are not independently verified."),
    "checkcif_hint": ("候选封存后可在这里单独运行本机 PLATON 诊断。它不会自动运行，也不会改变候选。",
                      "候補の封印後、ここで別途ローカル PLATON 診断を実行できます。自動実行はされず、候補も変更しません。",
                      "Once a candidate is sealed, local PLATON diagnostics can be run here separately. They never run automatically and never change the candidate."),
    "checkcif_status": ("诊断状态 {status}：{meaning}", "診断の状態 {status}：{meaning}", "Diagnostic status {status}: {meaning}"),
    "checkcif_NOT_EVALUATED": ("未评估（没有可评估的报告；不是“零警告”）", "未評価（評価できるレポートなし。「警告ゼロ」ではない）",
                               "not evaluated (no report was evaluated; this is not \"zero alerts\")"),
    "checkcif_FAILED": ("失败（候选仍保留）", "失敗（候補は保持）", "failed (the candidate is kept)"),
    "checkcif_ALERTS_RECORDED": ("已记录警告，需要专家复核", "警告を記録、専門家の確認が必要", "alerts recorded; expert review required"),
    "checkcif_counts": ("警告计数 A {A} · B {B} · C {C} · G {G}", "警告数 A {A} · B {B} · C {C} · G {G}", "Alert counts A {A} · B {B} · C {C} · G {G}"),
    "checkcif_no_report": ("没有读取到报告，因此不显示警告计数。", "レポートを読めていないため、警告数は表示しません。", "No report was read, so no alert counts are shown."),
    "checkcif_program": ("程序 {program} · 可执行文件 {sha}… · 调用 {calls} 次 · 启动与输出已核实：{verified}",
                         "プログラム {program} · 実行ファイル {sha}… · 呼び出し {calls} 回 · 起動と出力の確認：{verified}",
                         "Program {program} · executable {sha}… · {calls} calls · startup and output verified: {verified}"),
    "checkcif_candidate_kept": ("候选仍可用。", "候補は引き続き利用できます。", "The candidate remains available."),
    "checkcif_candidate_lost": ("⛔ 封印校验失败，候选不可再用于诊断。", "⛔ 封印の検証に失敗。候補は診断に使えません。", "⛔ Seal verification failed; the candidate cannot be used for diagnostics."),
    "checkcif_scope": ("_这是本机 PLATON 诊断，不是 IUCr 在线 checkCIF；不会上传任何数据。没有警告也不等于通过。_",
                       "_これはローカルの PLATON 診断で、IUCr のオンライン checkCIF ではありません。データはアップロードされません。警告が無くても合格ではありません。_",
                       "_This is local PLATON diagnostics, not the IUCr online checkCIF; nothing is uploaded. No alerts is not a pass._"),
}

_HISTORY_COLUMNS = (
    ("轮", "ラウンド", "Round"), ("R1", "R1", "R1"), ("ΔR1", "ΔR1", "ΔR1"),
    ("最大位移 Å", "最大移動 Å", "Max shift Å"), ("+位点", "+サイト", "+sites"), ("−位点", "−サイト", "−sites"),
    ("元素改动", "元素変更", "Element changes"), ("稳定", "安定", "Stable"), ("决定", "判定", "Decision"),
    ("推理协议", "推論プロトコル", "Protocol"), ("退让级", "後退レベル", "Backoff"),
    ("精修方法", "精密化法", "Method"), ("精修净位移最大 Å", "精密化の最大正味移動 Å", "Max net shift Å"),
    ("位移超阈值位点", "閾値超えサイト", "Sites over threshold"),
    ("列出的峰", "列挙ピーク", "Peaks listed"), ("接受的新位点", "採用した新サイト", "New sites accepted"),
    ("主要拒绝原因", "主な却下理由", "Main rejection reason"),
)
_RUN_COLUMNS = (("运行编号", "実行 ID", "Run id"), ("标签", "ラベル", "Label"), ("状态", "状態", "Phase"),
                ("轮数", "ラウンド", "Round"), ("停止码", "停止コード", "Stop code"))

# Policy field labels, keyed by record path. A backend field without an entry fails the label test.
_FIELDS: dict[str, tuple[str, str, str]] = {
    "cycle.fourier_map": ("傅里叶图（2=差值，3=Fo，4=2Fo−Fc）", "フーリエ図（2=差、3=Fo、4=2Fo−Fc）", "Fourier map (2=difference, 3=Fo, 4=2Fo−Fc)"),
    "cycle.anisotropic_metals": ("仅金属使用各向异性 ADP（实验性）", "金属のみ異方性 ADP（実験的）", "Anisotropic ADPs for metals only (experimental)"),
    "cycle.metal_ripple_exclusion_angstrom": ("金属附近峰排除距离（Å，0=关闭）", "金属付近ピークの除外距離（Å、0=無効）", "Metal peak exclusion radius (Å; 0=off)"),
    "heavy_preparation": ("重原子准备路线", "重原子準備経路", "Heavy preparation route"),
    "cycle.max_refinement_net_shift_angstrom": ("单轮最大净位移（Å；留空关闭；超限整轮回滚）", "1 ラウンドの最大正味移動量（Å。空欄で無効、超過時はラウンド全体を戻す）",
                                               "Maximum net shift per round (Å; blank disables; exceeding it rolls back the whole trial)"),
    "cycle.refinement_method": ("SHELXL 精修方法", "SHELXL 精密化法", "SHELXL refinement method"),
    "cycle.early_cgls_rounds": ("开头用 CGLS 的轮数（0 = 不用）", "最初に CGLS を使うラウンド数（0 = 使わない）", "Initial rounds refined with CGLS (0 = none)"),
    "cycle.damp_factor": ("DAMP 第一参数（留空 = 不设置；L.S. 与 CGLS 下含义不同）", "DAMP 第 1 パラメータ（空欄 = 設定しない。L.S. と CGLS で意味が異なる）",
                          "DAMP first parameter (blank = not set; its meaning differs between L.S. and CGLS)"),
    "cycle.damp_shift_esd_limit": ("DAMP 第二参数：位移/esd 上限（仅在设置了 DAMP 时使用；CGLS 忽略）", "DAMP 第 2 パラメータ：shift/esd 上限（DAMP 設定時のみ。CGLS では無視）",
                                   "DAMP second parameter: shift/esd limit (only with DAMP set; ignored by CGLS)"),
    "cycle.refinement_displacement_threshold_angstrom": ("位移统计阈值（Å；只用于统计，不拒绝也不限制坐标）", "移動量の集計閾値（Å。集計のみで、棄却も座標制限もしない）",
                                                         "Displacement threshold for counting (Å; diagnostic only, nothing is rejected or clamped)"),
    "cycle.rejection_budget_scope": ("失败预算的计数方式", "失敗予算の数え方", "How the failure budget is counted"),
    "cycle.backoff_failure_budget": ("开启退让时的失败预算（≥ 4，取与失败预算的较大值）", "後退が有効なときの失敗予算（4 以上、失敗予算との大きい方）",
                                     "Failure budget with backoff on (≥ 4; the larger of this and the failure budget)"),
    "cycle.rejected_action_ttl_rounds": ("被拒动作的暂缓轮数（0 = 只排除完全相同的提议）", "棄却された操作を保留するラウンド数（0 = 完全に同じ提案のみ除外）",
                                         "Rounds a rejected action set stays quarantined (0 = exact duplicates only)"),
    "cycle.anchor_uiso_policy": ("锚点不被支持时的处理", "アンカーが支持されない場合の扱い", "What to do with unsupported anchors"),
    "cycle.anchor_min_supported": ("至少保留的受支持锚点数（不足则阻断）", "支持されるアンカーの最小数（不足ならブロック）", "Minimum supported anchors (fewer blocks the run)"),
    "cycle.plan_peaks": ("每轮列出的峰数 PLAN（0–999）", "各回に列挙するピーク数 PLAN（0–999）", "Peaks listed per round PLAN (0–999)"),
    "cycle.rejection_backoff": ("拒绝后排除重复并逐级减少动作", "棄却後に重複を除き操作を段階的に減らす", "Exclude rejected duplicates and reduce actions"),
    "cycle.anchor_uiso_guard": ("每轮精修后检查固定锚点的 Uiso（默认关闭）", "各精密化後に固定アンカーの Uiso を確認（既定はオフ）",
                                "Check fixed-anchor Uiso after each refinement (off by default)"),
    "cycle.anchor_uiso_cap": ("锚点 Uiso 上限", "アンカー Uiso 上限", "Anchor Uiso upper cap"),
    "cycle.adaptive.prune_evidence_mode": ("删除证据规则", "削除の根拠規則", "Pruning evidence rule"),

    "cycle.iterations": ("总重构轮数（留空 = 自动数值停止）", "総再構成ラウンド数（空欄 = 自動の数値停止）",
                         "Total reconstruction rounds (blank = automatic numerical stop)"),
    "cycle.max_rounds": ("安全上限轮数", "安全上限ラウンド数", "Safety maximum rounds"),
    "cycle.shelxl_cycles": ("每轮 SHELXL 精修循环数（不是重构轮数）", "1 ラウンドの SHELXL 精密化サイクル数（再構成ラウンド数ではない）",
                            "SHELXL cycles per round (not reconstruction rounds)"),
    "cycle.stable_transitions": ("所需连续稳定次数", "必要な連続安定回数", "Successive stable transitions required"),
    "cycle.coordinate_tolerance_angstrom": ("坐标变化容差（Å）", "座標変化の許容値（Å）", "Coordinate-change tolerance (Å)"),
    "cycle.r1_tolerance": ("R1 变化容差（绝对值，分数，不是 %）", "R1 変化の許容値（絶対値・分率、% ではない）", "Absolute R1-change tolerance (fraction, not %)"),
    "cycle.important_atom_ids": ("重要位点（可选，逗号分隔；留空用后端规则）", "重要サイト（任意、カンマ区切り。空欄はバックエンドの規則）",
                                 "Important site IDs (optional, comma-separated; blank = backend rule)"),
    "cycle.max_wall_seconds": ("整条链时间上限（秒，含初解）", "全体の時間上限（秒、初期解を含む）", "Whole-chain time cap (s, including the initial route)"),
    "cycle.trial_timeout_s": ("单次精修超时（秒）", "1 回の精密化のタイムアウト（秒）", "Per-refinement timeout (s)"),
    "cycle.inference_protocol": ("推理输入协议", "推論入力プロトコル", "Inference input protocol"),
    "cycle.inference_seed": ("记录的协议种子（不是置信度参数）", "記録するプロトコルシード（信頼度パラメータではない）", "Recorded protocol seed (not a confidence parameter)"),
    "cycle.stopping_policy": ("数值停止规则", "数値停止規則", "Numerical stopping policy"),
    "cycle.require_no_element_changes_for_stop": ("有元素改动的轮次不算稳定", "元素変更があったラウンドは安定としない", "A round with element changes is never stable"),
    "cycle.record_diagnostics": ("记录每轮无答案诊断指标", "各ラウンドの答え不要な診断指標を記録", "Record per-round answer-free diagnostics"),
    "cycle.max_new_peaks_per_round": ("每轮最多接受的新峰数", "1 ラウンドで受け入れる新ピークの最大数", "Max new peaks accepted per round"),
    "cycle.minimum_peak_height": ("新峰最小峰高（e/Å³）", "新ピークの最小高さ（e/Å³）", "Minimum new-peak height (e/Å³)"),
    "cycle.minimum_peak_distance_angstrom": ("新峰与已有位点的最小距离（Å）", "新ピークと既存サイトの最小距離（Å）", "Minimum new-peak distance to sites (Å)"),
    "cycle.minimum_assignment_probability": ("最小指认概率（未校准）", "最小割り当て確率（未較正）", "Minimum assignment probability (uncalibrated)"),
    "cycle.max_r1_regression": ("允许的 R1 回退上限（分数）", "許容する R1 悪化の上限（分率）", "Max R1 regression allowed (fraction)"),
    "cycle.max_failures": ("失败预算（被拒轮次）", "失敗予算（棄却ラウンド数）", "Failure budget (rejected rounds)"),
    "cycle.adaptive.hysteresis_enabled": ("滞后：改元素需重复确认", "ヒステリシス：元素変更に繰り返しの確認", "Hysteresis: element changes need repeated proposals"),
    "cycle.adaptive.uiso_checks_enabled": ("Uiso 证据检查", "Uiso 証拠チェック", "Uiso evidence checks"),
    "cycle.adaptive.uiso_directional_proposals": ("Uiso 方向建议（更重/更轻的候选）", "Uiso 方向の提案（重い/軽い候補）", "Uiso directional proposals (heavier/lighter candidate)"),
    "cycle.adaptive.pruning_enabled": ("删除位点（可回滚）", "サイト削除（ロールバック可）", "Site pruning (reversible)"),
    "cycle.adaptive.growth_filter_enabled": ("新峰筛选（相对峰高 + 成键距离）", "新ピーク選別（相対高さ＋結合距離）", "Growth filter (relative height + bond distance)"),
    "cycle.adaptive.confirmations": ("改元素所需连续确认次数", "元素変更に必要な連続確認回数", "Confirmations before an element change"),
    "cycle.adaptive.max_change_fraction": ("每轮改元素上限（位点比例）", "1 ラウンドの元素変更上限（サイト比率）", "Max element-change fraction per round"),
    "cycle.adaptive.freeze_confirmations": ("冻结位点所需确认次数", "サイト固定に必要な確認回数", "Confirmations to freeze a site"),
    "cycle.adaptive.thaw_on_uiso_anomaly": ("Uiso 异常时解冻", "Uiso 異常で固定を解除", "Unfreeze on a Uiso anomaly"),
    "cycle.adaptive.u_abs_low": ("Uiso 绝对下限（Å²）", "Uiso 絶対下限（Å²）", "Uiso absolute low (Å²)"),
    "cycle.adaptive.u_abs_high": ("Uiso 绝对上限（Å²）", "Uiso 絶対上限（Å²）", "Uiso absolute high (Å²)"),
    "cycle.adaptive.u_relative_low": ("Uiso 相对下限（相对同元素中位数）", "Uiso 相対下限（同元素の中央値比）", "Uiso low ratio (to same-element median)"),
    "cycle.adaptive.u_relative_high": ("Uiso 相对上限（相对同元素中位数）", "Uiso 相対上限（同元素の中央値比）", "Uiso high ratio (to same-element median)"),
    "cycle.adaptive.reference_min_count": ("参考中位数所需同元素位点数", "参照中央値に必要な同元素サイト数", "Same-element peers needed for a reference"),
    "cycle.adaptive.reference_min_unchanged_rounds": ("参考位点所需未改动轮数", "参照サイトに必要な無変更ラウンド数", "Unchanged rounds needed for a reference peer"),
    "cycle.adaptive.prune_high_u": ("删除 Uiso 过大的位点", "Uiso が大きすぎるサイトを削除", "Prune high-Uiso sites"),
    "cycle.adaptive.prune_isolated": ("删除孤立位点", "孤立サイトを削除", "Prune isolated sites"),
    "cycle.adaptive.prune_close": ("处理过近位点", "近すぎるサイトを処理", "Resolve too-close sites"),
    "cycle.adaptive.prune_confirmations": ("删除所需重复证据次数", "削除に必要な繰り返し証拠の回数", "Repeated evidence needed before pruning"),
    "cycle.adaptive.max_prune_fraction": ("每轮删除上限（位点比例）", "1 ラウンドの削除上限（サイト比率）", "Max pruning fraction per round"),
    "cycle.adaptive.close_distance_angstrom": ("过近距离阈值（Å）", "近すぎる距離の閾値（Å）", "Too-close distance (Å)"),
    "cycle.adaptive.close_survivor_policy": ("过近位点保留规则", "近すぎるサイトの残し方", "Which too-close site survives"),
    "cycle.adaptive.minimum_bond_distance_angstrom": ("成键计数的最小距离（Å）", "結合数え上げの最小距離（Å）", "Minimum distance counted as a bond (Å)"),
    "cycle.adaptive.covalent_radius_scale": ("共价半径和的倍数", "共有結合半径の和に掛ける係数", "Multiplier on summed covalent radii"),
    "cycle.adaptive.maximum_bond_distance_angstrom": ("非金属接触最大距离（Å）", "非金属接触の最大距離（Å）", "Maximum non-metal contact (Å)"),
    "cycle.adaptive.maximum_metal_bond_distance_angstrom": ("金属接触最大距离（Å）", "金属接触の最大距離（Å）", "Maximum metal contact (Å)"),
    "cycle.adaptive.isolation_exempt_oxygen": ("孤立的 O 不删（可能是水；不等于指认为水）", "孤立した O は削除しない（水の可能性。水と判定するわけではない）",
                                               "Keep isolated O (possible water; not a water assignment)"),
    "cycle.adaptive.growth_relative_peak_height": ("新峰相对最强残余峰的最小比例", "最強の残差ピークに対する新ピークの最小比", "New-peak height relative to the strongest residual peak"),
    "cycle.adaptive.early_growth_rounds": ("早期轮数（允许更多新峰）", "初期ラウンド数（新ピークを多めに許可）", "Early rounds (larger peak budget)"),
    "cycle.adaptive.early_peak_limit": ("早期每轮新峰上限", "初期ラウンドの新ピーク上限", "Early new-peak limit per round"),
    "cycle.adaptive.growth_bond_distance_angstrom": ("新峰到非金属骨架的最大距离（Å）", "新ピークから非金属骨格への最大距離（Å）", "Max new-peak distance to a non-metal site (Å)"),
    "cycle.adaptive.growth_metal_distance_angstrom": ("新峰到金属的最大距离（Å）", "新ピークから金属への最大距離（Å）", "Max new-peak distance to a metal (Å)"),
    "cycle.adaptive.max_sites": ("几何计算位点上限（资源）", "幾何計算のサイト上限（リソース）", "Geometry site bound (resource)"),
    "cycle.adaptive.max_image_candidates": ("对称/周期像上限（资源）", "対称・周期像の上限（リソース）", "Symmetry/periodic image bound (resource)"),
    "cycle.adaptive.max_neighbor_evaluations": ("接触计算次数上限（资源）", "接触計算回数の上限（リソース）", "Contact evaluation bound (resource)"),
    "cycle.adaptive.max_neighbor_pairs": ("近邻对上限（资源）", "近傍ペアの上限（リソース）", "Neighbour-pair bound (resource)"),
}

_CHOICES: dict[str, tuple[str, str, str]] = {
    "route.strategy.shelxt": ("SHELXT 初解", "SHELXT 初期解", "SHELXT"),
    "route.strategy.protein_cde": ("蛋白质实验定相 SHELXC/D/E", "タンパク質の実験的位相決定 SHELXC/D/E", "Protein experimental phasing: SHELXC/D/E"),
    "route.strategy.small_molecule": ("小分子 SHELXS / SHELXD", "小分子 SHELXS / SHELXD", "Small molecule: SHELXS / SHELXD"),
    "route.small_method.direct": ("直接法", "直接法", "Direct methods"),
    "route.small_method.patterson": ("Patterson 法", "Patterson 法", "Patterson"),
    "cycle.inference_protocol.current_labels": ("当前标签（旧行为，可能自反馈）", "現在のラベル（従来の動作、自己フィードバックの恐れ）",
                                                "Current labels (old behaviour; may feed back on itself)"),
    "cycle.inference_protocol.initial_labels": ("冻结的初解标签", "凍結した初期解のラベル", "Frozen initial-model labels"),
    "cycle.inference_protocol.upstream_ins": ("上游协议：原始 INS 的 SFAC/UNIT 初始化 + 两遍预测 + 覆盖约束",
                                              "上流プロトコル：元 INS の SFAC/UNIT で初期化＋2 パス予測＋網羅制約",
                                              "Upstream: original INS SFAC/UNIT initialisation, two passes, coverage"),
    "cycle.stopping_policy.movable_and_growth": ("可动位点 + 连续两轮无增删（默认）", "可動サイト＋2 ラウンド連続で増減なし（既定）",
                                                 "Movable sites + two rounds without additions/deletions (default)"),
    "cycle.stopping_policy.legacy_important": ("旧规则：重要位点坐标", "旧規則：重要サイトの座標", "Legacy: important-site coordinates"),
    "cycle.adaptive.close_survivor_policy.lower_u": ("保留 Uiso 较低者", "Uiso の低い方を残す", "Keep the lower-Uiso site"),
    "cycle.adaptive.close_survivor_policy.reference_consistency": ("保留与参考更一致者", "参照とより整合する方を残す", "Keep the reference-consistent site"),
    "candidate_element_source.original_ins": ("原始 INS 的 SFAC（默认）", "元 INS の SFAC（既定）", "Original INS SFAC (default)"),
    "candidate_element_source.initial_seed": ("初解模型的 SFAC（旧行为）", "初期解モデルの SFAC（従来の動作）", "Initial-model SFAC (old behaviour)"),
    "protein.preferred_hand.auto": ("自动（按 CC；解析器未完全验证）", "自動（CC による。パーサは完全には未検証）", "Automatic (by CC; parser not fully verified)"),
    "protein.preferred_hand.original": ("原手性", "元のハンド", "Original"),
    "protein.preferred_hand.inverted": ("反转", "反転", "Inverted"),
    "cycle.refinement_method.ls": ("全矩阵最小二乘 L.S.（默认）", "全行列最小二乗 L.S.（既定）", "Full-matrix least squares L.S. (default)"),
    "cycle.refinement_method.cgls": ("共轭梯度 CGLS（适合大结构早期；不给标准不确定度）", "共役勾配 CGLS（大きな構造の初期向け。標準不確かさなし）",
                                     "Conjugate gradient CGLS (early stages of large structures; no standard uncertainties)"),
    "cycle.rejection_budget_scope.base_model_streak": ("同一基础模型上连续被拒的不同提议（默认）", "同じ基礎モデルで連続して棄却された異なる提案（既定）",
                                                       "Consecutive distinct rejections on the same base model (default)"),
    "cycle.rejection_budget_scope.run_total": ("整个运行累计（旧行为）", "実行全体の累計（従来の動作）", "Total over the run (old behaviour)"),
    "cycle.anchor_uiso_policy.unlock_or_block": ("解除不受支持锚点的固定；剩余不足则阻断（默认）", "支持されないアンカーの固定を解除。残りが不足ならブロック（既定）",
                                                 "Unlock unsupported anchors; block if too few remain (default)"),
    "cycle.anchor_uiso_policy.block_any": ("任何一个不受支持就阻断", "1 つでも支持されなければブロック", "Block if any anchor is unsupported"),
    "cycle.anchor_uiso_policy.report_only": ("只记录，不处理", "記録のみ", "Report only"),
    "cycle.adaptive.prune_evidence_mode.any": ("任一证据即可（U 过大、孤立或过近）", "いずれかの根拠で可（U 過大・孤立・近すぎ）",
                                               "Any evidence (high U, isolated or too close)"),
    "cycle.adaptive.prune_evidence_mode.high_u_and_isolated": ("仅当 U 过大且孤立", "U 過大かつ孤立の場合のみ", "Only when both high-U and isolated"),
    "mode.real": ("本机真实程序", "ローカルの実プログラム", "Real local programs"),
    "mode.mock_demo": ("合成演示（MOCK，仅内置数据）", "合成デモ（MOCK、内蔵データのみ）", "Synthetic demo (MOCK, built-in data only)"),
    "device.cpu": ("CPU", "CPU", "CPU"),
    "device.auto": ("自动（可用时使用 Apple GPU）", "自動（利用可能なら Apple GPU）", "Auto (Apple GPU when available)"),
    "heavy_preparation.current_loop": ("现有循环", "既存の反復", "Current loop"),
    "heavy_preparation.upstream_heavy_v1": ("初解重原子阶段（待参考对照）", "初期重原子段階（参照比較待ち）", "Initial heavy stage (reference parity pending)"),
    "device.mps": ("Apple GPU（MPS）", "Apple GPU（MPS）", "Apple GPU (MPS)"),
}

# Stop codes and blockers. Exact codes first, then families by pattern; unknown codes stay verbatim.
_STOPS: dict[str, tuple[str, str, str]] = {
    "fixed_rounds_completed": ("完成了指定的固定轮数（不是收敛判断）", "指定した固定ラウンド数を完了（収束の判断ではない）",
                               "the requested fixed number of rounds ran (not a convergence judgement)"),
    "important_coordinates_and_r1_stable": ("旧规则：重要位点坐标和 R1 连续满足容差。数值稳定的候选，不是科学收敛",
                                            "旧規則：重要サイトの座標と R1 が連続して許容値内。数値的に安定した候補で、科学的な収束ではない",
                                            "legacy rule: important-site coordinates and R1 stayed within tolerance: a numerically stable candidate, not scientific convergence"),
    "movable_sites_r1_and_growth_stable": ("可动位点坐标和 R1 连续满足容差，且连续两轮没有增删位点。数值稳定的候选，不是科学收敛",
                                           "可動サイトの座標と R1 が連続して許容値内で、2 ラウンド続けてサイトの増減なし。数値的に安定した候補で、科学的な収束ではない",
                                           "movable-site coordinates and R1 stayed within tolerance with two rounds without additions or deletions: a numerically stable candidate, not scientific convergence"),
    "no_acceptable_step_in_current_action_set": ("不同的提议都被拒绝，逐级减少动作直到纯精修也没有可接受的一步；保留的是最后一个未经验证的候选",
                                                 "異なる提案がすべて棄却され、操作を段階的に減らして純粋な精密化まで試しても受け入れられる一歩がなかった。保持しているのは最後の未検証候補",
                                                 "distinct proposals were rejected and the reduced action sets, down to pure refinement, gave no acceptable step; the last retained candidate is kept, unvalidated"),
    "anchor_displacement_unsupported": ("锚点 Uiso 检查没有通过：受支持的固定锚点不足，或设置为任何一个不受支持就阻断。这组锚点需要人工复核",
                                        "アンカーの Uiso 確認が通らない：支持される固定アンカーが不足、または 1 つでも支持されなければブロックする設定。確認が必要",
                                        "the anchor Uiso check blocked the trial: too few supported fixed anchors remain, or block-any was requested; the anchors need review"),
    "workflow_budget_exhausted": ("控制器的资源预算用完（具体哪一项见详情）", "コントローラのリソース予算を使い切った（どの予算かは詳細を参照）",
                                  "a controller resource budget was used up (the detail names it)"),
    "preview_blocked": ("预检未通过：缺少输入、程序或路线前提（见阻断原因）", "事前確認が通らない：入力・プログラム・ルートの前提が不足（阻害要因を参照）",
                        "the preview prerequisites are not met: inputs, programs or route requirements are missing (see blockers)"),
    "initial_solution_blocked": ("初解路线没能给出可用的完整初始模型（见阻断原因和详情）", "初期解ルートが使える完全な初期モデルを出せなかった（阻害要因と詳細を参照）",
                                 "the initial route could not deliver a usable complete seed (see blockers and detail)"),
    "initial_solution_failed": ("初解程序运行失败（见阻断原因和详情）", "初期解プログラムの実行に失敗（阻害要因と詳細を参照）",
                                "the initial-solution execution failed (see blockers and detail)"),
    "reconstruction_blocked": ("输入、来源记录或一致性检查没有通过，重构被阻断（具体检查项见详情）", "入力・来歴・整合性の確認が通らず、再構成がブロックされた（どの確認かは詳細を参照）",
                               "an input, provenance or consistency check blocked the reconstruction (the detail names the check)"),
    "workflow_stopped": ("控制器在没有数值停止或固定轮数完成的情况下停止（原因见详情）", "数値停止や固定ラウンド完了なしにコントローラが停止（理由は詳細を参照）",
                         "the controller stopped without a numerical stop or fixed-round completion (the detail gives its reason)"),
    "refinement_failed": ("SHELXL 精修进程失败，或其列表文件报告了致命错误（见详情）", "SHELXL 精密化プロセスが失敗、またはリスティングに致命的エラー（詳細を参照）",
                          "the SHELXL refinement process failed or its listing reported a fatal error (see detail)"),
    "owner_process_interrupted": ("负责这次运行的进程已经不在了；不会自动重放", "この実行を担当していたプロセスが存在しない。自動の再実行はなし",
                                  "the process that owned this run is gone; nothing is replayed automatically"),
    "round_safety_budget_exhausted": ("达到安全上限轮数仍未满足停止条件", "安全上限ラウンド数に達しても停止条件を満たさなかった",
                                      "the safety round limit was reached without meeting the stop condition"),
    "failure_budget_exhausted": ("被拒绝的轮次用完了失败预算（例如 R1 回退过大）", "棄却ラウンドで失敗予算を使い切った（例：R1 の悪化が大きすぎる）",
                                 "rejected rounds used up the failure budget (e.g. R1 regressed too far)"),
    "whole_chain_wall_budget_exhausted": ("整条链的时间上限已用完", "全体の時間上限を使い切った", "the whole-chain time cap was used up"),
    "cancelled": ("按请求取消", "要求により取り消し", "cancelled on request"),
    "backend_error": ("后端出现未预期的错误；详情保留在运行目录", "バックエンドで予期しないエラー。詳細は実行ディレクトリに保存",
                      "unexpected backend error; details are kept in the run directory"),
    "pretrained_checkpoint_missing": ("没有找到预训练权重文件", "事前学習済みの重みファイルが見つからない", "the pretrained weights file was not found"),
    "at_least_two_declared_nonhydrogen_candidate_elements_required": ("至少需要两个声明的非氢候选元素", "宣言された非水素の候補元素が 2 つ以上必要",
                                                                      "at least two declared non-hydrogen candidate elements are required"),
    "no_requested_metal_sites_in_automatic_initial_model": ("初解里没有所请求的金属位点", "初期解に要求した金属サイトが無い",
                                                            "the initial model has none of the requested metal sites"),
    "inputs_changed_after_preview": ("预检之后输入文件变了", "事前確認の後で入力ファイルが変わった", "input files changed after the preview"),
    "original_input_changed_during_run": ("运行期间原始输入文件变了", "実行中に元の入力ファイルが変わった", "an original input file changed during the run"),
    "full_matrix_parameter_resource_bound_exceeded": ("全矩阵参数超过资源上限（不会自动放宽）", "全行列パラメータがリソース上限を超えた（自動では緩めない）",
                                                      "full-matrix parameters exceed the resource bound (never relaxed automatically)"),
    "synthetic_providers_cannot_be_used_on_measured_inputs": ("模拟程序不能用于测量数据", "模擬プログラムは測定データに使えない",
                                                              "synthetic stand-ins cannot be used on measured data"),
    "toolchain_mode_differs_from_request": ("程序链模式与请求不符", "ツールチェーンのモードが要求と異なる", "the toolchain mode differs from the request"),
    "heavy_stage_completed": ("初解重原子阶段按固定步骤跑完（一次指认 + 一次各向异性精修）；这不代表收敛或结构正确",
                              "初期重原子段階が固定手順で完了（割り当て 1 回＋異方性精密化 1 回）。収束や構造の正しさを意味しない",
                              "the initial heavy stage ran its fixed steps (one assignment and one anisotropic refinement); this is neither convergence nor a correct structure"),
    "non_difference_map_native_cif_requires_recorded_fmap2_refinement": (
        "保留的模型来自 Fo 或 2Fo−Fc 图的生长轮，SHELXL 在这种轮次不能同时写 CIF；需要一轮有记录的差值图（FMAP 2）精修才能出 CIF。RES 和 FCF 已封存",
        "保持したモデルは Fo または 2Fo−Fc 図の成長ラウンドのもので、SHELXL はそのラウンドで CIF を書けない。CIF には記録された差フーリエ（FMAP 2）精密化が 1 回必要。RES と FCF は封印済み",
        "the retained model comes from a growth round on an Fo or 2Fo−Fc map, where SHELXL cannot also write a CIF; a CIF needs one recorded difference-map (FMAP 2) refinement. RES and FCF are sealed"),
    "mps_requires_available_apple_silicon_runtime": ("要求使用 Apple GPU（MPS），但本机没有可用的 Apple 芯片 GPU 运行环境",
                                                     "Apple GPU（MPS）を指定したが、この機械に利用可能な Apple シリコン GPU 環境がない",
                                                     "Apple GPU (MPS) was requested but no usable Apple silicon GPU runtime is available"),
    "requested_mps_unavailable_without_fallback": ("要求的 Apple GPU（MPS）启动检查未通过；按规定不会改用 CPU",
                                                   "指定した Apple GPU（MPS）の起動チェックに失敗。規定により CPU には切り替えない",
                                                   "the requested Apple GPU (MPS) failed its startup check; by policy the run does not switch to CPU"),
    "synthetic_provider_does_not_load_weights_or_use_accelerators": ("合成演示不加载权重，也不使用 GPU",
                                                                     "合成デモは重みを読み込まず GPU も使わない",
                                                                     "the synthetic demo loads no weights and uses no GPU"),
    "cgls_native_cif_requires_recorded_nonzero_ls_refinement": (
        "保留的模型来自 CGLS 精修，SHELXL 不能在 CGLS 下写 CIF；需要一轮有记录、按同样规则判定的 L.S. 精修才能出 CIF（目前不自动追加）。RES 和 FCF 已封存",
        "保持したモデルは CGLS 精密化のもので、SHELXL は CGLS では CIF を書けない。CIF には記録され同じ規則で判定される L.S. 精密化が 1 回必要（現在は自動追加しない）。RES と FCF は封印済み",
        "the retained model comes from CGLS refinement, and SHELXL cannot write a CIF under CGLS; a CIF needs one recorded L.S. refinement judged by the same rules (not added automatically at present). RES and FCF are sealed"),
    "net_refinement_displacement": ("精修后有位点的净位移超过设定上限", "精密化後の正味移動量が設定した上限を超えたサイトがある",
                                    "a site moved further in refinement than the configured net-shift limit"),
    "shelxl_became_unavailable": ("SHELXL 在运行中变得不可用", "実行中に SHELXL が使えなくなった", "SHELXL became unavailable during the run"),
    "adaptive_geometry_provider_unavailable": ("自适应策略需要的几何计算组件（gemmi/numpy/scipy）不可用",
                                               "適応ポリシーに必要な幾何計算部品（gemmi/numpy/scipy）が使えない",
                                               "the geometry components the adaptive policy needs (gemmi/numpy/scipy) are unavailable"),
    "actual_inference_protocol_differs_from_frozen_policy": ("实际推理协议与冻结的策略不一致", "実際の推論プロトコルが凍結した方針と異なる",
                                                             "the inference protocol actually used differs from the frozen policy"),
    "synthetic_predictions_for_measured_data_refused": ("拒绝把模拟预测用于测量数据", "測定データへの模擬予測を拒否", "synthetic predictions for measured data refused"),
    "important_atom_selection_is_not_in_initial_model": ("指定的重要位点不在初解模型里", "指定した重要サイトが初期解モデルに無い",
                                                         "a requested important site is not in the initial model"),
    "anomalous_friedel_pairs_not_declared": ("没有声明保留 Friedel 对", "Friedel 対の保持が宣言されていない", "Friedel pairs were not declared as retained"),
    "mtz_to_supported_reflections_conversion_required": ("MTZ 需要先转换为支持的格式", "MTZ は対応形式への変換が必要", "MTZ must first be converted to a supported format"),
    "shelxd_small_requires_explicit_find_and_plop_counts": ("小分子 SHELXD 需要显式的 FIND 和 PLOP 计数", "小分子 SHELXD には明示的な FIND と PLOP が必要",
                                                            "small-molecule SHELXD needs explicit FIND and PLOP counts"),
    "protein_context_not_declared": ("没有确认这是蛋白质数据", "タンパク質データであることが確認されていない", "protein data was not confirmed"),
    "unit_cell_missing": ("INS 里没有晶胞参数", "INS にセル定数が無い", "the INS has no unit cell"),
}

_FAMILIES: tuple[tuple[str, str, tuple[str, str, str]], ...] = (
    ("prefix", "tool_unavailable:", ("所需程序不可用（详见预检里的程序表）", "必要なプログラムが使えない（事前確認の表を参照）",
                                     "a required program is unavailable (see the program table in the preview)")),
    ("suffix", "_budget_exhausted", ("预算已用完", "予算を使い切った", "a budget was used up")),
    ("suffix", "_budget_exceeded", ("超过资源上限（不会自动放宽）", "リソース上限を超えた（自動では緩めない）", "a resource bound was exceeded (never relaxed automatically)")),
    ("prefix", "protein_", ("蛋白质路线的前提或转换限制", "タンパク質ルートの前提条件または変換の制限", "a protein-route prerequisite or conversion limit")),
    ("prefix", "synthetic_", ("模拟程序只接受内置合成数据", "模擬プログラムは内蔵の合成データのみ", "synthetic stand-ins accept only built-in synthetic data")),
    ("prefix", "adaptive_", ("自适应策略的检查未通过", "適応ポリシーのチェックが通らなかった", "an adaptive-policy check failed")),
    ("prefix", "runtime_", ("预训练推理的前提或身份检查未通过", "事前学習推論の前提または同一性チェックが通らなかった", "a pretrained-inference prerequisite or identity check failed")),
    ("prefix", "inference_", ("预训练推理的身份检查未通过", "事前学習推論の同一性チェックが通らなかった", "a pretrained-inference identity check failed")),
    ("prefix", "upstream_", ("上游协议的输入前提不满足", "上流プロトコルの入力前提を満たさない", "an upstream-protocol input prerequisite is not met")),
    ("prefix", "refinement_", ("SHELXL 精修或其文件检查未通过", "SHELXL 精密化またはファイルの確認が通らなかった", "SHELXL refinement or its file checks failed")),
    ("prefix", "shelxl_", ("SHELXL 精修失败", "SHELXL 精密化に失敗", "SHELXL refinement failed")),
    ("prefix", "generated_initial_", ("复用的初解证据不一致", "再利用する初期解の証拠が不一致", "the reused initial-result evidence does not match")),
    ("prefix", "checkpoint_", ("检查点记录不一致", "チェックポイントの記録が不一致", "the checkpoint record does not match")),
    ("prefix", "initial_route_", ("初解路线未完成", "初期解ルートが完了しなかった", "the initial route did not complete")),
    ("prefix", "cancelled", ("已取消", "取り消し", "cancelled")),
    ("contains", "changed", ("文件或身份在运行中改变，已阻断", "実行中にファイルまたは同一性が変わったためブロック", "a file or identity changed during the run, so it was blocked")),
    ("contains", "missing", ("缺少前提条件", "前提条件が不足", "a prerequisite is missing")),
    ("contains", "required", ("缺少必须的声明或输入", "必須の宣言または入力が不足", "a required declaration or input is missing")),
    ("contains", "unsupported", ("当前适配器不支持", "現在のアダプタでは未対応", "not supported by the current adapter")),
    # CheckCIF reasons
    ("prefix", "no_recognized_alerts", ("没有识别到警告不等于通过", "警告が認識されなかったことは合格ではない", "no recognised alerts is not a pass")),
    ("prefix", "alerts_require", ("警告需要专家复核", "警告は専門家の確認が必要", "alerts require expert review")),
    ("prefix", "local_chk_parser", ("本机报告解析格式尚未验证", "ローカルのレポート解析形式は未検証", "the local report parser format is unverified")),
    ("prefix", "explicit_checkdef", ("需要提供 check.def 或明确允许内置 check.def", "check.def の指定か組み込み check.def の明示的な許可が必要",
                                     "provide check.def or explicitly allow the built-in one")),
    ("prefix", "simulated_toolchain", ("模拟程序不能做本机 PLATON 诊断", "模擬プログラムではローカル PLATON 診断はできない", "a simulated toolchain cannot run local PLATON diagnostics")),
)


def rt(key: str, lang: str = DEFAULT_LANG, **values: object) -> str:
    entry = _T.get(key)
    if entry is None:
        return key
    text = entry[_i(lang)]
    return text.format(**values) if values else text


def field_label(path: str, lang: str = DEFAULT_LANG) -> str | None:
    entry = _FIELDS.get(path)
    return None if entry is None else entry[_i(lang)]


def choice_label(path: str, value: str, lang: str = DEFAULT_LANG) -> str:
    entry = _CHOICES.get(f"{path}.{value}")
    return value if entry is None else entry[_i(lang)]


def stop_explanation(code: str, lang: str = DEFAULT_LANG) -> str:
    """A short meaning for a backend code, or an empty string when none is known (shown verbatim)."""
    if not code:
        return ""
    base = code.split(":", 1)[0]
    if base in _STOPS:
        return _STOPS[base][_i(lang)]
    if code in STOP_CODES:
        return STOP_CODES[code].description
    for how, pattern, texts in _FAMILIES:
        if ((how == "prefix" and code.startswith(pattern)) or (how == "suffix" and base.endswith(pattern))
                or (how == "contains" and pattern in base)):
            return texts[_i(lang)]
    return ""


# Per-round decisions of the reconstruction loop (``history[].decision``). Shown as "CODE — meaning";
# a code without an entry is shown verbatim.
DECISIONS: dict[str, tuple[str, str, str]] = {
    "RETAIN_UNVALIDATED_PENDING_CONTROLLER": ("本轮通过数值检查，等待流程确认保留（未验证）", "数値チェックを通過、保持の確定待ち（未検証）",
                                              "passed the numerical checks; waiting to be retained (not validated)"),
    "RETAIN_UNVALIDATED": ("保留为当前模型（未验证）", "現在のモデルとして保持（未検証）", "retained as the current model (not validated)"),
    "REJECT_R_REGRESSION": ("R1 变差超过允许值，整轮回滚", "R1 の悪化が許容値を超え、ラウンド全体を戻した",
                            "R1 got worse by more than the allowed amount; the whole round was rolled back"),
    "REJECT_REFINEMENT_DISPLACEMENT": ("有位点净位移超过设定上限，整轮回滚", "正味移動量が上限を超えたサイトがあり、ラウンド全体を戻した",
                                       "a site moved further than the configured net-shift limit; the whole round was rolled back"),
    "BLOCK_UNSUPPORTED_ANCHORS": ("锚定位点的证据不足，本轮不保留", "アンカーサイトの裏付けが不足し、このラウンドは保持しない",
                                  "the anchor sites were not sufficiently supported; the round was not retained"),
    "REJECTED_BY_CONTROLLER": ("流程控制器没有保留本轮", "ワークフロー制御がこのラウンドを保持しなかった", "the workflow controller did not retain this round"),
    "ACCEPT_PENDING_CONTROLLER": ("本轮通过检查，等待流程确认（未验证）", "チェックを通過、確定待ち（未検証）",
                                  "passed the checks; waiting for the controller (not validated)"),
    "ACCEPT_PROVISIONAL": ("暂时接受（未验证）", "暫定的に採用（未検証）", "provisionally accepted (not validated)"),
}


def decision_label(code: str, lang: str = DEFAULT_LANG) -> str:
    """``CODE — meaning`` for a known per-round decision; an unknown code is returned unchanged."""
    entry = DECISIONS.get(code)
    return code if entry is None else f"{code} — {entry[_i(lang)]}"


def stop_codes_without_text() -> list[str]:
    """Stop codes of the backend registry that this page cannot yet explain in all three languages."""
    return sorted(code for code in STOP_CODES if code not in _STOPS)


def history_columns(lang: str = DEFAULT_LANG) -> list[str]:
    return [c[_i(lang)] for c in _HISTORY_COLUMNS]


def run_columns(lang: str = DEFAULT_LANG) -> list[str]:
    return [c[_i(lang)] for c in _RUN_COLUMNS]
