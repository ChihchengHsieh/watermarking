# Agentic MetaSpiderMark 論文修改討論

這份文件用來逐項討論導師提出的修改意見。現階段先確認論文的主張、內容取捨與實驗需求，不直接修改正文。

## 目前已確認的修改範圍

### 確定要做

1. **改進 overview figure** — 已完成
   - 明確標示 `INNER LOOP` 與 `OUTER LOOP`。
   - 顯示 support adaptation：\(\theta \rightarrow \theta'_k\)。
   - 顯示 query loss 如何進行 outer/meta update。
   - 將 scheduler feedback 畫成與 gradient update 不同的控制迴路。
   - 將 task-weight 箭頭改為控制 attack episode sampling，而不是 SpiderMark injection。
   - 標示 SpiderMark injection 與 diffusion generator 為 fixed/unchanged。
   - 優先以現有 `figures/meta_learning_method_figure.tex` 向量圖為邏輯基礎。

2. **加入 previous-paper COCO comparison table** — 已完成
   - 放在 Experiments 的 backbone-selection／motivation 位置。
   - 用來解釋為何選擇 SpiderMark 作為 meta-learning backbone。
   - 清楚交代 unseen COCO、共同 evaluation protocol、數據來源與 FID 方向。
   - 不因加入表格而自行刪除其他內容；頁數取捨交由導師決定。

### 必要的技術收尾

1. [x] 修正 Method 中未定義的 `fig:Architecture` reference。
2. [x] 在 `main.bib` 補上原本缺失的 `hospedales2021meta`。
3. [x] 以簡短 SpiderMark preliminaries 取代紅色 TODO，未改寫導師的 meta-learning 段落。
4. [x] 修改完成後重新編譯並檢查 undefined references、citations 與 figure layout。

### 已確認暫不做

- 不修改導師已完成的 Abstract。
- 不重寫導師已補充的 meta-learning 文字與公式。
- 不因八頁限制自行刪除或移動內容。
- 不替換論文標題或模型名稱，等待導師正式命名。
- 不建立或重寫 Miguel 的實驗 README。

### 尚待確認，不屬於目前確定修改

- 是否需要新增或補跑 matched scheduler baselines。
- 是否需要多 seeds、held-out attacks 或額外 COCO evaluation。
- 最終 checkpoint-selection protocol。
- 八頁版本需要刪除或移至 supplementary 的內容。
- 模型正式名稱與論文正式標題。

## 標題與模型名稱（暫不修改）

導師將先為模型／方法命名。在正式名稱確認前，不進行全文標題或方法名稱替換。

目前文件中出現的標題與 `Agentic MetaSpiderMark`、`MetaSpiderMark` 等名稱先維持原狀。

### 等待導師確認

- [ ] 模型／方法的正式名稱。
- [ ] 論文的正式標題。
- [ ] 標題中是否保留 `SpiderMark`。
- [ ] `MetaSpiderMark` 與 agentic scheduler variant 的命名關係。

---

## 1. Abstract（導師已修改，不再更動）

### 決定

目前的 Abstract 已由導師修改並確認。本輪論文修改不再改寫、縮短或調整 Abstract。

### 執行原則

- [x] 保留導師修改後的 Abstract。
- [ ] 修改其他章節時，不連帶重寫 Abstract。
- [ ] 最後只檢查 LaTeX 編譯、引用、拼字與格式；除非導師再次要求，否則不改文字內容。

---

## 2. Meta-learning（導師已補充，不再重寫）

### 核對結果

目前的 `sec/3_method.tex` 已經包含導師要求的主要內容：

- `Preliminaries` 定義 task distribution、meta-knowledge、support set 與 query set；
- `Problem Setup` 將每一種 downstream attack 定義為 task；
- `Attack-Conditioned Meta-Training` 明確定義 support loss 與 query loss；
- inner loop 明確寫出 support-set adaptation；
- outer loop 明確寫出 adapted verifier 上的 query objective；
- 正文說明目前使用 first-order approximation，省略 second-order Hessian terms；
- `Agentic Task Scheduler` 將 scheduler update 與 verifier 的 inner/outer update 分開；
- Algorithm 1 依序列出 task sampling、inner adaptation、query loss、outer update 與 scheduler update。

因此，導師提出的「inner loop 與 outer loop 沒有明確說明」在目前版本中已經處理。本輪不再重寫這一部分，避免覆蓋導師修改的內容。

### 目前已有的 inner loop

\[
\theta'_k
=
\theta-\alpha\nabla_{\theta}
\mathcal{L}_{\mathcal{S}_k}(\theta).
\]

### 目前已有的 outer objective

\[
\min_{\theta}
\mathbb{E}_{\tau_k \sim \pi_t}
\left[
\mathcal{L}_{\mathcal{Q}_k}(\theta'_k)
\right].
\]

### 只需處理的收尾問題

- [x] 保留導師加入的 meta-learning 說明，不再進行內容改寫。
- [x] 將未定義的 `Fig. \ref{fig:Architecture}` 修正為 `fig:method_overview`。
- [x] 在 `main.bib` 加入 `hospedales2021meta`。
- [x] 補上簡短 SpiderMark preliminaries 並刪除紅色 TODO。
- [x] 重新編譯確認上述 reference、citation 與段落可正常輸出。

### Overview figure 仍需要改進

雖然正文已清楚定義 inner loop 與 outer loop，目前的 generated overview figure：

```text
papers/meta_learning/figures/metaspidermark_overview_imagegen.png
```

仍有以下問題：

1. 圖中只標示 `inner adaptation`，沒有明確標示 `outer loop`。
2. Query set 沒有清楚連到 query loss 與 outer/meta update。
3. 沒有呈現共享參數 \(\theta\) 經 inner loop 變成 task-adapted parameters \(\theta'_k\)。
4. 沒有呈現 outer loop 使用 \(\mathcal{L}_{\mathcal{Q}_k}(\theta'_k)\) 更新共享參數 \(\theta\)。
5. Scheduler feedback loop 與 meta-learning outer loop 在視覺上容易混為同一個更新。
6. `Task weights` 目前指向 `SpiderMark Injection`，可能讓讀者誤以為 scheduler 會改變 watermark injection；實際上 task weights 應控制 attack-conditioned episode sampling。
7. 圖中沒有強調 SpiderMark injection 與 diffusion generator 在本文中保持固定。

### 建議的新圖邏輯

```text
Fixed SpiderMark injection
          ↓
clean/watermarked images
          ↓
Scheduler samples attack task τ_k from π_t
          ↓
attack-conditioned support S_k and query Q_k
          ↓
┌──────────────── Bilevel meta-training ────────────────┐
│ INNER LOOP                                            │
│ Support S_k → adapt θ → θ'_k                          │
│                                                       │
│ OUTER LOOP                                            │
│ Query Q_k → L_Qk(θ'_k) → update shared initialization θ│
└───────────────────────────────────────────────────────┘
          ↓ query loss / failure / uncertainty
Scheduler updates future task distribution π_{t+1}
```

### 建議的視覺標示

- 使用明確的 `INNER LOOP` 與 `OUTER LOOP` 標籤。
- Inner loop 使用實線箭頭，標示 \(\theta \rightarrow \theta'_k\)。
- Outer loop 使用另一種顏色的回傳箭頭，標示 query gradient updates \(\theta\)。
- Scheduler control loop 使用虛線，避免和 gradient update 混淆。
- 將 `query loss`、`failure rate`、`uncertainty` 標在 query evaluation 之後。
- 在 SpiderMark injection 區塊加上 `fixed / unchanged`。
- Task-weight 箭頭應指向 attack-task sampler 或 attack-conditioned episodes。
- Caption 應明確說明 scheduler 不修改 watermark injection 或 verifier architecture。

### 可重用的既有向量圖

Repository 中已有一個更接近正確流程的圖：

```text
figures/meta_learning_method_figure.tex
figures/meta_learning_method_figure.pdf
figures/meta_learning_method_figure.png
```

該圖已包含：

- attack-conditioned episodes；
- support/query 分割；
- support adaptation；
- query evaluation 與參數更新；
- task scheduler；
- loss feedback；
- downstream verification。

建議以該 TikZ/向量圖作為邏輯基礎，再吸收 generated overview 的配色與 SpiderMark injection 視覺元素。相較於直接使用 AI-generated raster image，向量圖更容易精確標示公式、修改箭頭，並能在雙欄論文中保持清晰。

### Figure 修改決策

- [x] 使用 imagegen 重新生成 raster overview，並以既有 TikZ 圖作為邏輯參考。
- [x] 明確加入 `INNER LOOP`。
- [x] 明確加入 `OUTER LOOP`。
- [x] 顯示 \(\theta \rightarrow \theta'_k\) 與 query-gradient update。
- [x] 將 task-weight 箭頭改為指向 attack episode sampling。
- [x] 將 scheduler feedback 畫成獨立的虛線控制迴路。
- [x] 標記 SpiderMark injection 為 fixed/unchanged。
- [x] 新圖不包含 downstream fine-tuning。

---

## 3. 八頁限制（暫不刪除，由導師決定）

### 決定

目前不主動刪除或移動任何正文、圖表、roadmap 或 diagnostics。八頁限制如何處理，以及哪些內容應保留、刪除或移至 supplementary，交由導師決定。

### 目前狀況（僅供導師參考）

- 最新檢查的 PDF 共 10 頁。
- 正文目前延伸至第 9 頁，references 位於後段。
- 加入 COCO backbone-selection table 後，版面可能進一步增加。
- 以上只記錄現況，不代表本輪要自行刪除內容。

### 本輪執行原則

- [x] 暫時保留目前所有內容。
- [x] 已加入導師要求的 COCO 表格，未為控制頁數自行刪除其他內容。
- [x] 已重新編譯；目前全文為 11 頁，等待向導師回報。
- [ ] 由導師指定需要刪除、壓縮或移至 supplementary 的內容。
- [ ] 導師確認取捨後，再執行八頁版面調整。

---

## 4. 加入 COCO 表格，說明為何選擇 SpiderMark

### 建議位置

不要放在 Related Work，因為這是實驗證據。建議放在 Experiments 開頭：

> **Backbone Selection: Why SpiderMark?**

Introduction 可以提前引用該表，說明選擇 SpiderMark 並非任意決定。

### 建議引導文字

> Before studying verifier meta-learning, we compare candidate watermarking backbones under distribution shift. Table X reports verification results on COCO, which is unseen during training. SpiderMark achieves the highest accuracy and AUROC under every evaluated transformation while retaining image quality comparable to alternative methods. We therefore select SpiderMark as the fixed watermarking backbone and focus on improving its verifier training rather than modifying watermark injection.

### 表格要補充的資訊

- 說明 COCO 在此 protocol 中是 unseen dataset。
- 說明四個方法使用相同的 images、attacks 與 evaluation protocol。
- 說明 Acc 使用的 threshold policy。
- 說明 AUROC 的計算方式。
- 說明 FID 越低越好。
- 說明 non-watermarked FID 只是 reference。
- 說明結果來自 previous SpiderMark paper，或是在本研究中使用相同 protocol 重新計算。
- 如果結果沿用前一篇論文，應引用該論文並避免讓讀者誤以為這是新的實驗。

### 表格所支持的正確結論

這張表可以支持：

> SpiderMark is the strongest verifier backbone among the evaluated watermarking methods under COCO distribution shift.

這張表不能單獨支持：

> SpiderMark is universally the best watermarking method.

原因是比較範圍只涵蓋四種方法與目前定義的 attack suite。

### 需要確認

- [ ] 這些數字是直接沿用 previous paper，還是以目前程式重新跑出的結果？
- [ ] 四個方法是否使用完全一致的 evaluation samples？
- [ ] Accuracy threshold 是否對所有 attacks 固定？
- [ ] `SpiderMark` 數字與目前 non-meta baseline 不完全相同的原因是什麼？
- [ ] 是否要保留完整九種 attack，或只在正文放 summary、完整表放 supplementary？

### 完成條件

- [x] 表格出現在 Experiments 的 backbone-selection subsection。
- [x] 正文只得出表格能支持的有限結論。
- [x] 清楚標明數據來源、unseen COCO 與共同 evaluation protocol。
- [ ] 加入表格後目前全文為 11 頁；依既定決定，不自行刪減，由導師處理頁數取捨。

---

## 5. 全文替換為新標題（暫不執行）

### 決定

導師會先為模型／方法提供正式名稱。在此之前：

- 不替換論文標題；
- 不統一修改 `MetaSpiderMark` 或 `Agentic MetaSpiderMark`；
- 不變更圖表、README、supplementary 或實驗文件中的名稱；
- 避免現在進行大範圍替換，之後又因正式名稱不同而重做。

### 導師命名後需要確認

- [ ] 模型／方法的正式名稱。
- [ ] 論文的正式標題。
- [ ] 是否在標題中保留 `SpiderMark`。
- [ ] `MetaSpiderMark` 是否專指 meta-learning framework？
- [ ] `Agentic MetaSpiderMark` 是否只指使用 LLM/agentic scheduler 的版本？
- [ ] deterministic residual scheduler 是否仍稱為 agentic？

### 正式名稱確認後才檢查

- [ ] `papers/meta_learning/main.tex`
- [ ] `papers/meta_learning/README.md`
- [ ] Abstract 與 Introduction
- [ ] Method、figure captions 與 algorithm caption
- [ ] Supplementary material
- [ ] Rebuttal template
- [ ] PDF metadata（若有設定）
- [ ] Experiment documentation

---

## 6. Miguel 實驗 README（已完成，不再處理）

### 決定

Miguel 所需的實驗 README／執行說明已經完成。本輪不再建立新的 `README_MIGUEL.md`，也不重寫現有的實驗文件。

### 執行原則

- [x] Miguel 的實驗 README／執行說明已完成。
- [x] 不建立額外的 `README_MIGUEL.md`。
- [ ] 除非 Miguel 或導師提出缺失，否則不修改現有說明。

---

## 7. 額外需要討論的核心問題

這些問題不是導師六點中的獨立項目，但會直接影響論文是否能支持目前主張。

### 7.1 Agentic scheduler 的貢獻是否已被隔離？

目前 epoch 116 的結果比較：

```text
non-meta SpiderMark
vs.
meta-trained Agentic MetaSpiderMark
```

這個比較同時改變了：

- supervised training → episodic meta-training；
- fixed task exposure → adaptive scheduling；
- 可能加入 LLM/agent decision。

因此尚不能單獨將全部 improvement 歸因於 agentic scheduler。

建議最少加入：

| Verifier update | Scheduler | 用途 |
|---|---|---|
| Standard supervised | Fixed mixture | 原始 baseline |
| FOMAML | Uniform | 隔離 meta-learning 貢獻 |
| FOMAML | Deterministic residual | 隔離 adaptive scheduling 貢獻 |
| FOMAML | LLM residual | 隔離 agent/LLM 貢獻 |

### 7.2 Checkpoint selection

目前：

- epoch 116 mean accuracy：`0.9182`
- final checkpoint mean accuracy：`0.8959`

Final checkpoint 在 crop、warp 等條件下可能低於 non-meta baseline。因此需要事先定義：

- validation split；
- checkpoint metric；
- selection frequency；
- final test set；
- 是否禁止用最終報告的九個 attacks 選 checkpoint。

### 7.3 最終論文應報告的 metrics

主要：

- Mean accuracy；
- Mean AUROC；
- Worst-attack accuracy；
- Worst-attack AUROC；
- 每個 attack 的 accuracy/AUROC；
- 多 seeds 的 mean ± standard deviation。

建議增加：

- TPR at a fixed low FPR；
- threshold transfer across attacks；
- scheduler/API overhead；
- training time；
- agent call count and cost。

---

## 8. 建議修改順序

### Phase A：先做、不依賴新實驗

- [x] 暫不修改標題與方法名稱；等待導師命名。
- [x] 保留導師已加入的 meta-learning inner/outer-loop 說明。
- [x] 修正 Method 中的 figure reference、紅色 TODO 與 Hospedales citation。
- [x] 插入 COCO backbone-selection table。
- [x] 暫不因八頁限制刪除 roadmap、表格或 diagnostics；等待導師決定。
- [x] Miguel 的實驗 README 已完成，不再處理。

### Phase B：實驗完成後

- [ ] 填入 matched scheduler comparison。
- [ ] 加入多 seeds 結果。
- [ ] 固定 checkpoint-selection protocol。
- [ ] 完成 held-out attack 或 COCO generalization。
- [ ] 報告 agent/scheduler overhead。

### Phase C：最後寫作

- [ ] 導師確認模型名稱與論文標題後，再進行全文命名替換。
- [ ] 重寫 contributions。
- [ ] 重寫 discussion。
- [ ] 重寫 conclusion，移除所有 `next step`、`planned`、`preliminary` 或 draft 語氣。
- [ ] 檢查全文 title/method naming。
- [ ] 編譯並向導師回報頁數；由導師決定八頁版本的內容取捨。
- [ ] 修正所有 undefined references、citations 與 float layout。

---

## 9. 下一次討論建議

建議依以下順序作決定：

1. **論文主要 claim 是 meta-learning，還是 agentic scheduler？**
2. **Agentic scheduler 與 deterministic residual controller 的正式定義是什麼？**
3. **COCO 表格數字的來源與 protocol 是否完全可追溯？**
4. **哪些 scheduler experiments 必須由 Miguel 完成？**
5. **哪些圖表保留在八頁正文？**
6. **確認後再開始直接修改 LaTeX。**
