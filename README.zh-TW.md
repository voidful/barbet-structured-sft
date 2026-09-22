# Barbet 1B Structured SFT

用於 ASR 轉寫候選、TTS 文字讀法、OCR 文字候選與翻譯後編修的
**結構化文字任務模型**，共 1,118,799,096 個參數。

[模型](https://huggingface.co/OpenFormosa/barbet-1b-structured-sft) · [說明網站](https://voidful.github.io/barbet-structured-sft/zh.html) ·
[English](README.md) · [實際預測](docs/examples.md)

## 目前能做什麼

在提供參照文字、完整任務契約與指定輸出格式時，可以執行候選修正、
證據標記與判定。固定合成測試 408 題皆能解析，整份答案完全相符
214/408（52.45%）；判定與證據清單各為 408/408。

**自由形式指令仍弱。** 另外 16 題新指令皆未答對；程式碼及數學的
BPB（越低越好）分別退步 27.36%、16.21%。這次只訓練文字，沒有直接
處理音訊、影像像素或產生語音波形。1M 是設定上限，並非本版重新驗證的能力。

## 開始使用

```bash
git clone https://github.com/voidful/barbet-structured-sft.git
cd barbet-structured-sft
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python scripts/infer.py --example ocr_correction --device cuda:0
```

已測環境為 Python 3.12、PyTorch 2.11.0、CUDA 13.0 與 H200。
Mamba 與 causal-conv1d 需要相容的 CUDA 建置環境。載入器會保留
336 個 FP32 Mamba 參數，其餘權重為 BF16；不要對整個模型呼叫
`.half()` 或 `.bfloat16()`。短輸入可使用 `--device cpu`。

## 訓練與公開證據

- 完整 980,000 筆 train 資料、1 epoch、7,657 步。
- 1,490,165,039 個輸入 tokens，225,601,357 個答案／EOS tokens 參與 loss。
- 不截斷、不跨樣本 packing；最長樣本 168,442 tokens。
- 所有匯出權重逐張量核對通過。此次公開發布沿用 2026-09-16 的權重，沒有再訓練。
- 原始 base 是獨立的存取受限倉庫；舊的 6,955 筆保留測試語料未公開。
  公開腳本與完整原始重訓所需的可取得資源，分別記載於[重現文件](docs/reproducing.md)。

## 文件

[使用方法](docs/usage.md) · [評估與限制](docs/evaluation.md) ·
[範例](docs/examples.md) · [架構](docs/architecture.md) ·
[貢獻方式](CONTRIBUTING.md) · [引用](CITATION.cff)

**授權狀態：目前為公開存取，尚未取得明確開源授權指定。**
原有 `other` 與資料集存取聲明保留，未自行改成 Apache、MIT 或 CC。
詳見 [LICENSE](LICENSE) 與[授權說明](docs/licensing.md)。
