# OWLv2 ONNX エクスポートと推論

OWLv2（Objectness-aware Vision Language Model）をONNX形式にエクスポートし、軽量な依存関係で推論を実行するプロジェクトです。

## セットアップ

必要な依存関係をインストール:

```bash
uv add torch torchvision transformers Pillow requests optimum[onnxruntime] scipy opencv-python
```

## ONNXモデルのエクスポート

optimum-cliを使用してOWLv2モデルをONNX形式にエクスポート:

```bash
uv run optimum-cli export onnx --model google/owlv2-base-patch16-ensemble --task zero-shot-object-detection ./owlv2-onnx/
```

**重要な注意点:**
- `zero-shot-object-detection` タスクを使用（`object-detection`ではない）
- 約614MBのONNXモデルファイルが生成される
- 精度差に関する警告メッセージは正常で無視して構わない

**エクスポート出力:**
- `model.onnx` - メインのONNXモデルファイル
- `config.json` - モデル設定
- `tokenizer.json`, `vocab.json` - トークナイザファイル
- `preprocessor_config.json` - 前処理設定

## 推論実行

### transformersライブラリを使用した推論
```bash
uv run python owl_onnx_inference.py
```

## ファイル構成

- `owl_onnx_inference.py` - transformersを使用したONNX推論
- `owlv2_transformer_detec.py` - 元のPyTorch実装

## 使用例

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

# ONNXモデルを読み込み
session = ort.InferenceSession("./owlv2-onnx/model.onnx")

# 画像とテキストを処理
image = Image.open("test.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a person"]

# 推論実行
# ... (詳細な実装は owl_onnx_inference.py を参照)
```

## 結果

モデルはテキストクエリに基づいてオブジェクトを検出し、以下を出力:
- ピクセル座標でのバウンディングボックス
- 信頼度スコア
- クラスラベル
- `test_detected.jpg`として可視化結果を保存

## パフォーマンス情報

- ONNXモデルはデフォルトでCPUで実行
- 推論時間: CPUで約30-60秒
- 高速化にはCUDAによるGPUアクセラレーションを推奨
- モデルサイズ: 約614MB