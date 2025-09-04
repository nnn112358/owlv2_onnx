# OWLv2 物体検出

このプロジェクトは、PyTorchとONNX推論の両方を使用したOWLv2（Open-World Localization）モデルによる物体検出を実装しています。

## 特徴

- **PyTorch実装**: transformersライブラリを使用した直接的なモデル推論
- **ONNX最適化**: より高いパフォーマンスのためのONNXランタイムを使用した最適化推論
- **ゼロショット物体検出**: 自然言語記述を使用した物体検出
- **視覚的出力**: バウンディングボックス付きの検出結果を保存

## ファイル

- `owlv2_transformer_detec.py` - PyTorchベースのOWLv2物体検出
- `owl_onnx_inference.py` - ONNX最適化推論実装
- `owlv2-onnx/` - ONNXモデルファイルディレクトリ
- `test.jpg` - サンプル入力画像
- `test_detected.jpg` - 検出結果付き出力画像

## インストール

uvを使用して依存関係をインストール：

```bash
uv install
```

またはpipを使用：

```bash
pip install -r requirements.txt
```

## 使用方法

### PyTorch推論

```bash
python owlv2_transformer_detec.py
```

### ONNX推論

```bash
python owl_onnx_inference.py
```

## 依存関係

- numpy >= 1.21.0
- opencv-python >= 4.5.0  
- onnxruntime >= 1.15.0
- torch >= 2.5.1
- transformers >= 4.46.3
- pillow >= 10.4.0

## モデル

オープンボキャブラリ物体検出にGoogleのOWLv2ベースモデル（`google/owlv2-base-patch16-ensemble`）を使用。