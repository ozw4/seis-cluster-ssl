# F3 facies benchmark inspection report

このreportはF3 facies benchmark inspectionの出力を統合し、少量教師の岩相判別MVPへ進むための判断材料をまとめる。

## 1. Dataset files

- seismic SEGY: 1件 (`f3_seismic.sgy`)
- label SEGY: 1件 (`f3_labels.sgy`)
- class_info: 1件 (`interpretation/class_info.json`)
- train PNG labels: 12件
- validation PNG labels: 3件

## 2. Volume geometry

- shape: [601, 901, 255]
- label shape: [601, 901, 255]
- shape一致: True
- inline range: {"count": 601, "max": 700, "min": 100}
- crossline range: {"count": 901, "max": 1200, "min": 300}
- sample range: {"count": 255, "max": 1270, "min": 0}
- repo internal axis assumption: x=inline, y=crossline, z=sample/time
- z display convention: XZ/YZ断面ではz/sample/time方向が下向きに増える表示にする。

## 3. Seismic amplitude statistics

- min / p1 / p50 / p99 / max: -1 / -0.651444 / 0.00696203 / 0.572065 / 1
- finite / nonfinite / zero count: 138082755 / 0 / 1083002

## 4. Facies classes

| class ID | class name | RGB color | pixel count | voxel count |
|---:|---|---|---:|---:|
| 0 | Upper North Sea | [35, 92, 167] | 685639 | 34940426 |
| 1 | Middle North Sea | [125, 180, 213] | 299281 | 15112086 |
| 2 | Lower North Sea | [219, 241, 247] | 1374140 | 68326029 |
| 3 | Rijnland/Chalk | [254, 219, 124] | 173265 | 8651372 |
| 4 | Scruff | [252, 120, 59] | 173737 | 8535595 |
| 5 | Zechstein | [208, 10, 0] | 51763 | 2517247 |

## 5. Train/validation labels

- PNG label files: 15
- total pixels: 2757825
- unknown pixels: 0
- slice list:
  - train inline 250: `interpretation/train/0001_labels_inline_0250.png`
  - train inline 350: `interpretation/train/0002_labels_inline_0350.png`
  - train inline 450: `interpretation/train/0003_labels_inline_0450.png`
  - train inline 550: `interpretation/train/0004_labels_inline_0550.png`
  - train inline 650: `interpretation/train/0005_labels_inline_0650.png`
  - train crossline 450: `interpretation/train/0006_labels_crossline_0450.png`
  - train crossline 550: `interpretation/train/0007_labels_crossline_0550.png`
  - train crossline 650: `interpretation/train/0008_labels_crossline_0650.png`
  - train crossline 850: `interpretation/train/0009_labels_crossline_0850.png`
  - train crossline 950: `interpretation/train/0010_labels_crossline_0950.png`
  - train crossline 1050: `interpretation/train/0011_labels_crossline_1050.png`
  - train crossline 1150: `interpretation/train/0012_labels_crossline_1150.png`
  - validation inline 150: `interpretation/validation/0001_labels_inline_0150.png`
  - validation crossline 350: `interpretation/validation/0002_labels_crossline_0350.png`
  - validation crossline 750: `interpretation/validation/0003_labels_crossline_0750.png`

### Class distribution by split

- train: files=12
  - class 0 Upper North Sea: 569684 pixels (0.256434)
  - class 1 Middle North Sea: 241208 pixels (0.108576)
  - class 2 Lower North Sea: 1112637 pixels (0.500836)
  - class 3 Rijnland/Chalk: 135660 pixels (0.0610652)
  - class 4 Scruff: 117163 pixels (0.0527391)
  - class 5 Zechstein: 45208 pixels (0.0203497)
- validation: files=3
  - class 0 Upper North Sea: 115955 pixels (0.216227)
  - class 1 Middle North Sea: 58073 pixels (0.108292)
  - class 2 Lower North Sea: 261503 pixels (0.487638)
  - class 3 Rijnland/Chalk: 37605 pixels (0.0701239)
  - class 4 Scruff: 56574 pixels (0.105496)
  - class 5 Zechstein: 6555 pixels (0.0122234)

### Imbalance notes

- train: class imbalance ratioが24.6。
- validation: class imbalance ratioが39.9。

## 6. PNG vs SEGY label consistency

- status: PASS
- PNG labels: 15
- max mismatch threshold: 0.001
- max observed mismatch rate: 0.00392592
- max observed effective mismatch rate: 4.38687e-06
- ignored z-border samples: 1
- total mismatch pixels: 10816
- border-only mismatch slices: 14
- note: raw mismatchはz-border sampleを含み、readiness判定はeffective mismatch rateを使う。
- warnings: one or more PNG labels required non-default orientation; see per-slice JSON metadata; one or more PNG/SEGY label mismatches are confined to ignored z-border samples

## 7. Quicklook figures

- [quicklook/seismic/seismic_xz_y_mid.png](quicklook/seismic/seismic_xz_y_mid.png) (exists=True)
- [quicklook/overlays/train_inline_0250_overlay.png](quicklook/overlays/train_inline_0250_overlay.png) (exists=True)
- [quicklook/tokenization/train_inline_0250_tokenization.png](quicklook/tokenization/train_inline_0250_tokenization.png) (exists=True)

## 8. Tokenization preview

- patch size: [8, 8, 8]
- retained token ratio: 0.95893
- ambiguous/dropped token ratio: 0.0410701 / 0.0410701
- total / retained / dropped tokens: 43584 / 41794 / 1790
- warnings:
  - patch単位の代表classは粗い教師ラベルであり、境界付近ではfacies混合を含む可能性がある。

## 9. Readiness for downstream

- 判定: `caution`
- 推奨: MVPへ進む前にwarningと不足componentを確認する。
- 理由:
  - raw mismatch is confined to ignored z-border samples.
- training前のrequired fixes:
  - z-border sample差分の影響範囲を確認する。

## Warnings

- label_consistency: one or more PNG labels required non-default orientation; see per-slice JSON metadata
- label_consistency: one or more PNG/SEGY label mismatches are confined to ignored z-border samples
