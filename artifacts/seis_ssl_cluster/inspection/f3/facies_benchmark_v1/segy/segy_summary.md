# F3 SEGY geometry inspection

- F3 root: `/home/dcuser/data/public_data/field/F3`
- class_info: `/home/dcuser/data/public_data/field/F3/interpretation/class_info.json`
- XYZ仮定: cube axis 0 -> x / inline, axis 1 -> y / crossline, axis 2 -> z / sample/time
- label値0は有効classとして扱う。

## Geometry

| role | cube_shape | iline | xline | sample/time | dtype |
|---|---|---|---|---|---|
| seismic | [601, 901, 255] | 100-700 (601) | 300-1200 (901) | 0-1270 (255) | float32 |
| label | [601, 901, 255] | 100-700 (601) | 300-1200 (901) | 0-1270 (255) | uint8 |

## Shape対応

- seismic shape: [601, 901, 255]
- label shape: [601, 901, 255]
- 一致: True

## Seismic amplitude統計

- finite_count: 138082755, nonfinite_count: 0, zero_count: 1083002
- min/p50/max: -1.0 / 0.00696202740073204 / 1.0
- p1/p99: -0.6514441919326782 / 0.5720649909973154
- mean/std: 0.0017741844254441973 / 0.210987974070619

## Label unique値

- integer-like: True
- unique values: [0, 1, 2, 3, 4, 5]
- unexpected label values: []

| class_id | class_name | present | count | color |
|---:|---|---|---:|---|
| 0 | Upper North Sea | True | 34940426 | #235CA7 |
| 1 | Middle North Sea | True | 15112086 | #7DB4D5 |
| 2 | Lower North Sea | True | 68326029 | #DBF1F7 |
| 3 | Rijnland/Chalk | True | 8651372 | #FEDB7C |
| 4 | Scruff | True | 8535595 | #FC783B |
| 5 | Zechstein | True | 2517247 | #D00A00 |
