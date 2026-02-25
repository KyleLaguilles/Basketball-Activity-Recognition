```
(hangtime_har) jovyan@jupyter-kyle-laguilles-928-my-csun-edu---4d054b9a:~/hangtime_har$ python main.py --gpu cpu --epochs 1 --batch_size 64
Loading data...
Processing dataset files ...
Full dataset with size: | X (4933521, 4) | y (4933521,) | 
LOSO dataset with size: | (4933521, 5) |

CALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.


 VALIDATING FOR SUBJECT 05d8; 1 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1510 Train Acc (M): 30.02 (%) Train Prc (M): 32.97 (%) Train Rcl (M): 30.02 (%) Train F1 (M): 29.52 (%) 
Valid Loss: 1.1210 Valid Acc (M): 26.34 (%) Valid Prc (M): 69.09 (%) Valid Rcl (M): 26.34 (%) Valid F1 (M): 22.99 (%)
Performance improved... (0.0->0.22989383073821376)
SUBJECT 05d8 VALIDATION RESULTS: 
Accuracy: 26.34 (%)
Precision: 69.09 (%)
Recall: 26.34 (%)
F1: 22.99 (%)

 VALIDATING FOR SUBJECT 0846; 2 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1398 Train Acc (M): 30.38 (%) Train Prc (M): 35.01 (%) Train Rcl (M): 30.38 (%) Train F1 (M): 29.92 (%) 
Valid Loss: 1.3416 Valid Acc (M): 32.64 (%) Valid Prc (M): 78.02 (%) Valid Rcl (M): 32.64 (%) Valid F1 (M): 29.78 (%)
Performance improved... (0.0->0.2978200893579224)
SUBJECT 0846 VALIDATION RESULTS: 
Accuracy: 32.64 (%)
Precision: 78.02 (%)
Recall: 32.64 (%)
F1: 29.78 (%)

 VALIDATING FOR SUBJECT 846; 3 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1511 Train Acc (M): 30.19 (%) Train Prc (M): 32.66 (%) Train Rcl (M): 30.19 (%) Train F1 (M): 29.53 (%) 
Valid Loss: 0.9544 Valid Acc (M): 33.05 (%) Valid Prc (M): 80.88 (%) Valid Rcl (M): 33.05 (%) Valid F1 (M): 32.35 (%)
Performance improved... (0.0->0.3234520402140927)
SUBJECT 846 VALIDATION RESULTS: 
Accuracy: 33.05 (%)
Precision: 80.88 (%)
Recall: 33.05 (%)
F1: 32.35 (%)

 VALIDATING FOR SUBJECT 10f0; 4 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1700 Train Acc (M): 29.05 (%) Train Prc (M): 32.45 (%) Train Rcl (M): 29.05 (%) Train F1 (M): 28.51 (%) 
Valid Loss: 1.1216 Valid Acc (M): 27.48 (%) Valid Prc (M): 70.80 (%) Valid Rcl (M): 27.48 (%) Valid F1 (M): 26.11 (%)
Performance improved... (0.0->0.26105291363131006)
SUBJECT 10f0 VALIDATION RESULTS: 
Accuracy: 27.48 (%)
Precision: 70.80 (%)
Recall: 27.48 (%)
F1: 26.11 (%)

 VALIDATING FOR SUBJECT 2dd9; 5 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153

EPOCH: 1/1 
Train Loss: 1.1450 Train Acc (M): 29.41 (%) Train Prc (M): 33.69 (%) Train Rcl (M): 29.41 (%) Train F1 (M): 28.75 (%) 
Valid Loss: 1.0384 Valid Acc (M): 31.69 (%) Valid Prc (M): 83.39 (%) Valid Rcl (M): 31.69 (%) Valid F1 (M): 28.51 (%)
Performance improved... (0.0->0.2851226582752694)
SUBJECT 2dd9 VALIDATION RESULTS: 
Accuracy: 31.69 (%)
Precision: 83.39 (%)
Recall: 31.69 (%)
F1: 28.51 (%)

 VALIDATING FOR SUBJECT 4991; 6 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1493 Train Acc (M): 29.90 (%) Train Prc (M): 33.94 (%) Train Rcl (M): 29.90 (%) Train F1 (M): 29.19 (%) 
Valid Loss: 1.1191 Valid Acc (M): 27.70 (%) Valid Prc (M): 73.94 (%) Valid Rcl (M): 27.70 (%) Valid F1 (M): 25.81 (%)
Performance improved... (0.0->0.2580623483323252)
SUBJECT 4991 VALIDATION RESULTS: 
Accuracy: 27.70 (%)
Precision: 73.94 (%)
Recall: 27.70 (%)
F1: 25.81 (%)

 VALIDATING FOR SUBJECT 4d70; 7 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1421 Train Acc (M): 30.23 (%) Train Prc (M): 34.25 (%) Train Rcl (M): 30.23 (%) Train F1 (M): 29.75 (%) 
Valid Loss: 1.2403 Valid Acc (M): 27.34 (%) Valid Prc (M): 74.65 (%) Valid Rcl (M): 27.34 (%) Valid F1 (M): 22.93 (%)
Performance improved... (0.0->0.22927564921916688)
SUBJECT 4d70 VALIDATION RESULTS: 
Accuracy: 27.34 (%)
Precision: 74.65 (%)
Recall: 27.34 (%)
F1: 22.93 (%)

 VALIDATING FOR SUBJECT 9bd4; 8 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1569 Train Acc (M): 29.30 (%) Train Prc (M): 33.29 (%) Train Rcl (M): 29.30 (%) Train F1 (M): 28.98 (%) 
Valid Loss: 1.0309 Valid Acc (M): 31.92 (%) Valid Prc (M): 79.16 (%) Valid Rcl (M): 31.92 (%) Valid F1 (M): 28.13 (%)
Performance improved... (0.0->0.281337271513537)
SUBJECT 9bd4 VALIDATION RESULTS: 
Accuracy: 31.92 (%)
Precision: 79.16 (%)
Recall: 31.92 (%)
F1: 28.13 (%)

 VALIDATING FOR SUBJECT a0da; 9 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1476 Train Acc (M): 29.85 (%) Train Prc (M): 34.14 (%) Train Rcl (M): 29.85 (%) Train F1 (M): 29.22 (%) 
Valid Loss: 1.1034 Valid Acc (M): 27.77 (%) Valid Prc (M): 79.31 (%) Valid Rcl (M): 27.77 (%) Valid F1 (M): 23.18 (%)
Performance improved... (0.0->0.23177323958753243)
SUBJECT a0da VALIDATION RESULTS: 
Accuracy: 27.77 (%)
Precision: 79.31 (%)
Recall: 27.77 (%)
F1: 23.18 (%)

 VALIDATING FOR SUBJECT ac59; 10 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1548 Train Acc (M): 30.11 (%) Train Prc (M): 34.13 (%) Train Rcl (M): 30.11 (%) Train F1 (M): 29.22 (%) 
Valid Loss: 0.9708 Valid Acc (M): 32.50 (%) Valid Prc (M): 78.95 (%) Valid Rcl (M): 32.50 (%) Valid F1 (M): 28.27 (%)
Performance improved... (0.0->0.2826785857024034)
SUBJECT ac59 VALIDATION RESULTS: 
Accuracy: 32.50 (%)
Precision: 78.95 (%)
Recall: 32.50 (%)
F1: 28.27 (%)

 VALIDATING FOR SUBJECT b512; 11 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1531 Train Acc (M): 29.72 (%) Train Prc (M): 32.95 (%) Train Rcl (M): 29.72 (%) Train F1 (M): 28.96 (%) 
Valid Loss: 1.0283 Valid Acc (M): 32.68 (%) Valid Prc (M): 82.78 (%) Valid Rcl (M): 32.68 (%) Valid F1 (M): 29.65 (%)
Performance improved... (0.0->0.29646647611638544)
SUBJECT b512 VALIDATION RESULTS: 
Accuracy: 32.68 (%)
Precision: 82.78 (%)
Recall: 32.68 (%)
F1: 29.65 (%)

 VALIDATING FOR SUBJECT c6f3; 12 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1669 Train Acc (M): 29.58 (%) Train Prc (M): 32.93 (%) Train Rcl (M): 29.58 (%) Train F1 (M): 28.72 (%) 
Valid Loss: 1.1659 Valid Acc (M): 29.09 (%) Valid Prc (M): 82.59 (%) Valid Rcl (M): 29.09 (%) Valid F1 (M): 24.95 (%)
Performance improved... (0.0->0.24948222349846652)
SUBJECT c6f3 VALIDATION RESULTS: 
Accuracy: 29.09 (%)
Precision: 82.59 (%)
Recall: 29.09 (%)
F1: 24.95 (%)

 VALIDATING FOR SUBJECT ce9d; 13 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1745 Train Acc (M): 29.60 (%) Train Prc (M): 32.61 (%) Train Rcl (M): 29.60 (%) Train F1 (M): 28.84 (%) 
Valid Loss: 1.0351 Valid Acc (M): 33.58 (%) Valid Prc (M): 80.00 (%) Valid Rcl (M): 33.58 (%) Valid F1 (M): 31.15 (%)
Performance improved... (0.0->0.31150066630853757)
SUBJECT ce9d VALIDATION RESULTS: 
Accuracy: 33.58 (%)
Precision: 80.00 (%)
Recall: 33.58 (%)
F1: 31.15 (%)

 VALIDATING FOR SUBJECT e90f; 14 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1452 Train Acc (M): 29.92 (%) Train Prc (M): 45.40 (%) Train Rcl (M): 29.92 (%) Train F1 (M): 29.45 (%) 
Valid Loss: 1.0298 Valid Acc (M): 31.87 (%) Valid Prc (M): 84.87 (%) Valid Rcl (M): 31.87 (%) Valid F1 (M): 32.24 (%)
Performance improved... (0.0->0.3223804723797852)
SUBJECT e90f VALIDATION RESULTS: 
Accuracy: 31.87 (%)
Precision: 84.87 (%)
Recall: 31.87 (%)
F1: 32.24 (%)

 VALIDATING FOR SUBJECT f2ad; 15 OF 15
+----------------------------+------------+
|          Modules           | Parameters |
+----------------------------+------------+
| conv_blocks.0.conv1.weight |    704     |
|  conv_blocks.0.conv1.bias  |     64     |
| conv_blocks.0.conv2.weight |   45056    |
|  conv_blocks.0.conv2.bias  |     64     |
| conv_blocks.1.conv1.weight |   45056    |
|  conv_blocks.1.conv1.bias  |     64     |
| conv_blocks.1.conv2.weight |   45056    |
|  conv_blocks.1.conv2.bias  |     64     |
| lstm_layers.0.weight_ih_l0 |   98304    |
| lstm_layers.0.weight_hh_l0 |   65536    |
|  lstm_layers.0.bias_ih_l0  |    512     |
|  lstm_layers.0.bias_hh_l0  |    512     |
|         fc.weight          |    1152    |
|          fc.bias           |     9      |
+----------------------------+------------+
Total Params: 302153
EPOCH: 1/1 
Train Loss: 1.1575 Train Acc (M): 29.88 (%) Train Prc (M): 34.96 (%) Train Rcl (M): 29.88 (%) Train F1 (M): 29.33 (%) 
Valid Loss: 1.1307 Valid Acc (M): 29.35 (%) Valid Prc (M): 56.20 (%) Valid Rcl (M): 29.35 (%) Valid F1 (M): 27.99 (%)
Performance improved... (0.0->0.27994738702680017)
SUBJECT f2ad VALIDATION RESULTS: 
Accuracy: 29.35 (%)
Precision: 56.20 (%)
Recall: 29.35 (%)
F1: 27.99 (%)
FINAL VALIDATION RESULTS: 
Accuracy: 30.25 (%)
Precision: 59.10 (%)
Recall: 30.25 (%)
F1: 28.40 (%)
FINAL VALIDATION RESULTS (PER CLASS): 
Accuracy: [7.79994994e-02 1.61138714e-03 0.00000000e+00 0.00000000e+00
 3.21027287e-04 9.00799168e-01 8.43084913e-01 2.02108008e-01
 6.96321419e-01]
Precision: [0.65557468 0.13333333 1.         1.         0.0625     0.61984503
 0.59563031 0.58842139 0.66350122]
Recall: [7.79994994e-02 1.61138714e-03 0.00000000e+00 0.00000000e+00
 3.21027287e-04 9.00799168e-01 8.43084913e-01 2.02108008e-01
 6.96321419e-01]
F1: [1.39411932e-01 3.18429083e-03 0.00000000e+00 0.00000000e+00
 6.38773555e-04 7.34367564e-01 6.98076894e-01 3.00873504e-01
 6.79515250e-01]
GENERALIZATION GAP ANALYSIS: 
Train-Val-Accuracy Difference: -0.004316035313842614
Train-Val-Precision Difference: -0.256300937418463
Train-Val-Recall Difference: -0.004316035313842614
Train-Val-F1 Difference: 0.008106120850971787

ALL FINISHED
```
