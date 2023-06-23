# qulacs2023_benchmark

## 目的
このリポジトリは，新しい設計の Qulacs の行列演算バックエンドの選定を行う中で，各種ライブラリのベンチマークを取るためのリポジトリです．

## ベンチマークの対象と変数
- 対象となるライブラリ: OpenACC，SYCL，Qulacs(現行版)，Qulacs(AVX-512版)
- 変数:
  - 量子ビット数
- 定数:
  - 計測の繰り返し回数
  - 回路の種類
  - CPU や GPU に関する情報

## ベンチマークの流れ
1. 対象となるライブラリを指定
2. 指定したライブラリを用いた実装に対して，ベンチマークを取る
3. 各実装は，各量子ビット数に対して，指定された繰り返し回数だけ実行時間(ms 単位)を計測し，その平均値を出力する
4. 出力された実行時間やその他の情報を json ファイルに保存する(古い計測結果は上書き)

## ベンチマーク対象のプログラムのビルド/実行の要件
- プログラムは，`./benchmarks/${LIBRARY_NAME}/`(OpenACC の場合 `./benchmarks/OpenACC/`)に配置する
- 各ライブラリごとに Dockerfile を用意し，依存ライブラリのインストールなどを行う
- ローカルの `./benchmarks/${LIBRARY_NAME}/` は，Docker コンテナの `/benchmarks/` にバインドマウントされる
- ビルドは `./benchmarks/${LIBRARY_NAME}/build.sh` に記述する
- 実行は，`./benchmarks/${LIBRARY_NAME}/main` によって行う

### 入力
量子ビット数 `n` と繰り返し回数 `r` が標準入力として与えられます．
```
n r
```
例:
```
15 5
```

### 出力
量子ビット数 `n` の量子回路を繰り返し回数 `r` 回だけ実行し，それぞれの実行時間(ms 単位)を空白区切りで出力します．
以下は繰り返し回数が5回の場合の出力例です．
例:
```
2.5e-3 2.4e-3 2.3e-3 2.4e-3 2.4e-3
```

### 実装例
```cpp
#include <iostream>

int main() {
    int n;
    std::cin >> n;
    std::cout << n << std::endl;

    // ここに計測対象のプログラムを記述

    for (auto result : results) {
        std::cout << result << " ";
    }
    return 0;
}
