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
量子回路の種類を表す整数`c` と量子ビット数 `n` ，繰り返し回数 `r` がコマンドライン引数として与えられます．
量子回路の種類を表す整数 `c` は，以下のように定義します．
- `c = 0`: {X,Y,Z,H}の中からランダムに選び、ランダムな量子ビットに作用させる
- `c = 1`: {RX,RY,RZ}の中からランダムに選び、ランダムな量子ビットに[0,2π)のランダムな角度で作用させるのを10回やるテスト
- `c = 2`: CNOTをランダムなターゲットとコントロールビットに10回作用させるテスト
- `c = 3`: ターゲット1コントロール0のDenseMatrixをランダムな量子ビットに作用させるのを10回やるテスト
- `c = 4`: ターゲット2コントロール0のDenseMatrixをランダムな量子ビットに作用させるのを10回やるテスト
- `c = 5`: ターゲット1,コントロール2のDenseMatrixをランダムな量子ビットに作用させるのを10回やるテスト
```
c n r
```
例:
```
15 5
```

### 出力
量子ビット数 `n` の量子回路を繰り返し回数 `r` 回だけ実行し，それぞれの実行時間(ms 単位)を空白区切りで `durations.txt` に出力します．
CUDA のコンテナを利用すると標準出力にバージョンの情報などが出力されるため，ファイルに出力するようにしています．
以下は繰り返し回数が5回の場合の出力例です．
例:
```
2.5e-3 2.4e-3 2.3e-3 2.4e-3 2.4e-3
```

### 実装例
```cpp
#include <iostream>

int main() {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }

    const auto n_qubits = std::strtoul(argv[1], nullptr, 10);
    const auto n_repeats = std::strtoul(argv[2], nullptr, 10);

    /*
    ここに計測対象のプログラムを記述
    */

    std::ofstream ofs("durations.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    for (int i = 0; i < n_repeats; i++) {
        ofs << measure() << " ";
    }
    ofs << std::endl;

    return 0;
}
