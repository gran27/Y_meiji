# Y-meiji
## Quick start
### A. gitを使う方法
~~1. gitのインストール~~
~~2. クローン~~
### B. zipでダウンロードする方法
Download ZIPからダウンロードできる。
フォルダの名前をY_meijiに直しておく。
![savezip](https://github.com/gran27/Y_meiji/blob/main/figs/savezip.png)
## How to use
### 1. コマンドプロンプトを立ち上げる。
Windowsキーを押してcmdと入力しEnter
### 2. ダウンロードした場所まで移動する。
- 例）Desktop上においてある場合
  ```
  cd Desktop/Y_meiji
  ```
### 3. 動画ファイルをdataフォルダに入れておく。
### 4. 実行
オプションの使い方はこちら（未完成）を参照。
- 例）01504.MTSを開始5.5秒の位置から評価するとき
  ```
  python tools/eval.py data/01504.MTS --show -s 5.5
  ```
#### 注意
- -sオプションの数字はマウスを置いた瞬間に合わせること。この時間から8分がカウントされる。
- 開始位置から動画の長さが8分もないときはプログラムが中断される。
  例）動画の長さが8分5秒で`python tools/eval.py data/XXXXX.MTS -s 10`と実行したとき
### 5. Y迷路の予測を確認する。
Y迷路に沿った線が表示されるので、確認する。確認し終わったら何かキーを押す（だいたいなんでもOK）。
- 良い例
![example_Y](https://github.com/gran27/Y_meiji/blob/main/figs/points_auto.png)
![example_Y](https://github.com/gran27/Y_meiji/blob/main/figs/points_auto_bad.png)
- 悪い例
### 6. ラインが迷路に沿っているかどうかを入力する。
ラインが迷路に沿っている場合は`y`、沿っていない場合は`n`を入力して`Enter`を押す。
### 7. （5で`n`を押したときのみ）手動で4つの点を上のバーで調整する。
決定するときは`**q**`を押すこと。
### 8. 処理が終わるのを待つ。
`q`で中断することができる。resultフォルダに結果がcsvかtxt形式で保存されている。
![example](https://github.com/gran27/Y_meiji/blob/main/figs/ex_show.png)
- 中央の赤い円の内側にマウスがいる時は追跡しない。
![red circle](https://github.com/gran27/Y_meiji/blob/main/figs/incircle.png)

