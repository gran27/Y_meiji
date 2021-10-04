# Y-meiji
## Quick start
~~### A. gitを使う方法~~
~~1. gitのインストール~~
~~2. クローン~~
### B. zipでダウンロードする方法
Download ZIPからダウンロードできる。
フォルダの名前をY_meijiに直しておく。
![savezip](https://github.com/gran27/Y_meiji/blob/main/figs/savezip.png)
## How to use
1. コマンドプロンプトを立ち上げる。
2. ダウンロードした場所まで'cd Y_meiji'などで移動する。
3. MTSファイルをdataフォルダに入れておく。
4. 'python tools/eval.py data/01504.MTS --show -s 5'を実行
5. Y迷路に沿った線が表示されるので、確認する。確認し終わったら何かキーを押す（なんでもOK）。
![example_Y](https://github.com/gran27/Y_meiji/blob/main/figs/points_auto.png)
6. ラインが迷路に沿っている場合は'y'、沿っていない場合は'n'を入力してEnterを押す。
7. （5で'n'を押したときのみ）手動で4つの点を上のバーで調整する。決定するときは'**q**'を押すこと。
8. あとは待てば処理が終わる。'q'で中断することができる。resultフォルダに結果がcsvかtxt形式で保存されている。
![example](https://github.com/gran27/Y_meiji/blob/main/figs/ex_show.png)
![red circle](https://github.com/gran27/Y_meiji/blob/main/figs/incircle.png)

