# Deep Q-Networkをフレームワークを使用せず構築　CartPoleを解く

<br>フレームワークを使用せず、numpyだけでDeep Q-Networkを構築しました。CartPole問題を解いています。<br>
Multi-step LearningとHuber損失を組み込んでいます。<br>
<BR>
| 未訓練モデルでPlay<br>10ステップ程度ですぐに倒れます | 訓練済モデルでPlay<br>上限200ステップいっぱい粘ります<br>（Multi-step Learning n=2で訓練） |
|      :---:       |     :---:      |
|![beginner_300_450](https://user-images.githubusercontent.com/52105933/92696762-4f02b980-f385-11ea-9aa4-86e5272899ac.gif)|![202009121127_300_450_008](https://user-images.githubusercontent.com/52105933/92989792-cb2a0800-f511-11ea-9a23-140799071c0c.gif)|


## 概要
フレームワークを使用せず、numpyだけでDeep Q-Networkを構築しました。CartPole問題を解いています。<br>
Multi-step Learning、Huber損失、Target-QN固定、Experience Replayを組み込んでいます。<br><br>

###  フレームワークを使用せずnumpyだけで実装
フレームワークを使用せずにnumpyだけで実装しています。<br>
Huber損失、誤差逆伝播、その他諸々を自力で0から実装しています。<br>
動作が軽いです。
<br><br>

## ディレクトリ構成・動かすのに必要な物
CartPole.py<BR>
common/<br>
&nbsp;└tools.py<br>
-----以下、デモ用ノートブック関連-----<br>
CartPole_demo.ipynb<BR>
demo_model_params/<br>
&nbsp;└（デモ用の訓練済パラメーターのpickleファイル）<br>
※（必要ならば）OpenAI Gym　のインストール（デモ用ノートブックCartPole_demo.ipynbにインストールのコードセルがあり、その実行でもOK）<br>
-----------------------------------------------------------------------------------------------------<br>
- CartPole.py：モデル本体。中身はclass Planner です。モデルを動かすにはcommonフォルダが必要です。
- CartPole_demo.ipynb：デモ用のノートブックです。概要をつかむことが出来ます。このノートブックを動かすにはdemo_model_paramsフォルダが必要です。
<br>

## 組み込んでいる主な要素
- Huber損失
- Multi-step Learning
- Target-QN固定
- Experience Replay
<br>

###  Huber損失
本モデルでは、訓練の安定化のため、2乗和損失では無く、Huber損失を採用しています。
<br><br>

#### Huber損失の定義：
yを推論値、tを正解値として、
> 0.5 * |y - t|^2 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for |y - t|<=delta) <br>
> delta * (|y - t| - 0.5 * delta)&nbsp;&nbsp;&nbsp;&nbsp;(otherwise)
<BR>

#### Huber損失の計算グラフ：
順伝播と誤差逆伝播の計算グラフです。手書きですみません・・。<br>
推論値yと正解値tの誤差の絶対値abs_diffが定数deltaをいくら超えても、最終的な勾配の絶対値は定数deltaに抑えられることが分かります。
|\|y - t\|とdeltaの大小関係|勾配| 
|     :---:     |:---      | 
|\|y - t\| <= delta|y - t　（2乗和損失と同じ）| 
|\|y - t\| > delta|delta| 
 
![Huber損失_計算グラフ_加工3_80](https://user-images.githubusercontent.com/52105933/92437696-74a38d80-f1e2-11ea-8178-0887fcda21d4.png)

<br><br>

### Multi-step Learning
本モデルでは、Main-QNに与える教師信号の算出において、Multi-step Learningを採用しています。
<br><br>

#### Multi-step Learningでの教師信号の定義：
tを時間軸上の現在、kをそれより未来のステップ数として、<br>
![教師信号一般式](https://user-images.githubusercontent.com/52105933/92439352-a36f3300-f1e5-11ea-9cf4-6431b9ff1f1a.png)<BR>
Target-QN固定方式を採用しているので、第2項のQ値は、Target-QNの出力値です。
<BR><br>

#### Main-QNとTarget-QNとMulti-step Learningの図：
n=3、即ち3ステップ先の状態のQ値から教師信号を算出する場合の図です。
<BR><br>
![multi_step_learningでの訓練図_80](https://user-images.githubusercontent.com/52105933/92442205-73765e80-f1ea-11ea-9561-ea125d31ff3b.png)
<br><br>

### 報酬設計
1ステップ終了時には報酬を与えません。<br>
代わりに、1エピソード終了時、条件次第で報酬を与えます。<br>
以下2通りの報酬のタイプがあります。訓練関数train()の引数「reward_type」で選択します。<br>
※下記「閾値」：訓練関数train()の引数「steps_success_over」で、既定値は160<br>

#### ＜タイプ0＞成功エピソードのみ報酬　報酬はステップ数次第で増える（クリッピングしない）：
| エピソード成功/失敗| 条件| 報酬 |
|     :---:     |     :---:      | :---         | 
|成功| エピソードのステップ数 > 閾値|1.0 + (エピソードのステップ数 ÷ 閾値)| 
|失敗| エピソードのステップ数 <= 閾値|0| 

この報酬タイプでのMulti-step Learningのn=1～3での訓練過程の比較<br>
エピソード毎の到達ステップ数　※200ステップ/エピソード がCartPole仕様上の上限<br><br>
![step_count_グラフ_mslern123_reward0_all_80](https://user-images.githubusercontent.com/52105933/93416922-be2a6180-f8e1-11ea-8d59-0544dffb05e1.png)

#### ＜タイプ2＞成功エピソードの報酬は1、失敗エピソードの報酬は-1（クリッピングする）：
| エピソード成功/失敗| 条件| 報酬 |
|     :---:     |     :---:      | :---         | 
|成功| エピソードのステップ数 > 閾値|1| 
|失敗| エピソードのステップ数 <= 閾値|-1| 

この報酬タイプでのMulti-step Learningのn=1～3での訓練過程の比較<br>
エピソード毎の到達ステップ数　※200ステップ/エピソード がCartPole仕様上の上限<br><br>
![step_count_グラフ_mslern123_reward2_all_80](https://user-images.githubusercontent.com/52105933/93418104-938dd800-f8e4-11ea-97f2-d8cb1db138f6.png)

<br>
報酬のクリッピングをし、Multi-step Learningのnが高いほどが訓練が安定し、且つ効果があるように見えます。<br>
ただし、全く同一の条件でも、<br>
・訓練開始から上限ステップ数200に初めて到達するまでの早さ（エピソード数）<br>
・上限ステップ数200に到達したエピソード回数<br>
に変動が毎回あり、<b>一概に報酬のクリッピングをし、Multi-step Learningのnを高くすれば効果がある、とは言えない</b>です。<br>
現時点の感覚では、上記2つの尺度に照らし合わせると、<br>
報酬のクリッピングをした方がやや良くなるものの、Multi-step Learningについてはn=2が一番良いがn=1（Single-step）と大差ないかな、と思っています。

<br><br>

## モデルの構成
CartPole.pyのclass Planner が、モデルの実体です。<br><br>
![モデル構成図](https://user-images.githubusercontent.com/52105933/92447938-e552a600-f1f2-11ea-961e-8f8c24277405.png)

このclass Planner をアプリケーション内でインスタンス化して、訓練やCartPoleのPlayといったpublicインターフェースを呼び出す、という使い方をします。<br>
Main-QN、Target-QN、経験データを蓄積するExperience BufferはPlanner内部に隠蔽され、外部から利用することはできません。
```
#モデルのインポート 
from CartPole import Planner #モデル本体

#Cart Poleの環境の活性化
env = gym.make("CartPole-v0")
  
#モデルのインスタンスを生成 
p_model_instance = Planner(hoge, hoge) 

#以下、モデルインスタンスに対してそのpublicインターフェースを呼ぶ

#このモデルインスタンスの訓練 
result = p_model_instance.train(hoge, hoge, …)

#この訓練済モデルインスタンスにCart PoleをPlayさせる 
try:

    curr_st = env.reset()
    env.render(mode='human')
    
    for st in range(200):
            
        #モデルインスタンスが最適な行動を推測
        action_predicted = p_model_instance.predict_best_action(curr_st) 
        
        #その行動を環境に指示
        next_st, _, done, _ = env.step(action_predicted)

        #レンダリング（注意！Google ColaboratoryのようなGUI描画ウィンドウが無い実行環境では、このままでは実行できません。）
        env.render(mode='human')

        if done==True:
            #エピソード終了
            #200ステップに達した（CartPole環境自体が強制終了する）か、倒れてしまったか
           break

        curr_st = next_st

finally:
    env.close()

#この訓練済モデルインスタンスの訓練済パラメーターの保存
p_model_instance.save_params_in_file(hoge, hoge, …)

#別の訓練済パラメーターをこのモデルインスタンスに読み込む
p_model_instance.overwrite_params_in_file(hoge, hoge, …)
```
<br><br>

## class Planner　のpublicインターフェース

#### class Planner　のpublicインターフェース一覧
| 名前 | 関数/メソッド/プロパティ | 機能概要・使い方 |
| :---         |     :---:      | :---         |
|Planner|     -      |class Planner　のモデルインスタンスを生成する。<br>*model_instance* = Planner(name="hoge", env=env, state_dim=4, action_dim=2, exp_buffer_size=10000, huber_loss_delta=1.0)|
|train|     関数      |モデルインスタンスを訓練する。<br>result = *model_instance*.train(episodes=250, episodes_stop_success=5, episodes_main_params_copy=1, reward_type=0, steps_per_episode=200, steps_warm_up=10, steps_success_over=160, steps_multi_step_lern=1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_rate=0.001, batch_size=32, verbose=True)|
|predict_best_action|     関数      |モデルインスタンスが最適な行動を推測する。<br>best_action = *model_instance*.predict_best_action(a_state=hoge)|
|save_params_in_file|     関数      |モデルインスタンスのパラメーターをファイル保存する。<br>file_name = *model_instance*.save_learnable_params_in_file(file_dir=hoge, file_name=hoge)|
|overwrite_params_in_file|     メソッド      |モデルインスタンスのパラメーターを、ファイル保存された別のパラメーターで上書きする。<br>*model_instance*.overwrite_learnable_params_in_file(file_path=hoge)|
|env|     getterプロパティ      |モデルインスタンスが対象としている環境。インスタンス化時に指定された物。<br>env = *model_instance*.env|
|state_dim|     getterプロパティ      |モデルインスタンスが認識している、状態の要素数。インスタンス化時に指定された物。<br>state_dim = *model_instance*.state_dim|
|action_dim|     getterプロパティ      |モデルインスタンスが認識している、行動の要素数。インスタンス化時に指定された物。<br>action_dim = *model_instance*.action_dim|
|name|     getter/setterプロパティ      |モデルインスタンスの名前。<br>getter : hoge = *model_instance*.name<br>setter : *model_instance*.name = hoge|
|count_experiences_in_buffer|     getterプロパティ      |モデルインスタンスの経験バッファが保持している経験データ数。<br>count_exp_data = *model_instance*.count_experiences_in_buffer|
|huber_loss_delta|     getterプロパティ      |モデルインスタンスのHuber損失の定数delta。インスタンス化時に指定された物か、overwrite_params_in_file()で上書きされた物。<br>delta = *model_instance*.huber_loss_delta|
<br>

### class Planner　のインスタンス化　*model_instance* = Planner(name, env, state_dim=4, action_dim=2, exp_buffer_size=10000, huber_loss_delta=1.0))
class Plannerのインスタンスを生成する。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|name|文字列|必須|このモデルインスタンスの名前。|
|env|gym.wrappers.time_limit.TimeLimit|必須|CartPole環境のオブジェクトインスタンス。<br>env = gym.make("CartPole-v0")　と生成する。|
|state_dim|整数|4|状態の要素数。<br>今後、このインスタンスは、状態の要素数はここで指定されたものであるという前提で挙動する。変更方法は無い。|
|action_dim|整数|2|行動の要素数。<br>今後、このインスタンスは、行動の要素数はここで指定されたものであるという前提で挙動する。変更方法は無い。|
|exp_buffer_size|整数|10000|経験バッファのサイズ。<br>経験バッファが満杯になった場合、過去の経験データから順に消されていく。|
|huber_loss_delta|浮動小数点数|1.0|Huber損失の定数delta。|

<br>

### ＜関数＞result = *model_instance*.train(episodes, episodes_stop_success, episodes_main_params_copy=1, reward_type=0, steps_per_episode=200, steps_warm_up=10, steps_success_over=160, steps_multi_step_lern=1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_rate=0.001, batch_size=32, verbose=True)
モデルインスタンスを訓練します。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|episodes|整数|必須|訓練に費やすエピソード数。|
|episodes_stop_success|整数|必須|訓練を終了する成功エピソード数。成功したエピソード回数がこの数に達すると、訓練を終了する。|
|episodes_main_params_copy|整数|1|エピソード何回毎に、Main-QNのパラメーターをTarget-QNにコピーするか。<br>指定値にかかわらず、訓練の冒頭と終了時にもコピーされる。|
|reward_type|整数|0|報酬設計のタイプ。<br>「0」を指定：成功エピソードのみ、1.0 + (エピソードのステップ数 ÷ steps_success_over)の報酬を与える。<br>「2」を指定：成功エピソードでは1、失敗エピソードでは-1の報酬を与える。<br>※「1」は指定しない。|
|steps_per_episode|整数|200|1エピソードで何ステップまで行うか。<br>CartPole仕様上の上限は200なので、200超の数字を指定しても200ステップとなる。|
|steps_warm_up|整数|10|訓練開始後、経験バッファに経験データを格納しない待機ステップ数。|
|steps_success_over|整数|160|何ステップを超えたら、そのエピソードを成功とみなすか。|
|steps_multi_step_lern|整数|1|Multi-step Learningのn。Single-stepにしたい場合は1を指定。|
|gamma|浮動小数点数|0.99|教師信号算出要素のγ。報酬やQ値の現在価値算出のための割引率。|
|epsilon_start|浮動小数点数|1.0|ε-greedyのεについての引数。<br>εをステップのたびに下記の計算式で逓減させていく。その初期値。<br>ε = ε下限値 + (ε初期値 - ε下限値) * exp(-減衰率*現在のステップ総数)|
|epsilon_end|浮動小数点数|0.1|ε-greedyのεについての引数。上記のε下限値。|
|epsilon_decay_rate|浮動小数点数|0.001|ε-greedyのεについての引数。上記の減衰率。|
|batch_size|整数|32|Main-QNのパラメーター更新時に経験バッファのデータを何件ミニバッチにして与えるか。|
|verbose|boolean|True|訓練途中で途中経過の文字列を出力するか。|

#### 戻り値「result」（Dictionary）の内部要素：
| key文字列 | 型 | 意味 |
| :---         |     :---:      | :---         |
|name|文字列|このモデルインスタンス名。|
|consecutive_num_train|整数|このモデルインスタンスにおける訓練通し番号。train()が呼ばれるたびに1増えていく。|
|episode_count|整数|実際のエピソード数。|
|step_count_episodes|list|各エピソードのステップ数のエピソード毎の履歴。listの1要素は1エピソード。|
|loss_episodes|list|各エピソードのMain-QNのlossのエピソード毎の履歴。listの1要素は1エピソード。|
|success_episodes|list|各エピソードの成功（1）/失敗（0）のエピソード毎の履歴。listの1要素は1エピソード。|
|episode_count_success|整数|実際の成功エピソード数。|
|total_step_count|整数|実際のステップ総数。|
|train()の引数|-|引数の指定値。|

<br>

#### train()内の流れ：<br>
訓練開始からsteps_warm_upのステップを実行後、経験バッファに経験データを蓄積し始めます。<br>
経験バッファにbatch_size分の経験データが蓄積されてから、Main-QNの訓練を始めます。<BR><BR>
![train関数処理単位と時間軸2](https://user-images.githubusercontent.com/52105933/92565255-50fd4780-f2b5-11ea-8f4e-365bddd7119f.png)
<!--![train関数処理単位と時間軸](https://user-images.githubusercontent.com/52105933/92488826-01be0500-f22a-11ea-911f-7e3bcf635333.png)-->

<BR>

#### 訓練中のベストなパラメーターを採用：<br>
1回の訓練中、複数回のエピソード、さらにエピソード1回の中でも複数回のステップを繰り返し、そのたびにMain-QNのパラメーターが更新されます。<BR>
たとえ訓練途中でステップ数200を記録したエピソードがあったとしても、訓練終了時の最終エピソードでのステップ数が少なかった場合、モデルインスタンスは”悪い”性能で訓練を終えた、ということになってしまいます。<BR>
そうならないために、最多ステップ数を記録した直近のエピソード時のMain-QNのパラメーターを一時退避しておき、訓練終了時にそのパラメーターを採用するようにしています。<BR><BR>
![step_count_グラフ_mslern1_reward0_202009062353_bestパラメーター採用説明_加工](https://user-images.githubusercontent.com/52105933/92502674-325a6a80-f23b-11ea-861d-3ae79b351fd8.png)

<br>

### ＜関数＞best_action = *model_instance*.predict_best_action(a_state)
与えられた状態（1個）での最適な行動を推測します。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|a_state|ndarray<br>shapeは(インスタンス化時指定のstate_dim,)<br>又は(1, インスタンス化時指定のstate_dim)|必須|行動を取ろうとしている状態。1個のみ。<br>```a_state = env.reset()```<br>又は<br>```a_state, _, done, _ = env.step(直前のaction)```<br>の戻り値を利用する。|

#### 戻り値：<BR>
- best_action<BR>
推測結果の行動。shapeはスカラー。「0」が左で、「1」が右。<BR>
ここで得られたbest_actionは、以下のように環境に指示して、利用する。<br>
```next_state, _, done, _ = env.step(best_action)```

<br>

### ＜関数＞file_name = *model_instance*.save_params_in_file(file_dir, file_name="")
現在のモデルインスタンスのMain-QNの訓練対象パラメーター（weightsとbiases）とHuber損失の定数deltaをpickleファイルに保存し、後に再利用できるようにします。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|file_dir|文字列|必須|保存ファイルを置くディレクトリ。|
|file_name|文字列|空文字列|保存ファイル名（拡張子も含める）。<br>空文字列の場合、モデルインスタンス名.pickle　というファイル名になる。|

#### 戻り値：<BR>
- file_name<BR>
実際の保存ファイル名。

<br>

### ＜メソッド＞*model_instance*.overwrite_params_in_file(file_path)
pickleファイルに保存された訓練対象パラメーターとHuber損失の定数deltaを読み込み、現在のモデルインスタンスのMain-QNのパラメーターにします（パラメーターを上書きします）。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|file_path|文字列|必須|保存ファイルのパス（拡張子も含める）。|

<br>

### ＜getterプロパティ＞*model_instance*.env
このモデルインスタンスが対象としている環境を返します。<br>
インスタンス化時に指定された物です。

<br>

### ＜getterプロパティ＞*model_instance*.state_dim
このモデルインスタンスが認識している、状態の要素数を返します。<br>
インスタンス化時に指定された物です。

<br>

### ＜getterプロパティ＞*model_instance*.action_dim
このモデルインスタンスが認識している、行動の要素数を返します。<br>
インスタンス化時に指定された物です。

<br>

### ＜getter/setterプロパティ＞*model_instance*.name
getterは、このモデルインスタンスの名前を返します。<br>
setterは、このモデルインスタンスの名前を設定します。<br>

#### setterが受け取る値：
| 型 |  意味 |
|     :---:      | :---         |
|文字列|モデルインスタンスの新しい名前。|

<BR>

### ＜getterプロパティ＞*model_instance*.count_experiences_in_buffer
このモデルインスタンスの経験バッファが保持している経験データ数を返します。

<br>

### ＜getterプロパティ＞*model_instance*.huber_loss_delta
このモデルインスタンスのHuber損失の定数deltaを返します。<br>
インスタンス化時に指定された物か、overwrite_params_in_file()で上書きされた物です。

<br>

※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。