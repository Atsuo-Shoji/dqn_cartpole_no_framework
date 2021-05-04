# Deep Q-NetworkでCartPoleを解く（フレームワーク不使用）

<br>Deep Q-Networkを構築しました。CartPole問題を解いています。<br>
Multi-step LearningやHuber損失などを組み込んでいます。<br>
報酬のクリッピングの有無や、Multi-step Learningの長さの違いでの比較を行っています。<br>

フレームワークを使用せず、主にnumpyだけで構築しています。<br>

<BR>
 
| 未訓練モデルでPlay<br>10ステップ程度ですぐに倒れます || DQN訓練済モデルでPlay<br>上限200ステップいっぱい粘ります<br>（Multi-step Learning n=2で訓練） |
|      :---:       |     :---:      |     :---:      |
|![beginner_300_450](https://user-images.githubusercontent.com/52105933/92696762-4f02b980-f385-11ea-9aa4-86e5272899ac.gif)|![矢印（赤）](https://user-images.githubusercontent.com/52105933/110228721-b0779f80-7f46-11eb-8cd9-469501beea50.png)|![202009121127_300_450_008](https://user-images.githubusercontent.com/52105933/92989792-cb2a0800-f511-11ea-9a23-140799071c0c.gif)|


## 概要
Deep Q-Networkを構築しました。CartPole問題を解いています。<br>
Multi-step Learning、Huber損失、Target-QN固定、Experience Replayを組み込んでいます。<br>
報酬のクリッピングの有無や、Multi-step Learningの長さの違いでの比較を行っています。<br>

実装に際しては、フレームワークを使用せず、主にnumpyだけを使用しています。<br>
Huber損失、誤差逆伝播、その他諸々を自力で0から実装しています。<br>
動作が軽いです。<br>

※理論の説明は基本的にしていません。他のリソースを参考にしてください。<br>
&nbsp;&nbsp;ネットや書籍でなかなか明示されておらず、私自身が実装に際し情報収集や理解に不便を感じたものを中心に記載しています。<br><br>

## 組み込んでいる主な要素
- Huber損失
- Multi-step Learning
- Target-QN固定
- Experience Replay
- 報酬のクリッピング
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
 
![Huber損失_計算グラフ_加工3_70](https://user-images.githubusercontent.com/52105933/116974395-096e7480-acf9-11eb-894c-4f69ff6b8054.png)

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
![multi_step_learningでの訓練図_80](https://user-images.githubusercontent.com/52105933/116348285-c90a8480-a828-11eb-8037-ccfb1fff2855.png)
<br><br>

### 報酬設計
1ステップ終了時には報酬を与えません。<br>
代わりに、**1エピソード終了時のみ、条件次第で報酬を与えます。**<br>
以下2通りの報酬のタイプがあります。訓練関数train()の引数「reward_type」で選択します。<br>
※下記「閾値」：訓練関数train()の引数「steps_success_over」で、既定値は160<br>

#### ＜クリッピングしない＞成功エピソードのみ報酬　報酬はステップ数次第で増える：
| エピソード成功/失敗| 条件| 報酬 |
|     :---:     |     :---:      | :---         | 
|成功| エピソードのステップ数 > 閾値|1.0 + (エピソードのステップ数 ÷ 閾値)| 
|失敗| エピソードのステップ数 <= 閾値|0| 

この報酬タイプでのMulti-step Learningのn=1～3での訓練過程の比較<br>
エピソード毎の到達ステップ数　※200ステップ/エピソード がCartPole仕様上の上限<br><br>
![step_count_グラフ_mslern123_reward0_all_80](https://user-images.githubusercontent.com/52105933/93416922-be2a6180-f8e1-11ea-8d59-0544dffb05e1.png)

#### ＜クリッピング＞成功エピソードの報酬は1、失敗エピソードの報酬は-1：
| エピソード成功/失敗| 条件| 報酬 |
|     :---:     |     :---:      | :---         | 
|成功| エピソードのステップ数 > 閾値|1| 
|失敗| エピソードのステップ数 <= 閾値|-1| 

この報酬タイプでのMulti-step Learningのn=1～3での訓練過程の比較<br>
エピソード毎の到達ステップ数　※200ステップ/エピソード がCartPole仕様上の上限<br><br>
![step_count_グラフ_mslern123_reward2_all_80](https://user-images.githubusercontent.com/52105933/93418104-938dd800-f8e4-11ea-97f2-d8cb1db138f6.png)

<br>
報酬のクリッピングをし、Multi-step Learningのnを高くするほど訓練が安定し、且つ効果があるように見えます。<br>
ただし、全く同一の条件で訓練しても、<br>
・訓練開始から上限ステップ数200に初めて到達するまでの早さ（エピソード数）<br>
・上限ステップ数200に到達したエピソード回数<br>
に変動が毎回あり、<b>一概に報酬のクリッピングをし、Multi-step Learningのnを高くすれば効果がある、とは言えない</b>です。<br>
現時点の感覚では、上記2つの尺度に照らし合わせると、<br>
報酬のクリッピングをした方がやや良くなるものの、Multi-step Learningについてはn=2が一番良いがn=1（Single-step）と大差ないかな、と思っています。

<br><br>

### 訓練関数train()内の流れ
訓練開始からsteps_warm_upのステップを実行後、経験バッファに経験データを蓄積し始めます。<br>
経験バッファにbatch_size分の経験データが蓄積されてから、Main-QNの訓練を始めます。<BR><BR>
 ![train関数処理単位と時間軸3](https://user-images.githubusercontent.com/52105933/93423730-97742700-f8f1-11ea-9c78-90edb1793520.png)
<!--![train関数処理単位と時間軸2](https://user-images.githubusercontent.com/52105933/92565255-50fd4780-f2b5-11ea-8f4e-365bddd7119f.png)-->
<!--![train関数処理単位と時間軸](https://user-images.githubusercontent.com/52105933/92488826-01be0500-f22a-11ea-911f-7e3bcf635333.png)-->

<br><br>

## 実行確認環境と実行の方法

### 実行確認環境

以下の環境での実行を確認しました。<br>

- numpy 1.19.1
- gym 0.17.2

#### インストール

gymのインストール
```
pip install gym
```

### 実行の方法

訓練済モデルの使用、訓練と推論の具体的な方法は、ファイル「CartPole_demo.ipynb」を参照してください。<br>

<br>

## ディレクトリ構成
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
<br>

### class Planner　のpublicインターフェース一覧
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

<br><BR>

※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。