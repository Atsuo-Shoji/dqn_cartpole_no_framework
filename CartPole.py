#フレームワーク不使用　multi-step
import numpy as np
from collections import deque, OrderedDict
from common.tools import *

#モデル本体
#方策を決定する
class Planner:
    
    def __init__(self, name, env, state_dim=4, action_dim=2, exp_buffer_size=10000, huber_loss_delta=1.0):
        
        #name：このPlannerインスタンスの名前
        #env：このPlannerインスタンスが対象とする環境
        #state_dim：このPlannerインスタンスが認識する状態の要素数
        #action_dim：このPlannerインスタンスが認識する行動の要素数
        #exp_buffer_size：このPlannerインスタンスが内部で保持する経験バッファが蓄積できる経験データ上限数
        #huber_loss_delta：Huber損失の定数delta
        
        self._name = name
        self._env = env
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        self._main_dqn = Planner.Dqn(state_dim=state_dim, action_dim=action_dim, huber_loss_delta=huber_loss_delta)
        self._target_dqn = Planner.Dqn(state_dim=state_dim, action_dim=action_dim, huber_loss_delta=huber_loss_delta)
        #最初に両者のパラメーターを同一にしておく（ただし特にそうする必要は無い）
        self._copy_params_to(self._main_dqn, self._target_dqn)
        
        #train()を複数回実行してもちゃんとシームレスに訓練が継続するように、経験バッファはメンバー変数とする。
        self._exp_buffer = Planner.ExperienceBuffer(exp_buffer_size, state_dim=state_dim, action_dim=action_dim)
        
        #訓練通し番号。train()1回に付き1個振られる。経験バッファ中で経験データの識別子の1つとして使用される。
        self._consecutive_num_train = 0
        
        
    def train(self, 
             episodes, episodes_stop_success, episodes_main_params_copy=1, reward_type=0, 
             steps_per_episode=200, steps_warm_up=10, steps_success_over=160, steps_multi_step_lern=1, 
             gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_rate=0.001, batch_size=32, verbose=True):
        
        #episodes：試行するエピソード回数。
        #episodes_stop_success：何回エピソードが成功すれば訓練を終了するか。
        #episodes_main_params_copy：エピソード何回毎に、main_dqnのパラメーターをtarget_dqnにコピーするか。
        #reward_type：報酬のあげ方。
        #steps_per_episode：1エピソードで何ステップまで行うか。
        #steps_warm_up：訓練開始後、経験バッファに経験データを格納しない待機ステップ数。
        #steps_success_over：何ステップを超えたら、そのエピソードを成功とみなすか。
        #steps_multi_step_lern：Multi-step Learningのn。
        #gamma：教師信号算出要素のγ。報酬やQ値の現在価値算出のための割引率。
        #epsilon_start：訓練開始直後のε-greedyのε。
        #epsilon_end：ε-greedyのεの下限値。
        #epsilon_decay_rate：ε-greedyのεの減衰率。
        #batch_size：main：main_dqnのパラメーター更新の際のバッチサイズ。
        #verbose：訓練途中で途中経過の文字列を出力するか。

        
        #訓練通し番号の採番
        self._consecutive_num_train += 1
        
        #エピソードループでの記録媒体
        loss_episodes = []
        step_count_episodes = []
        success_episodes = []
        reward_episodes = []
        
        #エピソードループ
        episode_count_success = 0
        total_step_count = 0
        episode_count = 0
        best_step_count = 0 #今までのepisodeの中で最も多いstep回数
        for ep in range(episodes):
            
            if verbose==True:
                print("\nEpisode:", ep)
            
            #loss_ep = None 各エピソード終了時点でのmain_dqnのloss。はじめは訓練に至らないエピソードもあるので、Noneにすべきだが、
            #未訓練エピソード分全部Noneだとグラフ表示で横軸がNoneの分だけ無くなるので、グラフの横軸左端（エピソード=0）だけ、何か数字を入れる。
            if ep==0:
                loss_ep = 2
            else:
                loss_ep = None
            success_ep = False #各エピソードの成功/失敗            
            
            #エピソード初端の現状態
            #Stateは、単独であっても(N, self._state_dim)というshapeに統一する。
            curr_state = self._env.reset().reshape(1, self._state_dim)    
            
            #ステップループ
            this_episode_shoudle_be_end = False
            step_count_ep = 0
            for st in range(steps_per_episode):
                
                #print("Episode:" + str(ep) + " Step:" + str(st) + " Total Step Count:" + str(total_step_count))
                
                #εの逓減
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp( -epsilon_decay_rate*(total_step_count+1) )
                
                ###経験バッファの蓄積###
                
                #actionの決定　ε-greedy
                if np.random.rand() < epsilon:
                    #Explore：
                    action = self._env.action_space.sample()
                else:
                    #Exploit:
                    action = self._main_dqn.predict_best_action(curr_state)
                
                ##指示した行動に基づくAgent（CartPole）の遷移と即時報酬の獲得##
                
                #環境envにactionを指示、Agent（CartPole）が次のStateへ遷移する。
                #※envから返してくる即時報酬「r」は、本タスクでは使用せず、別の独自定義の即時報酬とする。
                next_state, r, done, info = self._env.step(action)
                #Stateは、単独であっても(N, self._state_dim)というshapeに統一する。
                next_state = next_state.reshape(1, self._state_dim) 
                
                ##このステップでのこのエピソード終了判定
                #以下のor
                #・このエピソードでのステップ数がsteps_per_episodeに達した。
                #・環境envがdone=Trueを返してきた（倒れた）。
                if (st+1)>=steps_per_episode or done==True:
                    this_episode_shoudle_be_end = True
                
                #即時報酬の獲得と経験バッファへの追加
                #以下、ラストのステップかどうかでやることが変わる
                
                if this_episode_shoudle_be_end==False:
                    
                    #ステップがまだ続く場合
                    #即時報酬rewardは0
                    reward = 0
                    
                    #経験バッファに経験タプルを追加
                    if (st+1)>steps_warm_up:
                        #タプル(エピソード終端信号、遷移元状態、取った行動、即時報酬、遷移先状態, 
                        #訓練通し番号, エピソード、ステップ、総ステップ数)
                        self._exp_buffer.add( (this_episode_shoudle_be_end, curr_state, action, reward, next_state, 
                                                self._consecutive_num_train, ep, st, total_step_count) )
                    
                    #注意！！経験バッファに経験データをadd↑する前に、curr_state = next_state　とやらぬように。
                    curr_state = next_state
                    
                else:
                    #this_episode_shoudle_be_end==True
                    #このステップでこのエピソードは終わり。
                    #以下のor
                    #・このエピソードでのステップ数がsteps_per_episodeに達した。
                    #・環境envがdone=Trueを返してきた（倒れた）。
                    
                    #このエピソードの成功/失敗の判定
                    #即時報酬rewardの決定
                    if (st+1)>steps_success_over:
                        #このエピソードは”成功”で終えた
                        success_ep = True
                        if reward_type==0:
                            #成功時、即時報酬rewardは、step数に応じたものにゲタ(1.0)を履かせる。
                            #失敗時は、報酬は0。
                            reward = 1.0 + (st+1)/steps_success_over
                        else:
                            #成功時、即時報酬rewardは、step数に応じたものにゲタ(2.0)を履かせる。
                            #失敗時は、報酬はstep数に応じたものを与える。
                            reward = 2.0 + (st+1)/steps_success_over
                        episode_count_success += 1
                    else:
                        #このエピソードは”失敗”で終えた
                        success_ep = False
                        if reward_type==0:
                            #成功時、即時報酬rewardは、step数に応じたものにゲタ(1.0)を履かせる。
                            #失敗時は、報酬は0。
                            reward = 0
                        else:
                            #成功時、即時報酬rewardは、step数に応じたものにゲタ(2.0)を履かせる。
                            #失敗時は、報酬はstep数に応じたものを与える。
                            reward = (st+1)/steps_success_over
                    
                    #next_stateは、「無し」を意味するNoneにする。
                    next_state = None
                
                    #経験バッファに経験タプルを追加
                    if (st+1)>steps_warm_up:
                        #warm_upしたら、経験バッファに経験タプルを追加
                        #タプル(エピソード終端信号、遷移元状態、取った行動、即時報酬、遷移先状態, 
                        #訓練通し番号, エピソード、ステップ、総ステップ数)
                        self._exp_buffer.add( (this_episode_shoudle_be_end, curr_state, action, reward, next_state, 
                                                self._consecutive_num_train, ep, st, total_step_count) )
                
                ##素地した行動に基づくAgent（CartPole）の遷移と即時報酬の獲得　終わり##
                
                ###経験バッファの蓄積　終わり###
                
                ###経験バッファに基づき、main_dqnを訓練###
                
                if self._exp_buffer.count>=batch_size:
                    #経験バッファの経験保持数がバッチサイズに達すると、main_dqn訓練開始。
                    
                    #target_dqnとmain_dqnの入力と教師データのndarrayの空箱
                    #入力はStateのbatch_size分のndarray　(batch_size, self._state_dim)
                    inputs_states = np.zeros( (batch_size, self._state_dim) )
                    #教師データ　(batch_size, self._action_dim)
                    targets = np.zeros( (batch_size, self._action_dim) )
                    
                    #multi step learning
                    #経験データ(tuple)の時系列1個分のlistをミニバッチ分含んだlistを抽出（listの入れ子）
                    minibatch_list_list_a_series_of_exps_multi_steps = self._exp_buffer.extract_randomly(
                            num_experience=batch_size, steps_multi_step_lern=steps_multi_step_lern)
                    #※復元ランダム抽出なので、minibatch_list_list_a_series_of_exps_multi_stepsのlenはbatch_size
                                        
                    #ミニバッチ内1経験タプルをもとに1訓練データ（入力State）と1教師データ（Q値）を生成するループ
                    #ループ終了時には、batch_size分の教練データ（入力State）ができている　(len_minibatch, self._sate_dim)
                    #ループ終了時には、batch_size分の教師データ（Q値）ができている　(len_minibatch, self._action_dim)
                    for idx_of_list_a_series_of_exps_in_minibatch in range(batch_size):
                        
                        ##実際に取ったactionのみに対しての教師データを計算##
                        
                        #1個の時系列経験データ(list)を取得
                        #基本的には、このlistの中に、steps_multi_step_lern個分の経験データ（tuple）が時系列に入っている。
                        list_a_series_of_exps_multi_steps = minibatch_list_list_a_series_of_exps_multi_steps[idx_of_list_a_series_of_exps_in_minibatch]
                        
                        #この時系列経験データ(list)のうち、最初の経験データ（tuple）は時間軸上の基点となる物で、baseと呼ぶ。
                        #baseの経験データ（tuple）を取得。
                        #そのcurr_stateとactionを後で教師データ作成に使用するため。
                        exp_base_tpl = list_a_series_of_exps_multi_steps[0]
                        action_exp_base = exp_base_tpl[2] #実際に取った行動actionは経験タプルの2番目
                        curr_state_exp_base = exp_base_tpl[1] #遷移元stateは経験タプルの1番目
                        #タプル(エピソード終端信号、遷移元状態、取った行動、即時報酬、遷移先状態, 
                        #訓練通し番号, エピソード、ステップ、総ステップ数)
                        
                        #訓練データ（入力State）
                        inputs_states[idx_of_list_a_series_of_exps_in_minibatch] = curr_state_exp_base                  
                        
                        target_for_action_taken = self._calc_target_of_a_series_of_exps_multi_steps(
                                list_a_series_of_exps_multi_steps=list_a_series_of_exps_multi_steps, 
                                steps_multi_step_lern=steps_multi_step_lern, gamma=gamma)
                            
                        ##実際に取ったactionのみに対しての教師データを計算　終わり##
                        
                        ##最終的な教師データを生成##
                        #全actionについてのQ値をmain_dqnから出力する。「実際に取らなかったactionについての教師データ」にする。
                        targets[idx_of_list_a_series_of_exps_in_minibatch] = self._main_dqn.predict_Q(curr_state_exp_base)
                        #このうち、実際に取ったactionに対しては、算出した「実際に取ったactionに対しての教師データ」で上書き。
                        targets[idx_of_list_a_series_of_exps_in_minibatch][action_exp_base] = target_for_action_taken
                        #これで、教師データ1件は、以下のようになった
                        #targets[idx_of_list_a_series_of_exps_in_minibatch][実際に取ったaction]
                        #   = 実際に取ったactionに対しての教師データ
                        #targets[idx_of_list_a_series_of_exps_in_minibatch][実際に取らなかったaction]
                        #   = main_dqnの出力Q値　（誤差が無い）
                        #以降、これをmain_dqnに教師データとして投入し、普通のDNNの訓練をする。
                        
                    #ミニバッチ内1経験タプルをもとに1訓練データ（入力State）と1教師データ（Q値）を生成するループ　終わり
                    
                    #main_dqnの訓練。
                    #これ1回で「1イテレーション」と表現している。main_dqn内の「イテレーション」とは異なることに注意。
                    #訓練データはinputs_states
                    #教師データはtargets
                    loss_itr = self._main_dqn.fit(states=inputs_states, targets=targets, epochs=1,
                                                     batch_size=batch_size, weight_decay_lmd=0)
                    
                    #「エピソード終了時点でのmain_dqnの訓練状態」をエピソード終了時点で記録する。
                    #ただし、エピソード終了時点であらためて計測する、ということはしない。
                    #毎イテレーション（ステップ）終了時のlossを単一に保持しておく。
                    loss_ep = loss_itr
                
                ###経験バッファに基づき、main_dqnを訓練　終わり###
                
                #ステップの最後にインクリメント
                step_count_ep+=1
                total_step_count+= 1
                
                if this_episode_shoudle_be_end==True:
                    #このエピソードはこのステップで終了                    
                    break
                    
            #ステップのforループ　終わり
            
            #このエピソードのいろんな記録
            step_count_episodes.append(step_count_ep) #ステップ数
            loss_episodes.append(loss_ep) #main_dqnのloss
            reward_episodes.append(reward) #reward
            if success_ep==True: #成功/失敗
                success_episodes.append(1) #Trueでもいいが、グラフ表示の時困るだろう。
            else:
                success_episodes.append(0) #Falseでもいいが、グラフ表示の時困るだろう。
                        
            if step_count_ep>=best_step_count:
                #現在のmain_dqnのパラメータをbestなものとして、一時退避させる。
                #step_count_ep>=best_step_countの「>=」の「=」：同じ成績でも、後の方がより訓練されているものと推測
                
                if verbose==True:
                    print("最多ステップ数　訓練済パラメーター一時退避　step_count_ep:" + str(step_count_ep) + " best_step_count:" + str(best_step_count))
                self._main_dqn.keep_temporarily_all_learnable_params()
                
                best_step_count = step_count_ep
                        
            if verbose==True:
                print(" Step Count:", step_count_ep)                    
            
            #mainのパラメーターをtargetのパラメーターにコピー
            if (ep+1)%episodes_main_params_copy==0 or ep==(episodes-1):
                #指定エピソード回数おきに、又は最終エピソードで、コピー
                self._copy_params_to(self._main_dqn, self._target_dqn)                
            
            if episode_count_success >= episodes_stop_success:
                #成功エピソード数が指定のエピソード回数に達した。
                #エピソード反復を中止し、訓練を終了する。
                #既定成功エピソード数に達し、最終エピソードとなったので、targetにmainのパラメーターをコピー
                self._copy_params_to(self._main_dqn, self._target_dqn) 
                episode_count+=1
                break
                
            episode_count+=1
                
        #エピソードのforループ　終わり
        
        #現在のmain_dqnの一時退避済パラメータを正式採用する。
        if verbose==True:
            print("\n全エピソード終了　ベストステップ数更新時に一時退避した訓練済パラメーターを正式採用")
        self._main_dqn.adopt_all_learnable_params_kept_temporarily()                
                
        result = {}
        result["name"] = self._name
        result["consecutive_num_train"] = self._consecutive_num_train #訓練通し番号
        result["episode_count"] = episode_count #実際のエピソード数
        result["step_count_episodes"] = step_count_episodes #1エピソードでのステップ数　履歴
        result["loss_episodes"] = loss_episodes #1エピソードでのmain_dqnのloss　履歴
        result["success_episodes"] = success_episodes #1エピソードでの成功（1）/失敗（0）　履歴
        result["reward_episodes"] = reward_episodes #1エピソードでのreward　履歴
        result["total_step_count"] = total_step_count #実際のステップ総数
        result["episode_count_success"] = episode_count_success #実際の成功エピソード数
        #以下は主に引数
        result["episodes"] = episodes
        result["reward_type"] = reward_type
        result["steps_multi_step_lern"] = steps_multi_step_lern
        result["steps_per_episode"] = steps_per_episode
        result["steps_warm_up"] = steps_warm_up
        result["gamma"] = gamma
        result["epsilon_start"] = epsilon_start
        result["epsilon_end"] = epsilon_end
        result["epsilon_decay_rate"] = epsilon_decay_rate
        result["batch_size"] = batch_size
        result["episodes_main_params_copy"] = episodes_main_params_copy
        result["steps_success_over"] = steps_success_over
        result["episodes_stop_success"] = episodes_stop_success
        
        return result      
    
    def _calc_target_of_a_series_of_exps_multi_steps(self, list_a_series_of_exps_multi_steps, steps_multi_step_lern, gamma):
        #multi step learning対応　main_dqnに充当する教師データの算出。
        #Σ_k=1..n(γ^(k-1)*r_t+k) + γ^n*max_a(Q(S_t+n, a))
        #list_a_series_of_exps_multi_steps：1個の時系列経験データ（tuple）のlist1個
        #　例：multi step learningでn=3の場合、 3個の経験データ（tuple）が時系列で並んでいるlist1個
        #　　※list内の経験データが必ずしも3個あるとは限らない。エピソード終端に達していた場合、2個目や1個目で終わっている。
        
        reward_discounted_sum = 0
        Q_target_max = 0
        Q_target_max_discounted = 0
        
        len_list_a_series_of_exps_multi_steps = len(list_a_series_of_exps_multi_steps)
        
        #t_diff_from_base：時系列series（multi step）には、時間軸上の基点となる経験データ（tuple）がある。list中のそのindex。
        #t_diff_from_base=0　の経験データは、時間軸上の基点となる（「base」）経験データ（tuple）である。
        #（single step learningの場合に使用される経験データである。n=1。）
        for t_diff_from_base in range( steps_multi_step_lern ):
            
            #print(" t_diff_from_base:", t_diff_from_base)
        
            if t_diff_from_base>( len_list_a_series_of_exps_multi_steps-1 ):
                #このbaseを基点とした時系列series（multi step）は、steps_multi_step_lern個分の経験データ（tuple）を満たす前に、
                #エピソードの終端に達してしまった。又は経験バッファから抽出時、経験バッファの終端に達してしまった。
                #list_a_series_of_exps_multi_steps　には、steps_multi_step_lern未満の個数の経験データ（tuple）しか無い。
                break
            
            #経験データ（tuple）を取得する。
            #タプル(エピソード終端信号、遷移元状態、取った行動、即時報酬、遷移先状態, 訓練通し番号, エピソード、ステップ、総ステップ数
            exp_tpl = list_a_series_of_exps_multi_steps[t_diff_from_base] 
            end_of_episode_exp = exp_tpl[0] #このstepでエピソード終端となったか
            reward_tpl = exp_tpl[3] #このstepで獲得したreward

            #maxQ(S_t+n, a)の算出
            #t_diff_from_base=steps_multi_step_lern-1　である経験データのみが対象。
            #　例：multi step learningでn=3の場合、 curre_stateがS_t+2の経験データのみが対象。S_t+3はnext_state。
            #ただし、たまたまエピソードの最後でもある場合、S_t+nは無いので、その場合はmaxQ(S_t+n, a)は算出しない。        
            if t_diff_from_base==steps_multi_step_lern-1:

                if end_of_episode_exp==False:
                    #エピソードの最後ではない。S_t+nは有るので、maxQ(S_t+n, a)を算出する。
                    #S_t+n
                    next_state_exp = exp_tpl[4]
                    #maxQ(S_t+n, a)
                    Q_target_max = self._target_dqn.predict_maxQ(next_state_exp)
                    ##この時系列経験データ（tuple）のlist全体の、maxQ(S_t+n, a)（γでdiscounted）
                    Q_target_max_discounted =  (gamma**(t_diff_from_base+1))*Q_target_max
                else:
                    #たまたまエピソードの最後でもあった。S_t+nは無いので、maxQ(S_t+n, a)は算出しない。
                    Q_target_max_discounted = 0
            else:
                #そもそも、maxQ(S_t+n, a)の算出対象では無い。
                Q_target_max_discounted = 0

            
            #rewardにγのt_diff_from_base乗を掛ける
            reward_discounted = (gamma**t_diff_from_base)*reward_tpl
            #この時系列経験データ（tuple）のlist全体の、reward（γでdiscounted）の合計
            reward_discounted_sum += reward_discounted
            
        #reward（γでdiscounted）の合計と、maxQ(S_t+n, a)（γでdiscounted）を足したのが、
        target = reward_discounted_sum + Q_target_max_discounted
        
        return target
    
    def predict_best_action(self, a_state):
        #与えられたstate1個でのbestなactionを推測して返す。
        
        best_action = self._main_dqn.predict_best_action(a_state)
        
        return best_action            
    
    def _copy_params_to(self, from_dqn, to_dqn):
        
        #fromのdqnから全パラメーターをコピーして取得
        all_params = from_dqn.copy_all_params()
        #それをtoのdqnに上書き
        to_dqn.overwrite_all_params(all_params)
        
    def save_params_in_file(self, file_dir, file_name=""):
        #このモデルインスタンスの全learnableパラメーターをファイル保存する。
        
        if file_name=="":
            file_name = self._name + ".pickle"
        
        file_path = file_dir + file_name
        
        self._main_dqn.save_all_params_in_file(file_path)
        
        return file_name
        
    def overwrite_params_in_file(self, file_path):
        #このモデルインスタンスの全パラメーターを、ファイル保存されている別の物に差し替える。
        
        self._main_dqn.overwrite_all_params_in_file(file_path)        
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def env(self):
        return self._env
    
    @property
    def state_dim(self):
        return self._state_dim
    
    @property
    def action_dim(self):
        return self._action_dim
    
    @property
    def count_experiences_in_buffer(self):
        #経験バッファの経験データ保持数
        return self._exp_buffer.count
    
    @property
    def huber_loss_delta(self):
        #mainのhuber_loss_deltaを返す。
        return self._main_dqn.huber_loss_delta
    
    #モデル内部のニューラルネットワーク
    #class Plannerの内部クラス
    class Dqn:

        def __init__(self, state_dim, action_dim, huber_loss_delta):

            #state_dim：stateの要素数
            #action_dim：actionの種類数

            self._state_dim = state_dim
            self._action_dim = action_dim

            ###Layerの定義###

            self._layers = OrderedDict()

            #①Affine 「afn1」　instance of Affine
            opt_afn1 = Adam(lr=0.001, rho1=0.9, rho2=0.999)
            afn1 = Affine(name="afn1", input_shape=(state_dim,), output_shape=(16,), optimizer=opt_afn1, 
                          init_weight_option="He")
            self._layers[afn1.name] = afn1
            prev_layer = afn1

            #②ReLU　「ReLU_afn1」　instance of ReLU
            relu_afn1 = ReLU(name="ReLU_afn1", input_shape=prev_layer.output_shape)
            self._layers[relu_afn1.name] = relu_afn1
            prev_layer = relu_afn1

            #③Affine 「afn2」　instance of Affine
            opt_afn2 = Adam(lr=0.001, rho1=0.9, rho2=0.999)
            afn2 = Affine(name="afn2", input_shape=prev_layer.output_shape, output_shape=(16,), optimizer=opt_afn2, 
                          init_weight_option="He")
            self._layers[afn2.name] = afn2
            prev_layer = afn2

            #④ReLU　「ReLU_afn2」　instance of ReLU
            relu_afn2 = ReLU(name="ReLU_afn2", input_shape=prev_layer.output_shape)
            self._layers[relu_afn2.name] = relu_afn2
            prev_layer = relu_afn2

            #④Affine 「afn3」　instance of Affine
            opt_afn3 = Adam(lr=0.001, rho1=0.9, rho2=0.999)
            afn3 = Affine(name="afn3", input_shape=prev_layer.output_shape, output_shape=(16,), optimizer=opt_afn3, 
                          init_weight_option="He")
            self._layers[afn3.name] = afn3
            prev_layer = afn3

            #⑤ReLU　「ReLU_afn3」　instance of ReLU
            relu_afn3 = ReLU(name="ReLU_afn3", input_shape=prev_layer.output_shape)
            self._layers[relu_afn3.name] = relu_afn3
            prev_layer = relu_afn3

            #⑥Affine 「afn4」　instance of Affine　順伝播のpredict出力
            opt_afn4 = Adam(lr=0.001, rho1=0.9, rho2=0.999)
            afn4 = Affine(name="afn4", input_shape=prev_layer.output_shape, output_shape=(self._action_dim,), optimizer=opt_afn4, 
                          init_weight_option="xavier")
            self._layers[afn4.name] = afn4
            prev_layer = afn4

            #＜最終loss＞　instance of HuberLoss　loss算出
            self._last_loss_layer = HuberLoss(name="last_loss", input_shape=prev_layer.output_shape, delta=huber_loss_delta) 
            self._layers[self._last_loss_layer.name] = self._last_loss_layer
            

        def copy_all_params(self):

            #Dictionaryにして返す。keyはlayer.name。
            #last_loss_layerにも対応することにした。last_loss_layerのパラメーターはlearnableではないし、last_loss_layer自体trainableではない。
            #all_learnable_paramsなんて名前が、混乱を招くものになってしまった。
            #all_learnable_params(Dictionary)
            # --learnable layer1の全learnableなパラメーターのtuple(weightsのndarray, biasesのndarray)
            # --learnable layer2の全learnableなパラメーターのtuple(weightsのndarray, biasesのndarray)
            #　・
            #　・
            # --last_loss_layerの全パラメーターのtuple(HuberLossだけなのでdelta)

            all_learnable_params_dic = {}
            
            #trainableな全layerのlearnableなパラメーター分
            for layer in self._layers.values():
                if layer.trainable == True:
                    learnable_params_tpl = layer.copy_learnable_params()
                    all_learnable_params_dic[layer.name] = learnable_params_tpl
            
            #last_loss_layerのパラメーター分
            #現時点ではHuberLossクラスのみであるが、それなりに拡張性を考慮する。
            all_learnable_params_dic[self._last_loss_layer.name] = self._last_loss_layer.copy_params()

            return all_learnable_params_dic        

        def overwrite_all_params(self, all_params_dic):

            for layer_name in all_params_dic.keys():

                #上書きするパラメーターをLayer毎に取り出す
                layer_params_tpl = all_params_dic[layer_name] 

                #コピー先Layer毎に上書きする。
                #名前が同じLayerがコピー先Layer。
                to_layer = self._layers[layer_name]
                
                if to_layer is None:
                    #ありえない。異常事態。
                    err_msg = "コピー先に対象となるLayerがありません。" + layer_name
                    raise ValueError(err_msg)
                
                if to_layer.trainable==True:
                    #ltrainableなLayer　上書き
                    to_layer.overwrite_learnable_params(layer_params_tpl)
                    
                if to_layer.last_loss_layer==True:
                    #last_loss_layer
                    #現時点ではHuberLossクラスのみであるが、それなりに拡張性を考慮する。
                    to_layer.overwrite_params(layer_params_tpl)                   


        def fit(self, states, targets, epochs=1, batch_size=32, weight_decay_lmd=0):

            #states : 訓練データの入力側。状態State。
            #targets : 訓練データの出力側教師データ。   

            num_states = states.shape[0]
            iters_per_epoch = np.ceil(num_states / batch_size).astype(np.int) #1エポックでのイテレーション数        

            #エポックのfor
            for epoch in range(epochs):  # e:1エポック 

                #print("\nEpoch:", epoch)            

                #イテレーション毎のミニバッチ抽出のための、全train_xのインデックスのシャッフル
                idxes_all = np.arange(num_states)
                np.random.shuffle(idxes_all)

                #イテレーションのfor
                for it in range(iters_per_epoch):

                    mask = idxes_all[batch_size*it : batch_size*(it+1)]

                    #ミニバッチの生成
                    x_ = states[mask]
                    t_ = targets[mask]

                    #推論ではなくパラメーター更新のための順伝播。
                    #パラメーター更新のための誤差逆伝播が控えているので、lossまで計算。
                    #中でprivateの_predict_Qs()を呼んでいる。
                    _ = self._loss(x_, t_, weight_decay_lmd, train_flg=True, batch_size=batch_size) 

                    #逆伝播
                    dout = 1 #誤差逆伝播のスタートでの勾配は必ず1
                    layers = list(self._layers.values())
                    layers.reverse()
                    for layer in layers:
                        if layer.last_loss_layer==True:
                            #forループ1回目は、self._layersの一番後ろのこのlast_loss_layerのはず。
                            dout = self._last_loss_layer.backward(dout) 
                        else:
                            dout = layer.backward(dout)

                    # 学習パラメーターの更新
                    self._update_all_learnable_params(weight_decay_lmd)

                #イテレーションのfor　終わり

            #エポックのfor　終わり

            #一旦全データで順伝播してlossを計測する。計測値の記録のため。              
            final_loss = self._loss(states, targets, weight_decay_lmd, train_flg=False, batch_size=batch_size)

            return final_loss    

        def _loss(self, states, targets, weight_decay_lmd, train_flg, batch_size):

            #損失関数
            #states : 入力データ。状態State。
            #targets : 教師データ。Q値。   

            #順伝播時は、必ず直接でも間接でもprivateのpredict_Qs()を呼ぶこと。
            Qs_predicted = self._predict_Qs(states=states, train_flg=train_flg, batch_size=batch_size)
            loss = self._last_loss_layer.forward_calc_loss(x=Qs_predicted, t=targets) 

            if weight_decay_lmd > 0:
                sum_all_weights_square = self._sum_all_weights_square()
                loss = loss + 0.5*weight_decay_lmd*sum_all_weights_square

            return loss

        def _predict_Qs(self, states, train_flg, batch_size):
            #クラス内部用のprivate関数。train()からミニバッチ順伝播で呼ばれる。
            #戻り値のshapeは必ず(stateの個数, self._state_dim)。stateの個数が1であってもこのshapeとする。

            if states.ndim==1:
                states = states.reshape(1, self._state_dim)            

            #順伝播予測の本体
            def __predict(x, train_flg): 

                for layer in self._layers.values():
                    if layer.last_loss_layer==True:
                        #順伝播予測ではlossの計算はしない。
                        #最終のlossのlayerに到達したが、もう計算しないのでbreak
                        break
                    else:
                        x = layer.forward(x, train_flg)            
                return x
            #順伝播予測の本体　終わり

            #多量のデータを渡された場合の実行環境負担軽減　ミニバッチに分ける

            calc_num = np.ceil(states.shape[0] / batch_size).astype(np.int) #計算回数

            pred_list = []

            for i in range(calc_num):
                tx = states[i*batch_size:(i+1)*batch_size]
                ty = __predict(tx, train_flg)
                pred_list.append(ty)

            Qs = np.concatenate(pred_list, 0)
            
            return Qs

        def predict_Q(self, a_state):
            #与えられたstate1個でのQ値を推測して返す。shapeはベクトル(self._action_dim, )。

            #順伝播時は、必ず直接でも間接でもprivateのpredict_Qs()を呼ぶこと。
            Q = self._predict_Qs(a_state, train_flg=False, batch_size=256)[0] 
            #[0]は、shapeを(1, self._action_dim)ではなく(self._action_dim, )にするため

            return Q

        def predict_maxQ(self, a_state):
            #与えられたstate1個で推測されたQ値1個の中の（action_dim個あるうちの）、最大数値を返す。
            #shapeはスカラー。

            #順伝播時は、必ず直接でも間接でもprivateの_predict_Qs()を呼ぶこと。
            Q = self.predict_Q(a_state) #shapeは(self._action_dim, )
            maxQ = np.amax(Q)

            return maxQ

        def predict_best_action(self, a_state):
            #与えられたstate1個でのbestなactionを推測して返す。

            #順伝播時は、必ず直接でも間接でもprivateの_predict_Qs()を呼ぶこと。
            Q = self.predict_Q(a_state) #shapeは(self._action_dim, )
            best_action = np.argmax(Q) 

            return best_action
        
        def save_all_params_in_file(self, file_path):
            #全learnableパラメーターとlast_loss_layerのパラメーターをファイル保存する。
            
            param_layer_tpls_dic = self.copy_all_params()            
            save_pickle_file(param_layer_tpls_dic, file_path)

        def overwrite_all_params_in_file(self, file_path):
            #ファイル保存した全learnableパラメーターとlast_loss_layerのパラメーターを読み込んで、利用可能にする。
            
            param_layer_tpls_dic = read_pickle_file(file_path)
            self.overwrite_all_params(param_layer_tpls_dic)
        
        def _update_all_learnable_params(self, weight_decay_lmd):
            #trainableな全Layersのtrainableなパラメーターを一括更新する。

            for layer in self._layers.values():
                if layer.trainable == True:
                    layer.update_learnable_params(weight_decay_lmd)

        def keep_temporarily_all_learnable_params(self):

            #配下の各trainableなLayerに対し、現時点でのlearnableパラメーターの一時退避を指示
            for layer in self._layers.values():
                if layer.trainable == True:
                    layer.keep_temporarily_learnable_params()

        def adopt_all_learnable_params_kept_temporarily(self):

            #配下の各trainableなLayerに対し、一時退避していたlearnableパラメーターの正式採用を指示
            for layer in self._layers.values():
                if layer.trainable == True:
                    layer.adopt_learnable_params_kept_temporarily()

        def _sum_all_weights_square(self):
            #weightを持つtrainableな全Layerのweightの2乗の総和を返す。
            #荷重減衰（weight decay）のため。

            sum_of_weights_square = 0
            for layer in self._layers.values():
                if layer.trainable == True and isinstance(layer, Affine):
                    sum_of_weights_square += layer.sum_weights_square()

            return sum_of_weights_square 

        @property
        def huber_loss_delta(self):
            #last_loss_layerはHuberLossクラスと限定
            if  isinstance(self._last_loss_layer, HuberLoss):
                return self._last_loss_layer.delta
            else:
                return None
        
    #モデル内部の経験バッファ
    #class Plannerの内部クラス
    class ExperienceBuffer:

        def __init__(self, size, state_dim, action_dim):

            #state_dim：stateの要素数
            #action_dim：actionの種類数

            self._state_dim = state_dim
            self._action_dim = action_dim

            #https://docs.python.org/ja/3/library/collections.html#collections.deque
            self._exp_buffer = deque(maxlen=size)

        def add(self, experience):
            #経験を追加
            #experienceは、タプル(遷移元状態、取った行動、即時報酬、遷移先状態)
            self._exp_buffer.append(experience)

        def extract_randomly(self, num_experience, steps_multi_step_lern):
            #経験データをnum_experience分、randomにlistにして返す。
            #steps_multi_step_lern：multi step learningの未来方向のstep数。
            #multi step learningに対応している

            #戻り値のlistは入れ子になっている。以下はsteps_multi_step_lern=3の例。
            #戻り値のlist「list_exps」
            # --経験データ1個の時系列グループのlist　･････････　1個目
            #   --baseとなる経験データ（tuple）　　　　 baseのstep（=tとおく）：　State_t、reward_t+1、State_t+1　が入っている
            #   --baseより1個未来の経験データ（tuple）　step=t+1：　State_t+1、reward_t+2、State_t+2　が入っている
            #   --baseより2個未来の経験データ（tuple）　step=t+2：　State_t+2、reward_t+3、State_t+3　が入っている
            # --経験データ1個の時系列グループのlist　･････････　2個目
            #   --baseとなる経験データ（tuple）　　　　 baseのstep（=tとおく）：　State_t、reward_t+1、State_t+1　が入っている
            #   --baseより1個未来の経験データ（tuple）　step=t+1：　State_t+1、reward_t+2、State_t+2　が入っている
            #   --（無し。↑でエピソード終端に達した。）
            # --経験データ1個の時系列グループのlist　･････････　3個目
            #   --baseとなる経験データ（tuple）　　　　 baseのstep（=tとおく）：　State_t、reward_t+1、State_t+1　が入っている
            #   --（無し。↑でエピソード終端に達した。）
            #　・
            #　・
            # --経験データ1個の時系列グループのlist　･････････　num_experience個目
            #   --baseとなる経験データ（tuple）　　　　 baseのstep（=tとおく）：　State_t、reward_t+1、State_t+1　が入っている
            #   --baseより1個未来の経験データ（tuple）　step=t+1：　State_t+1、reward_t+2、State_t+2　が入っている
            #   --baseより2個未来の経験データ（tuple）　step=t+2：　State_t+2、reward_t+3、State_t+3　が入っている
            #
            #steps_multi_step_lern=1はsingle step learningで、multi step learningをしない場合はこれ。未来の経験データ（tuple）不要。
            #（steps_multi_step_lern-1）個の未来の経験データ（tuple）をさらに取得するのがmulti step learning。


            idxes_all = np.arange(len(self._exp_buffer))
            idxes_bases = np.random.choice(idxes_all, size=num_experience, replace=False)

            idx_last_buffer = len(self._exp_buffer) - 1
            
            list_exps = []      
            for idx_base in idxes_bases:

                ###baseの経験データ(tuple)1個と、その未来の経験データ(tuple)群一式を取得###

                list_a_series_of_exps_multi_steps = [] #baseの経験データ(tuple)1個と、その未来の経験データ(tuple)群一式　のlist

                ##baseとなる経験データ（tuple）の取得##

                #baseの経験データの取得
                exp_base_tpl =  self._exp_buffer[idx_base]
                
                #baseの経験データ（tuple）をlistに追加
                list_a_series_of_exps_multi_steps.append(exp_base_tpl)

                #baseの経験データ（tuple）の属性のうち、ここで使用する属性を取得
                #タプル(エピソード終端信号、遷移元状態、取った行動、即時報酬、遷移先状態, 訓練通し番号, エピソード、ステップ、総ステップ数)
                end_of_episode_exp_base = exp_base_tpl[0]
                consecutive_num_train_exp_base = exp_base_tpl[5] #baseが所属する訓練の訓練通し番号
                episode_exp_base = exp_base_tpl[6] #baseが所属する訓練に所属するエピソード

                ##baseとなる経験データ（tuple）の取得　終わり##

                ##未来方向に経験データ（tuple）を必要な分取得するforループ##
                #ただし、episode終端に達したらor経験バッファの終端に達したら、その次の経験データ（tuple）は取得しない。
                this_step_is_end_of_episode = end_of_episode_exp_base            

                for idx_diff_from_following_base in range(1, steps_multi_step_lern, 1): #range(start, stop1個後, step)
                    #range(1, steps_multi_step_lern-1, 1)のはじめの「1」：「0」は取得済のbase。だから「1」からはじまる。

                    prev_step_is_end_of_episode = this_step_is_end_of_episode

                    if prev_step_is_end_of_episode==True:
                        #1個前のstepの経験データ（tuple）はエピソード終端であった。
                        #経験データ（tuple）は取得しない。
                        break

                    #baseから「idx_diff_from_following_base」step分未来のidx
                    idx_following_base = idx_base + idx_diff_from_following_base

                    if idx_following_base>idx_last_buffer:
                        #経験バッファの終端に達した。
                        #経験データ（tuple）は取得しない。
                        break

                    #baseから「idx_diff_from_following_base」step分未来の経験データ（tuple）を取得
                    exp_following_tpl = self._exp_buffer[idx_following_base]
                    
                    consecutive_num_train_exp_following = exp_following_tpl[5] #このexpが所属する訓練の訓練通し番号
                    episode_exp_following = exp_following_tpl[6] #このexpが所属する訓練に所属するエピソード
                    
                    #一応、baseと同一エピソードかチェック
                    if consecutive_num_train_exp_following!=consecutive_num_train_exp_base or episode_exp_following!=episode_exp_base:
                        #同一エピソードではない。異常事態。経験バッファがおかしい。
                        err_msg = "経験バッファに不正なデータがあります。別インスタンスを作成して、再度訓練してください。"
                        raise RuntimeError(err_msg)

                    #baseから「idx_diff_from_following_base」step分未来の経験データ（tuple）をlistに追加
                    list_a_series_of_exps_multi_steps.append(exp_following_tpl)

                    #エピソード終端なら、次のループを実行させない。
                    this_step_is_end_of_episode = exp_following_tpl[0] 

                ##未来方向に経験データ（tuple）を必要な分取得するforループ　終わり
                ###baseの経験データ(tuple)1個と、その未来の経験データ(tuple)群一式を取得　終わり###

                #list_a_series_of_exps_multi_steps　が、1個のbase分一式、完成している。
                #これを、戻り値のlist_expsにappendする。
                list_exps.append(list_a_series_of_exps_multi_steps)

            return list_exps

        def extract_states_array_randomly(self, num_experience):
            #経験バッファ中のStateのみ、randomに抽出し、ndarrayにして返す。
            #dqnの性能計測用。

            idxes_all = np.arange(len(self._exp_buffer))
            idxes = np.random.choice(idxes_all, size=num_experience, replace=False)

            array_exps = np.zeros( (num_experience, self._state_dim) ) 
            for i in idxes:            
                exp =  self._exp_buffer[i]
                array_exps[i] = exp

            return array_exps

        @property
        def count(self):
            return len(self._exp_buffer)