# -*- coding: utf-8 -*-
import numpy as np
import copy
import pickle


###Layerの親抽象クラス###

#public abstract class Layer 
class Layer:
    
    def __init__(self, name):        
        raise TypeError("このクラスは抽象クラスです。インスタンスを作ることはできません。")
        
    def forward(self, x):
        raise NotImplementedError("forward()はオーバーライドしてください。")
        
    def backward(self, dout):
        raise NotImplementedError("backwardはオーバーライドしてください。")
        
    def update_learnable_params(self, weight_decay_lmd=0):
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、update_learnable_params()はオーバーライドしてください。")
        else:
            pass
        
    def copy_learnable_params(self):
        #訓練対象パラメーターをコピーして、tupleに詰め込んで返す。
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、copy_learnable_params()はオーバーライドしてください。")
        else:
            pass
        
    def overwrite_learnable_params(self, learnable_params_tpl):
        #新しい訓練対象パラメーターのtupleを受けとって上書きする。
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、overwrite_learnable_params()はオーバーライドしてください。")
        else:
            pass
        
    def keep_temporarily_learnable_params(self):
        
        #訓練対象パラメーターを一時退避する。
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、keep_tempolary_learnable_params()はオーバーライドしてください。")
        else:
            pass
        
    def adopt_learnable_params_kept_temporarily(self):
        
        #一時退避した訓練対象パラメーターを正式採用し、使用再開する。
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、adopt_learnable_params_kept_tempolary()はオーバーライドしてください。")
        else:
            pass
    
    @property
    def name(self):
        raise NotImplementedError("nameはオーバーライドしてください。")
    
    @property
    def trainable(self):
        return False
    
    @property
    def last_loss_layer(self):
        raise NotImplementedError("last_loss_layerはオーバーライドしてください。")
    
    @property
    def optimizer(self):
        if self.trainable==True:
            raise NotImplementedError("trainableなクラスでは、optimizerはオーバーライドしてください。")
        else:
            return None
        
    @property
    def input_shape(self):
        raise NotImplementedError("input_shapeはオーバーライドしてください。")
        
    @property
    def output_shape(self):
        raise NotImplementedError("output_shapeはオーバーライドしてください。")
        
###Layerの親抽象クラス　終わり###


###順伝播/逆伝播Layer###

#public class Affine extends Layer 
class Affine(Layer):
    #全結合層

    def __init__(self, name, input_shape, output_shape, optimizer, init_weight_option, default_init_weight_std=0.1):
        
        self._name = name
        
        #input_shape:入力データshape。tuple。
        self._input_shape = input_shape
        #output_shape:出力データshape。tuple。
        self._output_shape = output_shape
        
        input_size = input_shape[0]
        output_size = output_shape[0]
        
        init_std = calculate_init_std_weight(input_size, output_size, init_weight_option, default_init_weight_std)
        
        self._W = init_std * np.random.randn(input_size, output_size)
        self._b = np.zeros(output_size) #biasの初期値は0で埋めるのが普通。所詮は補正項でありあまり気にしない。
        
        self._x = None
        self._original_x_shape = None
        # 重み・バイアスパラメータの微分
        self._dW = None
        self._db = None
        
        #_tempは、複数エピソードを消費する訓練中で最高の性能を示したエピソードでのパラメーターを一時退避し、
        #後にそれを正式採用するための退避領域。
        #最高の性能を示さずに性能劣化のままt訓練を終えてもインスタンス化時（init時）の性能を保証するために、init時点で一時退避してしまう。
        self._W_temp = copy.deepcopy(self._W)
        self._b_temp = copy.deepcopy(self._b)
        
        #optimizer
        self._opt = optimizer            
        
    def forward(self, x, train_flg=False):
        #順伝播
        #ｘ：入力データ
        
        #xがどのようなshapeであっても対応できるように（例えば画像形式)
        self._original_x_shape = x.shape
        x = x.reshape(x.shape[0], *self._input_shape)
        self._x = x

        out = np.dot(self._x, self._W) + self._b

        return out

    def backward(self, dout):
    
        dx = np.dot(dout, self._W.T)
        self._dW = np.dot(self._x.T, dout)
        self._db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self._original_x_shape)  #入力データxのshapeに戻す
        return dx
        
    def update_learnable_params(self, weight_decay_lmd):
    
        params = {}
        params['weight'] = self._W
        params['bias'] = self._b
    
        grads = {}
        grads['weight'] = self._dW + weight_decay_lmd*self._W
        grads['bias'] = self._db
        
        self._opt.update(params, grads)
        
    def copy_learnable_params(self):
        #訓練対象パラメーターをコピーして、tupleに詰め込んで返す。
        
        copy_of_W = copy.deepcopy(self._W)
        copy_of_b = copy.deepcopy(self._b)
        
        copy_of_learable_params_tpl = (copy_of_W, copy_of_b)
        
        return copy_of_learable_params_tpl
    
    def overwrite_learnable_params(self, learnable_params_tpl):
        #新しい訓練対象パラメーターのtupleを受けとって上書きする。
        
        #weightsの上書き
        W = learnable_params_tpl[0]
        if W.shape!=self._W.shape:
            err_msg = "weightのshapeが不正です。" + str(self._name) + " 正しいshape：" + str(self._W.shape) + " 受け取ったshape：" + str(W.shape) 
            raise ValueError(err_msg)
        
        #biasesの上書き
        b = learnable_params_tpl[1]
        if b.shape!=self._b.shape:
            err_msg = "biasのshapeが不正です。" + str(self._name) + " 正しいshape：" + str(self._b.shape) + " 受け取ったshape：" + str(b.shape) 
            raise ValueError(err_msg)
            
        self._W = W
        self._b = b
        
    def keep_temporarily_learnable_params(self):
        
        #現時点でのweightとbiasを一時退避する。
        self._W_temp = copy.deepcopy(self._W)
        self._b_temp = copy.deepcopy(self._b)
        
    def adopt_learnable_params_kept_temporarily(self):
        
        #一時退避したweightとbiasを正式採用する。
        self._W = self._W_temp
        self._b = self._b_temp
        
    def sum_weights_square(self): 
        #荷重減衰（weight_decay）用
        return np.sum(self._W**2)    
        
    @property
    def name(self):
        return self._name
    
    @property
    def trainable(self):
        return True
    
    @property
    def last_loss_layer(self):
        return False
    
    @property
    def optimizer(self):
        return self._opt
    
    @property
    def input_shape(self):
        return self._input_shape
        
    @property
    def output_shape(self):
        return self._output_shape
    
###順伝播/逆伝播Layer　終わり###


###Activation Layer###

#public class ReLU extends Layer 
class ReLU(Layer):

    def __init__(self, name, input_shape):
        
        self._name = name
        self._mask_negative_on_x = None
        self._input_shape = input_shape

    def forward(self, x, train_flg=False):
        
        self._mask_negative_on_x = (x <= 0)
        out = x * np.where( self._mask_negative_on_x, 0, 1.0 )

        return out

    def backward(self, dout):
        
        dx = dout * np.where( self._mask_negative_on_x, 0, 1.0 )
        
        return dx    
    
    @property
    def name(self):
        return self._name
       
    @property
    def trainable(self):
        return False
    
    @property
    def last_loss_layer(self):
        return False
    
    @property
    def input_shape(self):
        return self._input_shape
        
    @property
    def output_shape(self):
        return self._input_shape
    
###Activation Layers　終わり###


###損失Layers###

#public class HuberLoss extends Layer 
class HuberLoss(Layer):
    #Huber損失
    #活性化関数+損失計算
    #実装上は「last_layer」。1個前のAffineには活性化関数は付けないようにする。
    #https://www.wikiwand.com/ja/Huber%E6%90%8D%E5%A4%B1
    #https://axa.biopapyrus.jp/machine-learning/model-evaluation/loss-function.html
    
    def __init__(self, name, input_shape, delta=1.0):
        
        self._name = name
        self._loss = None
        self._y = None #活性化関数出力値
        self._t = None #教師データ
        self._delta = delta
        self._input_shape = input_shape
    
    def forward(self, x, train_flg=False):
        return super().forward(x, train_flg)
    
    def forward_calc_loss(self, x, t):
        
        self._t = t
        self._y = x #活性化関数出力値。教科書的「出力層」の出力値。活性化関数は恒等関数。
        
        #ここまでが活性化関数
        #以降は活性化関数出力値に対する損失計算
                                
        #単純な差異
        diff = self._y - t
        
        #①差異の絶対値
        #誤差逆伝播時のために、mask方式を取る。正の数の要素のmaskを取る。
        self._mask_positive_on_diff = (diff >= 0)
        self._abs_diff = diff * np.where( self._mask_positive_on_diff, 1.0, -1.0 )
        
        #誤差逆伝播時のために、以降はabs_diffのみを変数とする。誤差逆伝播時、②は∂L/∂(abs_diff)を途中結果としたい。
        
        #②差分の絶対値とdeltaの大小関係による条件分岐
        sl = 0.5 * (self._abs_diff**2) #2乗和損失
        otherwise_l = (self._delta * self._abs_diff) - (0.5 * (self._delta**2)) #その他の損失
        
        #deltaより小さい数値の要素は2乗和損失sl
        #2乗和損失slが適用される要素のmaskを取る。
        self._mask_sl_on_abs_diff = (self._abs_diff < self._delta)
        #maskを適用し、ミニバッチ内の個々のデータのHuber損失を算出する。
        loss = np.where(self._mask_sl_on_abs_diff, sl, otherwise_l)
        
        #ミニバッチでの損失の平均を取る。
        #ミニバッチ内各データのHuber損失の総和を取ってバッチサイズで割る。
        batch_size = self._y.shape[0]
        self._loss = np.sum(loss) / batch_size
        
        return self._loss

    def backward(self, dout):
        
        #②'　∂L/∂(abs_diff)まで
        #順伝播時、2乗和損失slを適用された場合、abs_diffについての微分は、2*abs_diff*0.5=abs_diff
        #∂L/∂(abs_diff) = (2*abs_diff*0.5) * dout = abs_diff * dout
        #順伝播時、2乗和損失slではなくその他の損失otherwise_lを適用された場合、abs_diffについての微分はdelta
        #∂L/∂(abs_diff) = delta * dout
        d_abs_diff = dout * np.where(self._mask_sl_on_abs_diff, self._abs_diff, self._delta)
        #①'　∂L/∂(diff)
        #順伝播時、単純差異diffの絶対値を取った。diff>=0ならx1、diff<0ならx(-1)で逆伝播する。
        d_diff = d_abs_diff * np.where( self._mask_positive_on_diff, 1.0, -1.0 )
        
        #∂L/∂y
        #順伝播時は単純差異 self._y - t
        dy = d_diff * 1.0
        
        #∂L/∂x
        #活性化関数は恒等関数 self._y = x
        dx = dy * 1.0     
        
        batch_size = self._t.shape[0]
        dx = dx / batch_size
        
        return dx
    
    def copy_params(self):
        #パラメーターをコピーして、tupleに詰め込んで返す。
        #HuberLossクラスにおいては、deltaのみ。
        
        copy_of_delta = copy.copy(self._delta)
        copy_of_params_tpl = (copy_of_delta,) #最後の「,」が無いとtupleと認識されない。
        
        return copy_of_params_tpl
    
    def overwrite_params(self, params_tpl):
        #新しいパラメーターのtupleを受けとって上書きする。
        #HuberLossクラスにおいては、deltaのみ。
        
        #deltaの上書き
        delta = params_tpl[0]        
        self._delta = delta        
    
    @property
    def name(self):
        return self._name
        
    @property
    def trainable(self):
        return False
    
    @property
    def last_loss_layer(self):
        return True
    
    @property
    def input_shape(self):
        return self._input_shape
        
    @property
    def output_shape(self):
        return () #スカラー 
    
    @property
    def delta(self):
        return self._delta
    
###損失Layer　終わり###


###正則化Layer###

#public class Dropout extends Layer 
class Dropout(Layer):
    
    def __init__(self, name, input_shape, dropout_ratio=0.5):
        
        self.dropout_ratio = dropout_ratio
        self._mask_on_x = None        
        self._name = name
        self._input_shape = input_shape

    def forward(self, x, train_flg=False):
        
        if train_flg==True:
            #np.random.rand(*x.shape)：xと同じ形状で、数値が0.0以上、1.0未満の行列を返す
            #直後にdropout_ratio(0以上1以下)との大小比較をするので、同様に0.0以上、1.0未満の乱数を返すrandでなければならない。
            self._mask_on_x = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self._mask_on_x
        else:
            #重みスケーリング推論則（weight scaling inference rule）
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self._mask_on_x
    
    @property
    def name(self):
        return self._name
    
    @property
    def trainable(self):
        return False
    
    @property
    def last_loss_layer(self):
        return False
    
    @property
    def input_shape(self):
        return self._input_shape
        
    @property
    def output_shape(self):
        return self._input_shape
    
###正則化Layer　終わり###


###Utility Functions###

def calculate_init_std_weight(n1, n2, option, default_init_std=0.1):

    #weightの初期値決定に使用する標準偏差の算出
    
    if option == "xavier":
        std = np.sqrt(2 / (n1 + n2))
    elif option == "He":
        std = np.sqrt(2 / n1)
    else:
        std = default_init_std
    return std

def read_pickle_file(file_path):
    #指定されたパスのpickleファイルを読み込む。
    
    with open(file_path, "rb") as fo:
        obj = pickle.load(fo)
        
    return obj
        
def save_pickle_file(obj, file_path):
    #指定されたオブジェクトを指定されたパスのpickleファイルとして書き込む。
    
    with open(file_path, 'wb') as fo:
        pickle.dump(obj , fo) 

###Utility Functions　終わり###

###Optimizers###

class RMSPropGraves:    
   
    def __init__(self, lr=0.001, rho=0.95):
        
        self._lr = lr
        self._h = None
        self._rho = rho
        self._epsilon = 0.0001
        
    def update(self, params, grads_params):
        
        if self._h is None:
            #初回のみ hとmを初期化
            
            self._h = {}
            self._m = {}
            
            for key, value in params.items():
                self._h[key] = np.zeros_like(value)
                self._m[key] = np.zeros_like(value)
                
        for key in params.keys():
            #h(t) = ρ * h(t-1) + (1-ρ) * g(t)^2
            self._h[key] = self._rho * self._h[key] + (1 - self._rho) * grads_params[key] * grads_params[key]
            #m(t) = ρ * m(t-1) + (1-ρ) * g(t)
            self._m[key] = self._rho * self._m[key] + (1 - self._rho) * grads_params[key]
            #∇W(t) = -(lr * ∇W(t-1) ) / sqrt( h(t) - m(t)^2 + ε )
            params[key] -= self._lr * grads_params[key] / ( np.sqrt(self._h[key] - (self._m[key] * self._m[key]) + self._epsilon) ) 
            
class Adam:
    
    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999):
        
        self._lr = lr
        self._rho1 = rho1
        self._rho2 = rho2
        self._m = None
        self._v = None
        self._epsilon = 1e-8
        self._iter_count = 0
        
    def update(self, params, grads_params):

        if self._iter_count==0:
            #初回のみ mとvを初期化
            
            self._m = {}
            self._v = {}

            for key, value in params.items():
                self._m[key] = np.zeros_like(value)
                self._v[key] = np.zeros_like(value) 
               
        for key in params.keys():
            
            self._m[key] = self._rho1*self._m[key] + (1-self._rho1)*grads_params[key] 
            self._v[key] = self._rho2*self._v[key] + (1-self._rho2)*(grads_params[key]**2)            
            
            m = self._m[key] / ( 1 - self._rho1**(self._iter_count+1) )
            v = self._v[key] / ( 1 - self._rho2**(self._iter_count+1) )
            
            params[key] -= self._lr * m / (np.sqrt(v) + self._epsilon)
            
        self._iter_count += 1

###Optimizers　終わり###
