==================================================
第3章　最尤推定法　Maximum Likelihood Estimation
==================================================

内容
====

3.1 確率モデルの利用

　　あるデータが得られる確率が最大となるようなパラメータを推定。
　　
　　（1） パラメータを含むモデル（数式）を設定する

　　（2） パラメータを評価する基準を定める

　　（3） 最良の評価を与えるパラメータを決定する

3.1.1 「データ発生確率」の設定

　　モデル： 

   (1) データの背後にはM次多項式の関係があり、さらに標準偏差 :math:`\sigma` の誤差が含まれている

.. math:: 

    \it{f}(x) &= w_0 + w_1 x + w_2 x^2 + \cdots + w_M x^M \\
            &= \sum_{m=0}^M w_m x^m  
　

　　(2) 観測点　:math:`x_n` における観測値 :math:`t` は、:math:`f(x_n)` を中心としておよそ :math:`f(x_n) \pm \sigma` の範囲に散らばる　　
　　(平均 :math:`f(x_n)`、分散 :math:`\sigma` の正規分布)

.. math:: 

    \it{N} ( t | f(x_n), \sigma^2 ) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{- \frac{1}{2 \sigma^2}(t - f(x_n))^2}





この２つの式の :math:`w_m` と :math:`\sigma` とを推定する。
最小二乗法との違い：データに含まれる誤差を合わせて推定する





3.1.2 尤度関数によるパラメーターの評価
   評価方法：　トレーニングセットに含まれるデータ :math:`{(x_n,t_n)}^N_{n=1}` が得られる確率
   

   .. math::

        P &= N(t_1| f(x_1),\sigma^2) \times \cdots \times N(t_N| f(x_N),\sigma^2) \\
          &= \prod_{n=1}^{N} N(t_n| f(x_n),\sigma^2)

　　尤度関数：　:math:`\left\{ \begin{array}{||} パラメータ： & w_m と \sigma \\
　　　　　　　値：　& トレーニングセットのデータが得られる確率 \end{array} \right .`

　　「最尤推定法」

　　* 「観測されたデータ（トレーニングセット）は、最も発生確率が高いデータに違いない」との仮説

　　* 確率Pが最大になるようなパラーメータを推定 

　　* 尤度関数の最大値問題 


3.1.3 

   .. math::
      
      P &= \prod_{n=1}^{N} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{- \frac{1}{2 \sigma^2}(t_n - f(x_n))^2} \\
        &= (\frac{1}{2 \pi \sigma^2})^{\frac{N}{2}} exp[- \frac{1}{2 \sigma^2} \sum_{n=1}^{N} \{t_n - f(x_n)\}^2 ]
 




　　ここで　自乗誤差 :math:`E_p = \frac{1}{2} \sum_{n=1}^{N} \{ f(x_n) - t_n\}^2 \}` 



   .. math:: 

        P = (\frac{1}{2 \pi \sigma^2})^{\frac{N}{2}} e^{- \frac{1}{\sigma^2} E_p}


   ここで　:math:`\beta = \frac{1}{\sigma^2}` とし、 :math:`E_p` とパラメータ :math:`w` の依存関係を明示

   .. math:: 

      P(\beta,w) = (\frac{\beta}{2 \pi})^{\frac{N}{2}} e^{- \beta E_p(w)}

　　  これを最大にするパラメータ :math:`(\beta,w)` を求める。　
      この尤度関数は単調増加関数なので対数をとっても単調増加する。（対数尤度関数）

   .. math:: 
      

      ln P(\beta,w) = \frac{N}{2}ln \beta - \frac{N}{2} ln 2 \pi - \beta E_p(w)

　　　対数尤度関数を最大化する条件：
   .. math:: 

      \frac{\partial (ln P)}{\partial w_m} &= 0  \qquad       (m = 0, \cdots, M) \\
      \frac{\partial (ln P)}{\partial \beta} &= 0


　　　:math:`w_m` について：
   .. math:: 


      \frac{\partial E_p}{\partial w_m} = 0 \qquad (m =0, \cdots,M)

これは自乗誤差を最小にする条件と同じ：　多項式の係数 :math:`\{w_m\}_{m=0}^{M}`　は最小二乗法と同じ

   .. math:: 
      
      w = (\Phi^T \Phi)^{-1} \Phi^T t

      \Phi = \left ( \begin{array}{llll}
               x_1^0 & x_1^1 & \cdots & x_1^M \\
               x_2^0 & x_2^1 & \cdots & x_2^M \\
               \vdots & \vdots & \ddots & \vdots \\
                x_N^0 & x_N^1 & \cdots & x_N^M \\
                \end{array} \right )





　　　:math:`\beta` について：

   .. math:: 

      \frac{1}{\beta} = \frac{2 E_p}{N}

      \sigma &= \sqrt{\frac{1}{\beta}} = \sqrt{\frac{2 E_p}{N}} = E_{RMS} \\
             &= \sqrt{\frac{1}{N} \sum_{n=1}^{N} \left ( \sum_{m=0}^{M} w_m x_n^m -t_n \right )^2 }
      
    これは最小二乗法の平方根平均自乗誤差


計算結果：
    　　.. image:: figure_3.png 
    　　.. image:: figure_4.png 
    　　.. image:: figure_5.png 
    　　.. image:: figure_6.png 
    　　.. image:: figure_5.png 
    　　.. image:: figure_6.png 
    　　.. image:: AISECjp20160824-fig1.png 


参考URL
人工知能に関する断創録　最尤推定、MAP推定、ベイズ推定
http://cp.the-premium.jp/

最尤法によるパラメータ推定の意味と具体例 | 高校数学の美しい物語
http://mathtrain.jp/mle
