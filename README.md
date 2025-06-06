# 使用说明

## 相关的文件及文件夹

```
 ..
|--data/
|--net/
|--output/
|--picture/
|--model.py
|--gendata.py
|--estimator.py
|--RNN_estimator.py
|--simulate.py
|--functions.py
|--params.py
|--run_train.sh
```

## 各文件及其主要功能

### model.py

定义系统模型。现有模型类Discrete1（数值案例）、Continuous1（洛伦兹吸引子）、Uncertain1（包含模型不确定项的洛伦兹吸引子）、Continuous4（四元数姿态估计），对外接口为getModel函数，通过模型名称创建模型实例。

如需创建新的模型类，仿照现有类，继承Model类进行实现，注意子类至少需要实现model类中内容为pass的函数。使用自定义的类之前，注意在文件末尾的接口函数getModel中加上一行来通过函数名返回需要的模型类。

### gendata.py

生成状态及观测数据，对外接口函数为getData，通过模型名称、单条轨迹的时间步数、轨迹数、随机数种子号来生成数据。如果数据文件存在，数据将直接从文件中读取出来，如果不存在，则会根据参数随机生成数据。相同的随机数种子号生成的数据应该一样（不同设备、不同库函数版本生成的数据可能不一样）。返回的数据类型是list，第一个维度对应不同的轨迹，第二个维度对应不同的时间步。生成数据所依赖的相关参数，例如初始状态的均值、协方差，噪声项的均值、协方差等，需要在params.py文件中修改getModelParams函数内模型名称所对应的参数定义。

也可以直接运行该文件，通过修改最下方main函数中的内容，来直接生成数据文件。当需要多次使用相同数据时，推荐先使用这种方法生成数据文件。运行之前，需要在该文件的路径下创建一个名为“data”的文件夹。在生成数据时，如果检测到相同文件名的数据文件存在，会在终端中提示"Data file already exist, input \"y\" to regenerate : "，如果想要覆盖生成数据文件，键入“y”回车即可，其他输入均会终止程序运行而保留原数据文件。

如果需要修改所保存文件的命名方式，修改generate_trajectories函数中的fileName变量定义即可。

### estimator.py

主要定义估计器相关的类。

Estimator类：估计器的通用类，规定了不同的估计器需要包含的元素实现的功能函数，方便编写统一的测试函数。

EKF类：通用的EKF估计器。

EKFForC4类：为四元数估计“Continuous4”模型所实现的特定EKF估计器，该类中的估计算法是参考 [presentation.pdf](quaternion-kalman-filter-main\doc\presentation.pdf) 和 “quaternion-kalman-filter-main”文件夹中的相关代码编写而成，有一些个人的改动，具体可参阅 [quaternion note.pdf](quaternion-kalman-filter-main\quaternion note.pdf) ，目前该估计器的估计性能与参考代码相当。

MHE类：通用的MHE估计器。

MHEForC4：为四元数估计“Continuous4”模型所实现的特定MHE估计器，相对应的为了求解所需的非线性最小二乘问题，有NLSFForC4、resForC4、jacForC4函数。这个估计器目前效果还不好。

NLSF_uniform函数：求解非线性最小二乘问题的通用函数，用于MHE类中。主要依赖scipy库中的least_squares函数，关于该函数的详细信息可以参考官方文档[scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。该函数中还增加了关于约束条件的参数，即下界lb和上界ub，它们可以通过参数传入或直接人为设置，目前参数传入的方式写得不够好，仅支持在RNN_estimator.py文件中使用该函数时以参数方式传入，simulate.py中调用MHE类时，必须人为设置约束条件。

Quadratic类：指定到达代价为二次型，定义了二次型到达代价所对应的非线性最小二乘目标函数及其对应的雅可比矩阵，目标函数以及雅可比矩阵的维度和定义方式参考官方文档[scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。

SumOfSquares类：指定到达代价为平方和形式，定义了平方和形式的到达代价所对应的非线性最小二乘目标函数及其对应的雅可比矩阵，目前这个类可能有问题，对应的MHE估计器性能不好。

UKF类：调用的UKF估计器。

### RNN_estimator.py

ActorRNN类：循环神经网络类，参数dim_fc1是list类型，用于指定RNN前的全连接层；参数dim_fc2是list类型，用于指定RNN后的全连接层；参数dim_rnn_hidden是int类型，用于指定RNN层的神经元个数；type_activate用于指定全连接层的激活函数类型；type_rnn用于指定RNN的类型；num_rnn_layers用于指定RNN的层数；还有一些pytorch标准库中对于神经网络层支持的超参数，可以自行添加实现。该类的输入输出维度在forward函数的注释中有写。

grad_clipping函数：梯度裁剪函数，用于防止梯度爆炸，该函数目前在代码中的调用都被注释掉了，如果训练正常可以不启用该函数。

RL_estimator类：强化学习估计器类，即真正用于状态估计的估计器类型，它进行状态估计的函数是estimate函数，训练是针对其policy元素进行训练，policy元素实际上就是上文中ActorRNN类的一个实例。初始化使用的函数是initialize，这里面可以改的主要有开始两行中的学习率lr，以及scheduler定义中的一些参数，关于这些参数的含义可以直接搜一下对torch.optim.lr_scheduler.ReduceLROnPlateau类的说明。另外，目前的代码将target_c_seq固定为0，实际上它是可以计算的，也可以进行相应的修改。训练使用的函数是train，这里面可以改动的主要是train_window参数，也就是选择以多长的窗口为学习的目标，这个参数可以在文件末尾的main函数中直接修改，也可以在param.py文件中修改，但main函数中的指定值会覆盖param.py中的默认参数。

main函数：相关的参数指定基本集中在函数的前面，一些默认的参数（在多个估计器中保持一致的）需要在param.py文件中修改，代码中都分块好了，应该是比较直观的。需要注意的是使用run_train.sh自动测试时需要将main函数中的下面这部分代码注释掉，因为这部分代码会覆盖默认参数值。

```python
#region 修改参数以便人工测试（自动测试时注释掉，否则参数无法自动变化）
trainParams["lr"] = 5e-3
trainParams["lr_min"] = 1e-6
trainParams["gamma"] = 1.0
args.hidden_layer = ([], 64, [128]) # 这几个要同步修改
nnParams["dim_fc1"] = [] # 这几个要同步修改
nnParams["dim_rnn_hidden"] = 64 # 这几个要同步修改
nnParams["dim_fc2"] = [128] # 这几个要同步修改
nnParams["dropout"] = 0.1
nnParams["num_rnn_layers"] = 3
nnParams["type_activate"] = "elu"
nnParams["type_rnn"] = "lstm"
#endregion
```

训练时，修改好main函数中的模型名称以及测试用的轨迹参数，直接运行该文件即可，也可以运行run_train.sh针对多种超参数进行自动训练。训练前，需要在当前路径下创建“output"文件夹。

### RNN_estimator_copy.py

该文件仅用于测试窗口长度不为1时，训练好的RL_estimator的估计性能，使用时，修改main函数中测试轨迹的参数，以及模型网络的相关参数，然后加载训练好的模型进行测试即可。

### RNN_estimator_C4.py

该文件用于训练针对四元数姿态估计的模型（Continuous4），目前效果不好，可以忽略。

### run_train.sh

用于自动化测试多种超参数的脚本文件，想使用什么超参数就在对应的list里面增减即可。然后在git bash终端中运行。

### params.py

主要作用是自动测试时能够解析传入的超参数，有什么想在自动测试时传入的超参数，就将其加到parseParams函数中，然后在主函数中将其读取出来即可。

另外，在新定义一个模型之后，需要在该文件中相应的地方新增一份针对该模型的相关参数定义，用于生成测试数据，以及指定状态估计器所使用的参数，在getModelParams函数以及getEstParams函数中，新增的方式很直观，复制原有的修改即可。

getTrainParams函数中定义了训练过程中会使用到的参数，包括训练数据的时间步长、集数、开始及最小学习率、学习目标的窗口长度等，aver_num参数是每个时间步上会生成多少噪声数据用于训练；seq_len参数是每隔多少个时间步就对RNN网络进行一次反向传播，相当于将长数据进行截断处理；saveFile参数指定训练好的模型保存时的名称，在多次训练时不需要修改它，有同名时程序会自动加后缀数字进行区分。

getNNParams函数中定义了神经网络所需要的相关参数，都比较直观，且都是在ActorRNN类中出现的，在此不做详细说明。

### functions.py

一些功能函数，基本上都比较直观，这里不做详细说明。

### simulate.py

用于测试的文件，在文件末尾的main函数的前几行，指定测试轨迹的相关参数，是否存在模型误差modelErr，是否需要打印isPrint（一般这个需要开启，因为不开启时没有任何反馈，是在别的函数中调用的），是否需要绘图isPlot（可选），然后指定需要测试的方法test_options，可以同时选择多种测试方法。然后运行该文件即可。相关结果会在output/test_result.txt文件中打印出来。

如果想新增测试方法，或者修改现有的测试方法，修改下面“测试”区块的代码即可，这部分代码是针对每个估计器，定义估计器实例，然后调用simulate函数实现的，比较直观。如果在别的文件中需要测试估计器的性能，调用simulate函数即可。

### plot.py

绘图文件，其中针对不同的需要定义了不同的函数，使用前需要执行一遍simulate函数，并取消“保存数据temp”区块的代码注释，以生成相应的数据文件。主要包括以下绘图函数：

plotReward：用于绘制奖励值的变化曲线，使用时，在主函数中用一个列表包含所有奖励值即可，代码中有示例。

plotStepMSE：绘制逐步估计误差曲线图。

plotStateMSE：绘制三维状态估计曲线图。

绘图效果可见对应的图片文件，相关绘图代码比较容易看，一些特殊的地方加了注释，这里不再赘述。