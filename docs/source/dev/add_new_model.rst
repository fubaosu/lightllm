.. _add_new_model:

添加新模型支持
===================


1. 当前推理框架介绍
--------------------

在 ``lightllm/common/basemodel`` 目录下，是整个推理架构的基类实现

::

    ├── basemodel.py   # 模型框架类
    ├── infer_struct.py # 推理用的状态类
    ├── __init__.py
    ├── layer_infer # 推理层的基类实现
    │   ├── base_layer_infer.py
    │   ├── __init__.py
    │   ├── post_layer_infer.py
    │   ├── pre_layer_infer.py
    │   ├── template # 推理层的模板实现，继承实现模板可以减少开发量和重复代码
    │   │   ├── __init__.py
    │   │   ├── post_layer_infer_template.py
    │   │   ├── pre_layer_infer_template.py
    │   │   └── transformer_layer_infer_template.py
    │   └── transformer_layer_infer.py
    ├── layer_weights # 权重基类的实现
    │   ├── base_layer_weight.py
    │   ├── hf_load_utils.py
    │   ├── __init__.py
    │   ├── pre_and_post_layer_weight.py
    │   └── transformer_layer_weight.py
    └── triton_kernel # 一些公共使用的 triton kernel 算子
        ├── apply_penalty.py
        ├── destindex_copy_kv.py
        └── __init__.py

如上所示，目前模型推理架构主要由权重和推理两个部分组成。

**权重**

layer_weights 目录下是权重相关的代码，理论上对于一个新添加的模型需要继承实现 pre_and_post_layer_weight.py 和 transformer_layer_weight.py 中的 PreAndPostLayerWeight 和 TransformerLayerWeight 类来实现权重的加载。

.. list-table:: 
   :header-rows: 1

   * - 权重
     - 职责
   * - PreAndPostLayerWeight
     - 负责对LLM模型的第一层Embedding层和最后一层后处理层的权重加载并按照所使用的tp参数对权重进行拆分
   * - TransformerLayerWeight
     - 负责对LLM模型transformer层进行权重的加载按照所使用的tp参数对权重进行拆分


**推理**

layer_infer 目录下是进行推理处理的相关基类，并在template目录下提供了一些模板，从模板类进行继承实现可以减少一些不必要的重复代码，简化实现，该目录下需要继承实现的推理类有三个。

.. list-table:: 
   :header-rows: 1

   * - 推理基类
     - 职责
   * - PreLayerInfer
     - 负责对 Embedding 层的推理
   * - TransformerLayerInfer
     - 负责 transformer 层的推理
   * - PostLayerInfer
     - 负责将网络最后的隐层输出转化为logits的推理


上述三个类的基类 BaseLayerInfer 提供了两个最重要的对外服务函数接口，所有的推理行为都会由这两个接口进入。

.. code-block:: python

    # Batch进行第一次推理（在代码中又被叫做prefill）
    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        ...

    # 单步decode阶段的推理
    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        ...

**算子**

triton_kernel 目录下是一些使用 openai triton 实现的推理需要用到的算子。

**状态类**

infer_struct.py 中的 InferStateInfo 类是进行一次模型推理时，在层间传递一些重要信息的状态类，不同的模型可以继承实现该类，添加每个模型需要传递的独特状态信息， InferStateInfo 类提供了一个供继承的init_some_extra_state接口，用于传递额外独特信息的初始化。

.. code-block:: python

    def init_some_extra_state(self, 
        model, 
        batch_size, 
        total_token_num,
        max_len_in_batch,
        input_ids : torch.Tensor,
        b_loc : torch.Tensor,
        b_start_loc : torch.Tensor,
        b_seq_len : torch.Tensor,
        is_prefill
    ):
        pass

**模型框架类**

basemodel.py 中的 TpPartBaseModel 类，是整个模型的入口，每个类型的模型都需要继承实现该类。该类通过类似搭积木的方式，使用推理类，权重类，状态类完成模型的加载，推理功能，其中有很多接口可以被继承实现，以完成每个模型类型自己独特的操作。

.. code-block:: python

    class TpPartBaseModel:
    # weight class
    pre_and_post_weight_class = None
    transformer_weight_class = None

    # infer class
    pre_layer_infer_class = None
    post_layer_infer_class = None
    transformer_layer_infer_class = None

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[]):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.weight_dir_ = weight_dir
        self.max_total_token_num = max_total_token_num
        self.load_way = load_way
        self.mode = mode

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_weights()
        self._init_mem_manager()
        self._init_infer_layer()
        self._init_some_value()
        self._init_custom()
        return

   ...


常用需要继承实现的接口:

.. code-block:: python

    def _init_config(self):
        # 读取初始化模型的 config.json, 并进行一些 key 名的同名合法化操作
        pass

    def _verify_params(self):
        # 校验参数
        pass

    def _init_mem_manager(self):
        # 初始化 token attention 使用的 mem manager 对象
        pass

    def _init_some_value(self):
        # 初始化推理框架会使用的一些成员变量的值
        pass 

    def _init_custom(self):
        # 一些模型自己的个性化初始化，比如 llama 初始化自己的Rotary值
        pass


2. 添加 bloom 模型的示例说明
-----------------------------------

具体实现在 lightllm/models/bloom 目录下，请对应源码进行阅读，其中 triton_kernel 目录下为推理类使用的一些 kernel，下文中不做详细介绍，同时 bloom 模型因为不需要传递特殊状态信息使用默认的状态类即可。如想更深入的理解整个框架，可以进一步参考 llama 和 llama2 等模型的接入实现源码。

(1) 添加实验权重类

* ``lightllm/models/bloom/layer_weights/pre_and_post_layer_weight.py`` : 预处理和后处理的权重支持
* ``lightllm/models/bloom/layer_weights/transformer_layer_weight.py`` : transformer 块的权重支持

(2) 添加实现推理类

* ``lightllm/models/bloom/layer_weights/pre_layer_infer.py`` : 预处理推理类
* ``lightllm/models/bloom/layer_weights/transformer_layer_infer.py`` : transformer块推理类
* ``lightllm/models/bloom/layer_weights/post_layer_infer.py`` : 后处理类

(3) 实现模型的框架类

* ``lightllm/models/bloom/layer_weights/model.py`` : 模型框架类

(4) 在server服务层加入对模型的支持

* ``lightllm/server/router/model_infer/model_rpc.py`` : 添加对解析模型文件的判断
 