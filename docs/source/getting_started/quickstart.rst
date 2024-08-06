.. _quickstart:

快速开始
==========

这个教程将按照以下步骤演示如何使用Lightllm提供LLM或者VLM模型服务：

* 准备Lightllm 所支持的模型文件
* 启动大语言模型的 API 服务
* 启动多模态大语言模型的 API 服务

在继续这个教程之前，请确保你完成了 :ref:`安装指南 <installation>` .



准备模型文件
-------------------------

下面的内容将会以 `Qwen2-0.5B <https://huggingface.co/Qwen/Qwen2-0.5B>`_ 和 `Qwen-VL-Chat <https://huggingface.co/Qwen/Qwen-VL-Chat>`_ 为例，分别演示lightllm对大语言模型以及多模态模型的支持。
下载模型的方法可以参考文章：`如何快速下载huggingface模型——全方法总结 <https://zhuanlan.zhihu.com/p/663712983>`_ 

这是下载模型的示例代码：

.. code-block:: console

    $ # mkdirs ~/models && cd ~/models
    $
    $ pip install -U huggingface_hub
    $
    $ huggingface-cli download Qwen/Qwen2-0.5B --local-dir Qwen2-0.5
    $
    $ huggingface-cli download Qwen/Qwen-VL-Chat --local-dir Qwen-VL-Chat

.. note::
    上面的下载模型的代码需要科学上网，并且需要花费大量时间，你可以使用其它下载方式或者其它支持的模型作为替代。最新的支持的模型的列表请查看 `项目主页 <https://github.com/ModelTC/lightllm>`_ 。


部署LLM服务
-------------------------

下载完Qwen2-0.5B模型以后，在终端使用下面的代码部署API服务：

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Qwen2-0.5B  \
    $                                       --host 0.0.0.0                  \
    $                                       --port 8080                     \
    $                                       --tp 1                          \
    $                                       --max_total_token_num 120000    \
    $                                       --trust_remote_code

.. note::
    上面代码中的 ``--model_dir`` 参数需要修改为你本机实际的模型路径

服务成功启动后，在另一个终端对API服务进行测试：

.. code-block:: console

    $ curl http://localhost:8080/generate \
    $      -H "Content-Type: application/json" \
    $      -d '{
    $            "inputs": "What is AI?",
    $            "parameters":{
    $              "max_new_tokens":17, 
    $              "frequency_penalty":1
    $            }
    $           }'

部署VLM服务
-------------------------

下载完Qwen-VL-Chat模型以后，在终端使用下面的代码部署API服务：

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Qwen-VL-Chat  \
    $                                       --host 0.0.0.0                    \
    $                                       --port 8080                       \
    $                                       --tp 1                            \
    $                                       --max_total_token_num 120000      \
    $                                       --trust_remote_code               \
    $                                       --enable_multimodal               \
    $                                       --cache_capacity 1000   

.. note::
    上面代码中的 ``--model_dir`` 参数需要修改为你本机实际的模型路径          

服务成功启动后，使用如下的python代码对API服务进行测试：

.. code-block:: python
    
    import json
    import requests
    import base64

    def run(query, uris):
        images = []
        for uri in uris:
            if uri.startswith("http"):
                images.append({"type": "url", "data": uri})
            else:
                with open(uri, 'rb') as fin:
                    b64 = base64.b64encode(fin.read()).decode("utf-8")
                images.append({'type': "base64", "data": b64})

        data = {
            "inputs": query,
            "parameters": {
                "max_new_tokens": 200,
                # The space before <|endoftext|> is important, 
                # the server will remove the first bos_token_id, 
                # but QWen tokenizer does not has bos_token_id
                "stop_sequences": [" <|endoftext|>", " <|im_start|>", " <|im_end|>"],
            },
            "multimodal_params": {
                "images": images,
            }
        }

        url = "http://127.0.0.1:8080/generate"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response

    query = """
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <img></img>
    这是什么？<|im_end|>
    <|im_start|>assistant
    """

    response = run(
        uris = [
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        ],
        query = query
    )

    if response.status_code == 200:
        print(f"Result: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

