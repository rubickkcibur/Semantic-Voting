# Setup
首先请保证宿主机正确安装docker, cuda>12.8，以及nvidia-container-toolkit, 否则无法正确使用cuda

请下载sentence-embedding模型到本地路径`<your_model_path>`，例如：
> cd <your_model_path> \
> git clone https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased

也可以下载别的embedding模型，只要该模型能被`transformers.AutoModel`正确识别，具体可以参考`app.py`

切换到本目录下，从dockerfile构建镜像
> docker build -t sv-image .

启动容器
> docker run -d --runtime=nvidia --gpus all --shm-size="10g" --cap-add=SYS_ADMIN -v .:/app -v <your_model_path>:/models -p 8111:8000 --name semantic-voting-app sv-image:latest

监控实时log(Optional)
> docker logs -f semantic-voting-app > app.log 2>&1 &

如果正常启动将看到
> INFO:     Started server process [1]\
INFO:     Waiting for application startup.\
INFO:     Application startup complete. \
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

可以运行测试样例进行初步检查
> python server_test.py

# 数据格式说明
## 加载embedding模型
启动服务后，应当首先调用load_model保证正确加载模型:
```
response = requests.post(
    "http://localhost:8111/load_model",
    json={
        "model_path": <emb_model_name>, # 相对于<your_model_path>的路径，以本markdwon中例子而言是"unsup-simcse-bert-base-uncased"
        "device": "cuda:0" # 加载位置，默认是"cuda:0"
    }
)
response.json()
```
如果加载正确，会看到`{'status': 'model loaded'}`, 如果错误`status`会返回报错原因

## 调用semantic-voting计算
输入参数：

一个字符串的列表`cands: list[str]`，实际意义是模型针对同一个prompt的多次采样输出。为保持voting有实际意义，应当保证`cands`里面的文本是对同一个输入的多个输出，即语义相近。如果场景中输入文本对语义影响很大，可以将输入与输出拼接起来。

(特别的，为保证效果稳定，`cands`长度最好大于16。此外，能够处理的的文本长度受限于embedding model的能力，对于"unsup-simcse-bert-base-uncased"模型而言，一两句话的长度效果会比较好，如果要处理更长的文本，需要使用能处理长文本的embedding模型。

批处理大小`batch_size`,embedding模型处理`cands`时的批处理大小，如果OOM了，可以降低这个值

聚类策略`clustering`, 目前支持"whole"和"HDBSCAN"两种方式。"whole"模式下，`cands`中的所有文本都会参与voting，如果保证`cands`中没有质量过差的文本(比如乱码，胡言乱语)，可以采用这种模式; "HDBSCAN"模式下，会调用HDBSCAN先进行聚类，排除掉语义偏移大的文本，只有最大聚类中的文本才会参与voting，这种情况下，需要额外设置两个聚类参数，见下

HDBSCAN聚类参数`HDBSCAN_min_cluster_size`和`HDBSCAN_min_samples`，控制聚类行为，具体含义可以参考[HDBSCAN文档](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html), 调整时，应保证`HDBSCAN_min_cluster_size`大于等于`HDBSCAN_min_samples`，另外`HDBSCAN_min_cluster_size`应小于`cands`长度

一个调用例子
```
cands = ["The cat is on the table.", "A cat is sitting on the table.", "The dog is in the garden.", "There is a cat on the table.", "The cat is on the roof.", "A dog is playing in the garden.", "The cat is sleeping on the table.", "The dog is barking in the garden.", "The cat is on the floor.", "A dog is running in the garden."]

json_data = {
    "candidates": cands,
    "clustering": "whole",
    "batch_size": 16,
    "HDBSCAN_min_cluster_size": 5,
    "HDBSCAN_min_samples": 2,
}

response = requests.post("http://localhost:8111/predict", json=json_data)
response.json()
```
如果调用正常，会返回`{'scores': [...], 'status': 'Done'}`
其中`scores`是voting得分，越高越好，长度等于`cands`,`status`在成功时为"Done"，其他情况下是报错信息

特别注意：如果使用HDBSCAN模式，返回的`scores`中会含有`None`值，表示这个文本没有参与voting过程，**这不暗含任何对于这个文本的质量评价**，但一般表示这个文本与大部分文本的语义距离较远。







