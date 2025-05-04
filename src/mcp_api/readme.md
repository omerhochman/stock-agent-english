# A 股 Mcp

## 安装

由于 Mcp 目前使用 uv 管理包最方便，所以这里采取不同于 conda 的包管理方式：

```bash
# 安装 uv
pip install uv

# 创建环境
cd mcp_api/
uv venv

# 激活环境
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 下载包
uv sync
```

## 使用

在 [Cherry Studio 客户端](https://www.cherry-ai.com/)，进行如下配置：

![mcp_config](../../assets/img/mcp_config.png)

参数配置：

```
--directory
C:\\Users\\xxx\\xxx\\stock_agent\\src\\mcp_api
run
python
mcp_server.py
```

### 结果展示

![mcp_1](../../assets/img/mcp_1.png)

![mcp_2](../../assets/img/mcp_2.png)

![mcp_3](../../assets/img/mcp_3.png)

![mcp_4](../../assets/img/mcp_4.png)
