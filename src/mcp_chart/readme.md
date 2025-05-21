# A 股辅助画图 MCP

Forked from: https://github.com/antvis/mcp-server-chart.git

本项目作为 `mcp_api` 的辅助 mcp，用于可视化数据，使得 ai 生成的报告更美观。

### 使用指南

1. 首先需要确保安装了`nodejs`和`npm`等前端工具

验证安装：

```bash
C:\Users\15170\Desktop\stock_agent>node -v
v20.15.1

C:\Users\15170\Desktop\stock_agent>npm -v
10.7.0
```

2. 在 Cline 中进行配置：

```json
{
  "mcpServers": {
    "mcp-server-chart": {
      "disabled": false,
      "timeout": 60,
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@antv/mcp-server-chart"],
      "transportType": "stdio"
    }
    // 其他mcp服务器配置
  }
}
```

3. 与 A 股 MCP 联合使用效果：

![mcp](../../assets/img/mcp_5.png)

![mcp](../../assets/img/mcp_6.png)

![mcp](../../assets/img/mcp_7.png)

![mcp](../../assets/img/mcp_8.png)

![mcp](../../assets/img/mcp_9.png)
