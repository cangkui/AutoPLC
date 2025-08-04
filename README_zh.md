# AutoPLC: Generating Vendor-Aware Structured Text for Programmable Logic Controllers

本仓库包含AutoPLC相关研究的补充材料。AutoPLC是一个用于IEC 61131-3 ST 生成的自动编程框架。


## 目录结构

### 1. 基准数据集
`data/benchmarks`目录包含实验中使用的4个数据集：
- **oscat**：与OSCAT库相关的任务和数据。
- **lgf**：涉及LGF函数的任务。
- **competition**：源自工业代码生成竞赛的数据集。
- **agents4plc benchmark**：由Agents4PLC团队提出的基准数据集。


### 2. 源代码
`src`目录包含AutoPLC代码实现：
<!-- - **baselines**：用于对比实验的基准模型实现。 -->
- **rag_data**：项目根目录下，RAG相关资源，包括Rq2ST基准和案例库。
- **autoplc_scl**：AutoPLC在Siemens SCL中的实现。
- **autoplc_st**：AutoPLC在CODESYS ST中的实现。


## 使用指南

### 环境安装

推荐在Windows 10专业版上运行AutoPLC，该系统兼容TIA Portal V19和CODESYS V3.5 SP20。

1. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

2. Bash环境配置  
   执行脚本前，请先准备环境遍历：将`.env.example`复制为`.env`并填写必要信息。

3. 编译工具配置  
   - Siemens SCL：使用西门子提供的TIA Openness API，需满足：  
     1) C# .NET Framework运行时  
     2) Windows系统上安装TIA Portal V19。详见[TIACompileService](https://github.com/cangkui/TIACompileService)，仓库提供源代码实现，并且Release中提供了经过测试的二进制包可用于下载。  
   - CODESYS ST：使用CODESYS脚本引擎，需在Windows系统上安装CODESYS V3.5 SP20。详见[CODESYSCompileService](https://github.com/cangkui/CODESYSCompileService)，Release中提供了经过测试的CODESYS版本安装包。  

   > 其中CODESYS编译工具参考了仓库[codesys-api](https://github.com/johannesPettersson80/codesys-api)的实现，特此感谢该仓库作者的贡献。  
   > 每个工具均被封装为标准HTTP服务供AutoPLC调用。为减轻单个服务的负载，我们在局域网内的多台机器上部署了多个服务实例，可通过配置文件进行设置。
   > 可在yaml配置文件当中修改编译服务地址。

4. 知识库配置  
   我们使用智谱AI团队提供的在线知识库，并集成其glm-airx模型用于重排序。详见[https://open.bigmodel.cn/dev/howuse/retrieval](https://open.bigmodel.cn/dev/howuse/retrieval)。


### 运行AutoPLC

执行以下命令运行AutoPLC：
```bash
python src/autoplc/main.py --benchmark [基准数据集名称] --config [配置名称，与配置文件名一致]
```

将`[基准数据集名称]`替换为目标数据集（如`oscat`、`lgf`、`competition`、`agents4plc`），`[配置名称]`替换为所需配置（需在`src/config`目录中创建）。

若需生成ST代码，将命令中的`main.py`替换为`main_st.py`即可。


---

2025 AutoPLC 团队。
