# Supplementary Materials Overview

This repository contains supplementary materials for our research on AutoPLC, a flexible framework for structured text (ST) generation. The materials are organized into the following main directories:

## Directory Structure

### 1. **Benchmarks**
This folder includes three datasets used in our experiments:
- **oscat**: Contains tasks and data relevant to the OSCAT library.
- **lgf**: Includes tasks related to LGF functions.
- **competition**: Comprises datasets derived from the industrial code generation competition.

### 2. **Experiment_Results**
This directory holds the experimental results obtained from our study. The results demonstrate the performance of different baselines and configurations of AutoPLC on the provided benchmarks.

### 3. **Scripts**
This folder contains the code used for our experiments and implementations. It is further organized into the following subdirectories:

- **experiment**: Contains the baseline implementations used for comparison in our study.
- **rag_data**: Includes resources for retrieval-augmented generation (RAG), such as the Rq2ST benchmark and the case library.
- **scl_team_coder**: Represents the implementation of AutoPLC applied to Siemens SCL.
- **st_team_coder**: Represents the implementation of AutoPLC applied to CODESYS ST.

---

## Usage Guide

### Installation
1. Install the necessary dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Bash Environment Setup
Prior to executing the scripts, please ensure that the following environment variables are properly configured. This step is exclusively for testing purposes and does not involve any leakage of personal information. 

   ```bash
   export GPT_API_KEY_LIST=sk-Poyu4IIJG4VXpQjj6a8cAdE6126e4e26BfA07f62Cb8cDd07,sk-le8JA4P1jeeltzcm6756F97d39A343339405F5Ca09Ef2957
   export GPT_API_KEY=sk-heJMQX0Z4FEBm1ve7a5320F48f034fE688D104E53c2fBe45
   export API_KEY_KNOWLEDGE=664db87328b7f19ffcd10b9a8a4d9147.Ak1fANcqRx4CJPC0
   export DEEPSEEK=sk-089e91649ae948f1b99665c1a8ef0c57
   ```

### Running Baselines
1. Modify the configuration in `experiment/<corresponding_baseline>/run.py` to suit your requirements.
2. Update the experiment entry point in `experiment/__main__.py`.
3. Run the baseline experiment from the root directory:
   ```bash
   python -m experiment
   ```

### Running AutoPLC
1. Modify the `dataset` and `prompt_file` parameter in `st_team_coder/__main__.py` (only oscat_en) or `scl_team_coder/__main__.py`(lgf_en or competition_en) to the one you want to experiment with.
2. Start the experiment from the root directory:
   ```bash
   python -m st_team_coder
   ```
3. Similarly, for Siemens SCL, update `scl_team_coder/__main__.py` and run:
   ```bash
   python -m scl_team_coder
   ```

case from github:
https://github.com/panasewicz/PLC_SCL_Motion_Control_Example/blob/main/Picker_Station.scl
https://github.com/lopez-dev/Siemens-SCL-Source-Files/blob/main/Meldungen/Meldungen.scl
https://github.com/LCC-Automation/OpenPID-TIA-SCL/blob/main/LLCCA_TimeLag_1_0_1.scl

---

Thank you!


## How to use Openness Project

### Chinese

我们强烈推荐使用安装有Siemens TIA Portal V19的Windows 10专业版系统虚拟机来测试部署本项目。

#### 先决条件

> * 系统运行环境：Windows 10 系统（强烈推荐使用专业版）
> * 软件运行环境：Siemens TIA Portal V19。**注意，安装TIA Portal V19时请务必勾选TIA Openness功能选项。**
> * 确保TIA Portal V19处于试用期或者许可证没有过期。

#### 部署教程

1. 将当前用户加入TIA Openness用户组。打开计算机管理=>本地用户和组=>组，选中Siemens TIA Openness用户组，右键属性=>添加到组，添加Administrator用户和当前用户，应用并退出。重启系统或者注销当前用户后重新登录。
2. 为了保证兼容性，请务必将`TiaImportExample.exe`同级目录下的`Siemens.Engineering.dll`和`Siemens.Engineering.Hmi.dll`替换为本地TIA Portal V19提供的相应文件。这两个文件一般位于TIA Portal V19安装目录下，默认位置在`C:\Program Files\Siemens\Automation\Portal V19\PublicAPI\V19`。
3. 在`TiaImportExample.exe`所在目录下，以管理员身份打开命令提示符. 运行`.\TiaImportExample.exe`，等待TIA Portal V19弹出申请窗口，点击全部允许。
4. 当程序打印出类似于以下内容时说明运行成功。

```
Start initializing ...
Project Name: evaluation
Project Version:
Opened: evaluation
Initializing success.
Controller founded: TiaImportExample.Controllers.HomeController
Controller founded: TiaCompilerCLI.Controllers.TiaApiController
TIA Portal API service started, access address: http://192.168.103.245:9000/
StatusCode: 200, ReasonPhrase: 'OK', Version: 1.1, Content: System.Net.Http.StreamContent, Headers:
{
  Date: Sun, 18 May 2025 14:45:33 GMT
  Server: Microsoft-HTTPAPI/2.0
  Content-Length: 18
  Content-Type: application/json; charset=utf-8
}
HTTP service initialization successful!
Press Enter to exit...
```

> 可以在同一局域网内通过 `curl http://192.168.103.245:9000/api/home` 测试接口访问是否正常，正常情况下应该显示`"Hello, World!"`。  
> 注意，正常情况下程序将自动扫描可用的局域网IP地址并将该地址作为程序监听地址，端口号默认为9000。如果您使用的是VMWare Workstation，请将虚拟机的网络设置为桥接（物理直连）以确保宿主机所在局域网能够访问到程序部署的HTTP服务。

### English

We strongly recommend testing and deploying this project on a Windows 10 Pro virtual machine installed with Siemens TIA Portal V19.

#### Prerequisites

> * System Environment: Windows 10 (Pro edition is strongly recommended).
> * Software Environment: Siemens TIA Portal V19. Note: When installing TIA Portal V19, ensure the TIA Openness feature is checked.
> * Ensure TIA Portal V19 is in the trial period or your license has not expired.

#### Deployment Tutorial

1. Add the current user to the TIA Openness user group.
> Tip  
> Open **Computer Management** > **Local Users and Groups** > **Groups**, select the **Siemens TIA Openness** group, right-click **Properties** > **Add**, include the **Administrator** user and the current user. Apply changes and exit. Restart the system or log out and log back in.  
2. Replace DLL.
> To ensure compatibility, replace the `Siemens.Engineering.dll` and `Siemens.Engineering.Hmi.dll` in the same directory as `TiaImportExample.exe` with the corresponding files from the local TIA Portal V19 installation.  
> These files are typically located in the TIA Portal V19 installation directory, by default at `C:\Program Files\Siemens\Automation\Portal V19\PublicAPI\V19`.  
3. Run `.\TiaImportExample.exe`, wait for the TIA Portal V19 permission window to pop up, and click **Allow All**. When the program prints output similar to the following code block, it indicates success. 

```
Start initializing ...
Project Name: evaluation
Project Version:
Opened: evaluation
Initializing success.
Controller founded: TiaImportExample.Controllers.HomeController
Controller founded: TiaCompilerCLI.Controllers.TiaApiController
TIA Portal API service started, access address: http://192.168.103.245:9000/
StatusCode: 200, ReasonPhrase: 'OK', Version: 1.1, Content: System.Net.Http.StreamContent, Headers:
{
  Date: Sun, 18 May 2025 14:45:33 GMT
  Server: Microsoft-HTTPAPI/2.0
  Content-Length: 18
  Content-Type: application/json; charset=utf-8
}
HTTP service initialization successful!
Press Enter to exit...
```

> You can test the interface access within the same local area network using `curl http://192.168.103.245:9000/api/home`, which should normally return `Hello, World!`.  
> **Note:** The program will automatically scan for available LAN IP addresses and use them as the listening address, with the default port being `9000`. If using VMWare Workstation, set the virtual machine's network to **Bridged (Physical Direct Connection)** to ensure the HTTP service deployed on the virtual machine is accessible from the host's local area network.