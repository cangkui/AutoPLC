import json
text = f"""
	- 变量定义
 代码中定义了以下几类变量：
 - VAR_INPUT：包含功能块的输入参数，例如执行标志、数据长度匹配标志和文件名。
 - VAR_OUTPUT：包含功能块的输出参数，例如完成标志、忙标志、错误标志、状态字、数据长度和诊断信息。
 - VAR_IN_OUT：包含在功能块调用期间可能会被修改的参数，例如缓冲字节数组和数据变体。
 - VAR：包含功能块内部使用的变量，例如状态、文件读取实例和辅助状态信息。
 - VAR_TEMP：包含功能块执行过程中临时使用的变量。
 - VAR_CONSTANT：包含功能块中使用的常量。
	- 主逻辑部分
-- REGION BLOCK INFO HEADER
   包含功能块的基本信息和版权声明，没有执行逻辑。
 
-- REGION DESCRIPTION
   为空，没有执行逻辑或描述信息。
 
-- REGION TRIGGERING
   初始化处理流程，当检测到 execute 输入由 FALSE 变为 TRUE 时，开始新的处理流程：
   - 使用临时变量 #tempExecute 保存输入 execute 的值，以避免在处理过程中发生变化。
   - 检查是否是第一次调用（由 #statInternal.executePrevious 和 #statInternal.diagnostics.status 判断）。
   - 如果是第一次调用，则初始化内部状态变量，设置状态机到 FB_STATE_INIT 状态，并准备调用文件读取实例（instFileRead）。
   - 更新 #statInternal.executePrevious 以检测下次调用时的状态变化。
 
-- REGION STATE_MACHINE
   根据 #statMainState 的值执行不同的处理逻辑：
   - FB_STATE_NO_PROCESSING：此状态为空，不执行任何操作。
   - FB_STATE_INIT：
     - Verify File Path：检查输入的文件名是否以预定义的路径前缀开头。
     - Verify buffer boundary：检查缓冲数组的边界，确保下界为0，并计算缓冲区大小。
     - Verify parameter data types：根据参数 data 的类型进行不同的处理，如果 data 是字节数组，则直接使用，否则序列化 data 到缓冲数组。
     - File handler init：初始化文件读取实例 instFileRead，设置读取长度，并尝试初始化，如果成功则进入 FB_STATE_FILE_READ 状态。
   - FB_STATE_FILE_READ：
     - File handling：调用 instFileRead 读取文件，处理读取完成或错误的情况。如果读取成功，根据数据长度匹配标志检查文件大小，如果匹配则进入 FB_STATE_PARAMETER_TO_INTERFACE 状态。
   - FB_STATE_PARAMETER_TO_INTERFACE：
     - Parameter handling：根据参数类型，将缓冲数组中的数据复制到参数 data 中，或者从缓冲数组反序列化数据到参数 data。
 
-- REGION OUTPUTS
   更新功能块的输出参数：
   - 根据内部状态诊断字 #statInternal.diagnostics.status，设置输出 done、busy、error 和 status。
   - 如果执行完成且没有错误，则设置 done 为 TRUE，并重置 busy 和 error。
   - 如果发生错误，则设置 error 为 TRUE，并将诊断信息输出到 diagnostics。
   - 如果 execute 输入为 FALSE 且之前的状态为完成或错误，则重置所有输出和内部状态。
   - 设置数据长度 output dataLength。
   - 设置ENO（错误输出）为内部错误状态。"""
print(json.dumps({'text':text},ensure_ascii=False))