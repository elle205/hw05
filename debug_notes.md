# 作业调试记录

## 问题1：VS Code 无法识别 Python 环境
- 现象：输入 `python` 命令无反应，右下角提示 `Select Python Interpreter`
- 解决：选择 Anaconda 的 `base` 环境，并安装 `torch` 等依赖库

## 问题2：代码运行无输出
- 现象：运行 `simple_cnn.py` 后终端无任何打印
- 解决：检查文件是否完整，重新复制完整代码并保存，按 `Ctrl+S` 确认保存

## 问题3：`git push` 认证失败
- 现象：`fatal: Authentication failed` 无法上传到 GitHub
- 解决：更换为官方 GitHub 仓库地址，配置 `user.name` 和 `user.email` 后重新提交