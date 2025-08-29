# 使用官方 Python 运行时作为父镜像
FROM python:3.11-slim

# 设置容器中的工作目录
WORKDIR /usr/src/app

# 将依赖文件复制到容器中
COPY requirements.txt ./

# 安装 requirements.txt 中指定的任何所需包
# --no-cache-dir 选项可以减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录的内容复制到容器的 /usr/src/app 中
COPY . .

# docker-compose.yml 文件中会指定运行应用的命令
