# 使用官方 Python 3.8 Slim Buster 镜像作为基础镜像，它比完整版更小
FROM python:3.8-slim-buster

# 设置工作目录为 /app
WORKDIR /app

# 复制 requirements.txt 文件到工作目录
COPY requirements.txt .

# ====================================================================
# 关键修复步骤：
# 1. 明确安装 PyTorch (CPU 版本)
# 2. 安装 requirements.txt 中的其他 Python 依赖
# 增加 pip 下载超时时间，并更换为国内的 PyPI 镜像源，以提高下载速度和稳定性。
# 这里使用清华大学的 PyPI 镜像源，您可以根据需要更换为其他源 (如阿里云、豆瓣等)。
# ====================================================================
RUN pip install --no-cache-dir --default-timeout=600 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --default-timeout=600 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# ====================================================================
# HanLP 模型部署：
# 假设您已将下载的 HanLP 模型文件放置在宿主机项目根目录下的 'hanlp_data' 文件夹中
# 例如：your_project_root/hanlp_data/.hanlp/model/tok/open_tok_pos_ner_srl_dep_sdp_con_electra_base_20201223_201906/...
# 这个 COPY 指令会将宿主机的 hanlp_data 文件夹及其内容复制到容器的 /app/hanlp_data 路径下
# ====================================================================
COPY hanlp_data /app/hanlp_data

# 设置 HanLP_HOME 环境变量，指向容器内 HanLP 模型数据的位置
# 这与 app.py 中的 os.getenv('HANLP_HOME', ...) 逻辑相对应
ENV HANLP_HOME=/app/hanlp_data

# 复制应用程序的所有文件到容器的工作目录
COPY . .

# 暴露 Flask 应用程序将监听的端口
EXPOSE 5000

# 定义容器启动时执行的命令
# 生产环境中推荐使用 Gunicorn 等 WSGI 服务器，这里为了简化部署直接用 Flask 内置服务器
CMD ["python", "app.py"]
