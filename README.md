字幕智能分割工具 (SRT Splitter App)
📝 简介
这是一个基于 Flask 框架和 HanLP 中文自然语言处理库构建的智能字幕（SRT 格式）分割工具。它旨在解决长字幕行在播放时显示不佳的问题，通过结合智能分词和字符宽度计算，自动将 SRT 文件分割成更符合阅读习惯的短行，并支持去除行末标点、文件持久化等功能。

✨ 主要特点
智能断句： 利用 HanLP 进行中文分词，在语义上更合理的位置进行断句。

字符宽度计算： 精确计算中英文字符的显示宽度（汉字占2个字符，英文/数字/半角符号占1个），确保每行符合设定的最大显示宽度。

自定义长度：： 用户可配置每行最大字符数。

标点处理： 可选择性移除行末的句号或逗号。

批量处理： 支持同时上传和处理多个 SRT 文件。

文件持久化： 处理后的文件将保留在 processed 目录下，不会自动删除，方便反复查看和下载。

Web 界面和 API 支持： 提供直观的 Web 上传界面，同时提供 RESTful API 接口，方便集成到其他系统。

Docker 部署： 提供 Docker 支持，简化环境配置和部署过程。

🛠️ 技术栈
后端： Python 3.8+

Web 框架： Flask

自然语言处理： HanLP (用于中文分词和智能断句)

其他库： Werkzeug (文件上传), filelock (HanLP模型下载锁), zipfile (文件压缩), shutil (文件操作)

前端： HTML, Tailwind CSS (用于美观的用户界面)

容器化： Docker, Docker Compose

🚀 部署方式
本项目推荐使用 Docker 进行部署，以简化环境配置。

前提条件
已安装 Docker 和 Docker Compose。

已安装 Git。

1. 克隆仓库
git clone https://github.com/AlixMeng/srt-splitter-app.git
cd srt-splitter-app

2. 准备 HanLP 模型数据
HanLP 模型文件通常较大，不直接包含在 Git 仓库中。您需要手动下载并将其放置到 hanlp_data 文件夹内，使其结构如下：

srt-splitter-app/
└── hanlp_data/
    └── .hanlp/
        └── model/
            └── tok/
                └── open_tok_pos_ner_srl_dep_sdp_con_electra_base_20201223_201906/
                    └── config.json
                    └── ... (其他模型文件，如 pytorch_model.bin)
    └── transformers/
        └── electra_zh_base_20210706_125233.zip (或其他transformer模型)
    └── thirdparty/
        └── file.hankcs.com/
            └── corpus/
                └── char_table.json.zip

请确保所有必要的 HanLP 模型文件（特别是 open_tok_pos_ner_srl_dep_sdp_con_electra_base_20201223_201906 文件夹内的内容，以及 electra_zh_base_20210706_125233.zip 和 char_table.json.zip）都正确地解压或放置在 hanlp_data 目录下。

您可以从 HanLP 官方网站或其 GitHub 仓库提供的下载链接获取这些模型。

3. 构建并运行 Docker 容器
在项目根目录下，使用 Docker Compose 构建镜像并启动服务。默认情况下，应用将在宿主机的 5173 端口上运行。

# 停止并移除所有旧容器和镜像，确保环境干净
docker-compose down -v --rmi all

# 强制重新构建 Docker 镜像，确保所有文件（包括 HanLP 数据）被正确复制
docker build --no-cache -t srt-splitter-app .

# 启动容器
docker-compose up -d

4. 访问应用
应用启动后，您可以通过浏览器访问：

http://<您的主机IP地址>:5173/

例如：http://192.168.50.168:5173/

👨‍💻 API 使用
您也可以通过发送 POST 请求到 /api/process_subtitles 接口来处理字幕文件，这方便您将此功能集成到其他自动化流程中。

请求详情
URL: /api/process_subtitles

方法: POST

Content-Type: multipart/form-data

表单字段 (Form Fields)
files: (文件, 必选) 一个或多个 .srt 格式的字幕文件。

max_chars_per_line: (数字, 可选) 每行最大显示字符数。默认值为 30。范围：10 到 100。

remove_punctuation_checkbox: (字符串, 可选) 是否移除行末句号或逗号。如果需要移除，请设置为 'on'。默认值为 'on'。

示例 (使用 curl)
curl -X POST -F "files=@/path/to/your/subtitle.srt" \
     -F "max_chars_per_line=40" \
     -F "remove_punctuation_checkbox=on" \
     http://<您的主机IP地址>:5173/api/process_subtitles \
     -o processed_subtitles.zip

注意：

将 /path/to/your/subtitle.srt 替换为实际的 SRT 文件路径。

将 <您的主机IP地址> 替换为您的 Docker 宿主机的 IP 地址。

-o processed_subtitles.zip 会将处理后的 ZIP 文件保存到本地。如果未提供，响应内容将直接输出到标准输出。

许可证
本项目采用 MIT 许可证。详见 LICENSE 文件。
