document.addEventListener('DOMContentLoaded', function() {
    // 示例：文件选择后，可以在这里添加一些前端验证或提示
    const fileInput = document.getElementById('files');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const files = this.files;
            if (files.length > 0) {
                console.log(`选择了 ${files.length} 个文件。`);
                // 可以在这里显示文件名列表
            } else {
                console.log('未选择文件。');
            }
        });
    }

    // 示例：如果下载ZIP的按钮是独立的，可以在这里监听点击事件
    const downloadAllBtn = document.querySelector('.download-btn-all');
    if (downloadAllBtn) {
        downloadAllBtn.addEventListener('click', function(event) {
            // 对于直接指向 Flask 路由的下载链接，通常不需要阻止默认行为
            // 但如果需要执行额外的JS逻辑，比如显示加载动画，可以在这里添加
            console.log('开始下载所有文件...');
            // event.preventDefault(); // 如果需要自定义下载逻辑，则取消注释
        });
    }

    // 可以在这里添加更多前端交互逻辑
});