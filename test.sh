# 配置Git使用本地SOCKS5代理（假设本地IP为192.168.1.100）
git config --global http.proxy 'socks5://10.62.217.120:1080'
git config --global https.proxy 'socks5://10.62.217.120:1080'

# 验证配置
git config --global --get http.proxy
git config --global --get https.proxy