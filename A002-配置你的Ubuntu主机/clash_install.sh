# clash vpn
version="v1.8.0"
mmdb_url="https://dl.nesnode.com/n2ray/Country.mmdb"
clash_url="https://dl.nesnode.com/n2ray/clash-linux-amd64-"$version".gz"

# 注意：需要手动将config.yaml文件拖到/etc/clash文件夹中
# config文件可以从windos端的clash app导出

# 1. 下载Clash二进制文件clash.gz
mkdir /etc/clash
echo '下载 Clash 二进制文件...'
wget -q -O /tmp/clash.gz $clash_url

if [ $? != 0 ] ; then
    echo -e "获取 Clash 二进制文件失败！"
    exit 0;
fi

# 2. 下载Clash国家配置文件Country.mmdb
wget -q -O /etc/clash/Country.mmdb $mmdb_url

# 3. 配置Clash
systemctl stop clash
gunzip -f -c /tmp/clash.gz > /usr/bin/clash
chmod a+x /usr/bin/clash
systemctl daemon-reload

echo "配置 systemctl clash 服务"

# 需要将User改成你的主机用户名
echo "[Unit]
Description=seuiv clash

[Service]
Type=simple
User=seuiv
ExecStart=/usr/bin/clash -d /etc/clash/
Restart=on-failure

[Install]
WantedBy=multi-user.target" > /etc/systemd/system/clash.service

# 4. 设置Clash开机自启动
systemctl enable clash
systemctl stop clash
systemctl start clash
sleep 3s

# 5. 检查Clash状态
systemctl status  clash --no-pager -l

