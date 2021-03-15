#!/bin/bash

### コンテナ実行時のroot権限剥奪 (Ubuntu16.04)
### 参考：http://qiita.com/kjtanaka/items/c461bc50fc0cc9c248c9
###     ：http://unskilled.site/ユーザー名前空間でdockerコンテナのrootとホストのユー/
### Yousuke 2017/4/22

# ユーザ追加
useradd dockremap
# ユーザー設定
sh -c 'echo dockremap:10000:65536 >> /etc/subuid'
sh -c 'echo dockremap:100:65536 >> /etc/subgid'
# docker.serviceの編集
sed -ie '15i DOCKER_OPTS="--userns-remap=default"' /etc/default/docker
# docker デーモンの再起動
restart docker

### 注意：このシェルスクリプトはroot権限で実行する必要がある
