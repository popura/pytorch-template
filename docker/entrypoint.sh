#!/bin/bash

USER=${USER_NAME:-user}
GROUP=${GROUP_NAME:-Domain}
USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"
groupadd -g $GROUP_ID $GROUP
useradd -u $USER_ID -g $GROUP_ID -o $USER
export HOME=/home/$USER

exec /usr/sbin/gosu $USER "$@"
