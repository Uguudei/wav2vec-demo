#!/usr/bin/env bash

echo 'Installing Oracle Instant Client ...'

# [Optional] Uncomment this section install Oracle instant client
# OS packages necessary for Oracle instant client
apt-get update && export DEBIAN_FRONTEND=noninteractive && \
apt-get -y install --no-install-recommends libaio1 && \
apt-get -y clean

# Add Oracle instant client location to Path
export PATH=/opt/oracle/instantclient_21_3:$PATH

# Set LD Library Path
export LD_LIBRARY_PATH=/opt/oracle/instantclient_21_3${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "setting env variable LD_LIBRARY_PATH=$LD_LIBRARY_PATH in /etc/bash.bashrc"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /etc/bash.bashrc

# Set Oracle language settings
export NLS_LANG="AMERICAN_AMERICA.AL32UTF8"
echo "setting env variable NLS_LANG=$NLS_LANG in /etc/bash.bashrc"
echo "export NLS_LANG=$NLS_LANG" >> /etc/bash.bashrc

# Install Oracle instant client
mkdir -p /opt/oracle && rm -rf /opt/oracle/* &&cd /opt/oracle && \
curl -SL "https://download.oracle.com/otn_software/linux/instantclient/213000/instantclient-basic-linux.x64-21.3.0.0.0.zip" -o instant_client.zip && \
unzip instant_client.zip && rm instant_client.zip

# [Optional] Uncomment this section to install Oracle SQL Plus.
# cd /opt/oracle && \
# curl -SL "https://download.oracle.com/otn_software/linux/instantclient/213000/instantclient-sqlplus-linux.x64-21.3.0.0.0.zip" -o sqlplus.zip && \
# unzip sqlplus.zip && rm sqlplus.zip

echo "Done!"