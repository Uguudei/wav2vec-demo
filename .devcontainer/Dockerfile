FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set Python timezone
ENV TZ "Asia/Ulaanbaatar"

# Options for setup script
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="true"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Copy library scripts to execute
COPY .devcontainer/library-scripts/*.sh /tmp/library-scripts/

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
RUN bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Setup default python tools in a venv via pipx to avoid conflicts
ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=/usr/local/py-utils/bin
ENV PATH=${PATH}:${PIPX_BIN_DIR}
RUN bash /tmp/library-scripts/python-debian.sh "none" "/opt/conda" "${PIPX_HOME}" "${USERNAME}" "true" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# [Optional] Uncomment this section install Oracle instant client
# RUN bash /tmp/library-scripts/oracle-instant-client-debian.sh \
#     && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Remove library scripts for final image
RUN rm -rf /tmp/library-scripts

# Update, change owner
RUN groupadd -r conda --gid 900 \
    && chown -R :conda /opt/conda \
    && chmod -R g+w /opt/conda \
    && find /opt -type d | xargs -n 1 chmod g+s \
    && usermod -aG conda ${USERNAME}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Copy environment.yml to a temp location to update the environment.
COPY .devcontainer/environment.yml* /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp