FROM astral/uv:0.8.22-python3.10-bookworm-slim
WORKDIR /workspaces/pathpyG

# Install git (and gpg for signed commits: https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials#_sharing-gpg-keys)
RUN apt-get update && apt-get -y install git gnupg2

# Install dependencies for .svg support in tikz
RUN apt-get update && apt-get -y install ghostscript

# Install dependencies for .svg support in tikz
RUN apt update && apt install -y ghostscript

# Install dependencies for manim
RUN apt-get update && apt-get -y install build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg
