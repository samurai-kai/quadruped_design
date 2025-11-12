# =====================================================
# ✅ ROS 2 Jazzy Dockerfile (with GUI + RViz + Gazebo math)
# =====================================================

FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root/ros2_ws

# --- Fix ROS repo keys and install base tools ---
RUN rm -f /etc/apt/sources.list.d/ros2-latest.list /usr/share/keyrings/ros2-latest-archive-keyring.gpg || true \
 && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg2 lsb-release python3-pip

# --- Refresh ROS APT source (for Jazzy packages) ---
RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') \
 && curl -L -s -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" \
 && apt-get update && apt-get install -y /tmp/ros2-apt-source.deb \
 && rm -f /tmp/ros2-apt-source.deb

# --- Add Gazebo Harmonic repo (for gz-math8 libs RViz2 needs) ---
RUN curl -sSL http://packages.osrfoundation.org/gazebo.gpg \
      -o /usr/share/keyrings/gazebo-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] \
      http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list

# --- Install ROS 2 visualization and math dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      git \
      ros-jazzy-xacro \
      ros-jazzy-joint-state-publisher-gui \
      ros-jazzy-rviz2 \
      libgz-math8 \
      libgz-math8-dev \
 && rm -rf /var/lib/apt/lists/*

# --- GUI display for Windows (VcXsrv) ---
ENV DISPLAY=host.docker.internal:0.0

# --- Auto-source ROS + workspace on container start ---
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /root/ros2_ws/install/setup.bash ]; then source /root/ros2_ws/install/setup.bash; fi" >> /root/.bashrc

CMD ["bash"]
