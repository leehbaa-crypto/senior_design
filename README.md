# 🤖 SO-ARM 101: "Third Arm" Project

This repository contains the simulation and control framework for the **SO-ARM 101**, a 6-DOF robotic arm. It features real-time hand/face tracking via MediaPipe and collision avoidance using Artificial Potential Fields (APF).

---

## 🛠️ System Specifications

### **Hardware Requirements**
- **Robot Arm:** SO-ARM 101 (6-DOF)
- **Actuators:** Feetech STS3215 Serial Bus Servos
- **Sensors:** **Intel RealSense L515** (Lidar Camera)
  - Recommended firmware: 01.05.08.01 or higher.

### **Software Stack & Verified Versions**
To ensure compatibility, the following versions were used during development:
- **Operating System:** Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Python:** 3.10.19
- **RealSense SDK:** `pyrealsense2` v2.53.1.4623
- **MediaPipe:** v0.10.15 (Supports Hands & 478-landmark Face Mesh)
- **MuJoCo:** v3.0.0+
- **IK Engine:** `ikpy` v3.3.3
- **ROS 2:** Humble Hawksbill (Used for inter-node communication)

---

## 🚀 Getting Started

### **1. Prerequisites**
Ensure you have a ROS 2 Humble environment set up.

### **2. Installation**
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd senior_design
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### **3. Running the Simulation**

1. **Start the Vision Node (Terminal 1):**
   ```bash
   # This will track your hands and face mesh using L515
   python3 simulation/face.py
   ```

2. **Start the Controller (Terminal 2):**
   ```bash
   # This will run the MuJoCo simulation with IK and APF avoidance
   python3 simulation/run_sim_avoidance_v2.py
   ```

---

## 📂 Project Structure
- `urdf/`: Robot description files including `so101.urdf`.
- `meshes/`: 3D STL assets for visualization.
- `simulation/`: 
  - `face.py`: RealSense + MediaPipe (Hands & Face Mesh) tracking.
  - `run_sim_avoidance_v2.py`: Main simulation with IK & APF Avoidance.
  - `scene.xml`: MuJoCo scene definition.
  - `so101_new_calib.xml`: Calibrated MuJoCo model.

---

## 📈 Current Progress
- [x] **High-fidelity Face Mesh:** Tracking 478 landmarks for detailed facial interaction.
- [x] **Inverse Kinematics:** Integrated `ikpy` for precise end-effector goal reaching.
- [x] **APF-based Avoidance:** Real-time repulsive force calculation based on hand proximity.
- [x] **Dynamic Camera-World Sync:** Automated coordinate transformation using MuJoCo's camera pose.

---

## 📝 Authors & Research
- **Researcher:** Lee Hyun-bin
- **Project:** 2026 Senior Design / URP (User Robot Program)
- **Institution:** (Please add your University name)
