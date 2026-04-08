# Senior Design: "Third Arm" Project (SO-ARM 101)

## 🎯 Project Goals
- **Autonomous Picking:** Detect and pick objects (baby bottles, toys) from the ground using a stroller-mounted robotic arm.
- **Dynamic Safety Avoidance:** Real-time tracking of a baby's hand using MediaPipe and avoiding contact/collision using Artificial Potential Fields (APF).
- **HCI (Human-Computer Interaction):** Smoothly transition between picking tasks and safety-priority modes.

## 🛠️ System Architecture
- **Hardware:** SO-ARM 101 (6-DOF Arm) with Feetech STS3215 servos.
- **Simulation:** MuJoCo (Physics-based verification).
- **Vision:** 
    - YOLOv8/v10 (Object Detection)
    - MediaPipe (Hand Tracking/Avoidance)
- **Control:** Python-based IK (`ikpy`) + Rule-based FSM + APF.

## 📂 Workspace Structure (~/senior_design/)
- `urdf/`: Robot description files (so101.urdf).
- `meshes/`: STL files for 3D visualization.
- `simulation/`: MuJoCo scene files and `run_sim.py` execution script.

## 🚀 Progress Tracking
- [x] Workspace initialized with URDF/STL assets.
- [x] MuJoCo simulation environment set up.
- [x] **Axis Visualization:** Identified RGB color coding (R:X, G:Y, B:Z).
- [x] **L515 Camera Integration:**
    - Modeled L515 using actual dimensions (61mm dia, 26mm thickness).
    - Optimized placement: Moved to the end of the `lower_arm` (forearm) to reduce wrist servo load and ensure stable vision.
    - Set 8cm offset to account for a realistic 3D-printed mount and avoid mesh collisions.
- [ ] Implement IK-based joint control in simulation.
- [ ] Add dynamic obstacle (baby hand sphere) and avoidance logic.
- [ ] Integrate YOLO/MediaPipe vision pipeline.

---
*Last Updated: 2026-04-08*
