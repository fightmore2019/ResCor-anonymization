<div align="center">   
  
# ResCor: Residual Correct Safe Reinforcement Learning for Multi-agent Safety-critical Scenarios
</div>


# Abstract
Low-frequency but high-risk safety-critical events are the root of the long-tail effect in autonomous driving. When individual intelligence expands to multi-agent safety-critical scenarios (MASCS), their complexity surpasses single-agent theory and demands new collective-decision paradigms. Existing safe reinforcement learning (RL) methods, ranging from explicit safety masks to implicit Lagrangian constraints, improve ssfety yet either lock policies inside rigid safe sets or degrade efficiency through poor constraint representation. We introduce weak-to-strong generalisation into safe RL for the first time and propose **ResCor**, a **res**idual **cor**rection safe reinforcement learning framework that raises both safety and performance in MASCS. ResCor combines three modules: (i) a mixed-action policy that adaptively calibrates safety boundaries, (ii) a multi-agent dynamic conflict zone that captures and quantifies risk interactions, and (iii) a risk-aware prioritised experience replay that focuses learning on rare, high-risk events. Notably, the lightweight ResCor uses only 10% of the main model’s parameters yet provides decisive safety guidance without hindering task efficiency. Experiments on five MASCS show that ResCor cuts collision rates by up to **90.6\%** and increases cumulative rewards by 51.6% over strong baselines. It also transfers smoothly to unseen, more complex MASCS, highlighting its promise as a lightweight pre-trained safety module for real-world autonomous driving. 


# Methods
![method](Figs/frame.png "model frame")


# Demonstration Video
### 1. HardBrake (2AVs)
https://github.com/user-attachments/assets/8b60a34a-82f8-4278-b703-b19d1efa959b

### 2. ParkingCrossingPed (2AVs)
https://github.com/user-attachments/assets/3cc03c36-0208-4cd7-9be1-9347349fa352

### 3. ParkingCutIn
https://github.com/user-attachments/assets/b3989b14-602d-41e6-b8cb-a36475f3be5b

### 4. DynamicCutIn
https://github.com/user-attachments/assets/892320d2-c43a-4ba8-9efd-59e575912c0a

### 5. IntersectionViolationPeds
https://github.com/user-attachments/assets/15dbce7c-8780-45ac-b6e7-2845707db8d9

### 6. HardBrake (3AVs)
https://github.com/user-attachments/assets/eaf87f40-7e76-41cb-80a7-cb8ab4c01102

### 7. ParkingCrossingPed (3AVs)
https://github.com/user-attachments/assets/128500ed-3f3e-40e3-978f-bcbdd0c88790


# Installation
```shell
# Clone the code to local
git clone https://github.com/fightmore2019/ResCor-anonymization.git
cd ResCor

# Create virtual environment
conda create -n ResCor python=3.7
conda activate ResCor

# Install basic dependency
pip install -r requirements.txt
```
