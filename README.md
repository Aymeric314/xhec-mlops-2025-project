<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to your MLOps project! In this hands-on project, you'll build a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Your Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Your Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python 3.10 or 3.11

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Set up your development environment**:
   ```bash
   # Create and activate a virtual environment
   uv sync
   source venv/bin/activate # on Windows: venv\Scripts\activate

   # Install pre-commit hooks for code quality
    uv pip install pre-commit
    uv run pre-commit install
   ```

   One-liner setup:
   ```bash
   uv sync --dev && uv run pre-commit install
   ```

## 📋 What You'll Build

By the end of this project, you'll have created:

### 🤖 **Automated ML Pipeline**
- Training workflows using Prefect
- Automatic model retraining on schedule
- Reproducible model and data processing

### 🌐 **Prediction API**
- REST API for real-time predictions
- Input validation with Pydantic
- Docker containerization

### 📊 **Production-Ready Code**
- Clean, well-documented code
- Automated testing and formatting
- Proper error handling

## 📝 How to Work on This Project

### The Branch-by-Branch Approach

This project is organized into numbered branches, each representing a step in building your MLOps system. Think of it like a guided tutorial where each branch teaches you something new!

**Here's how it works**:

1. **Each branch = One pull request** with specific tasks
2. **Follow the numbers** (branch_0, branch_1, etc.) in order
3. **Read the PR instructions** (PR_0.md, PR_1.md, etc.) before starting
4. **Complete all TODOs** in that branch's code
5. **Create a pull request** when done
6. **Merge and move to the next branch**

### Step-by-Step Workflow

For each numbered branch:

```bash
# Switch to the branch
git checkout branch_number_i

# Get latest changes (except for branch_1)
git pull origin main
# Note: A VIM window might open - just type ":wq" to close it

# Push your branch
git push
```

Then:
1. 📖 Read the PR_i.md file carefully
2. 💻 Complete all the TODOs in the code
3. 🔧 Test your changes
4. 📤 Open **ONE** pull request to your main branch
5. ✅ Merge the pull request
6. 🔄 Move to the next branch

> **💡 Pro Tip**: Always integrate your previous work when starting a new branch (except branch_1)!

### 🔍 Understanding Pull Requests

Pull Requests (PRs) are how you propose and review changes before merging them into your main codebase. They're essential for team collaboration!

**Important**: When creating a PR, make sure you're merging into YOUR forked repository, not the original:

❌ **Wrong** (merging to original repo):
![PR Wrong](assets/PR_wrong.png)

✅ **Correct** (merging to your fork):
![PR Right](assets/PR_right.png)

## 🚀 Running the Prefect Workflows

### Prerequisites
Make sure you have installed all dependencies:
```bash
uv sync
```

### Running the Training Flow

You can run the abalone training flow in several ways:

#### 1. **Direct Flow Execution**
```bash
# Navigate to the modelling directory
cd src/modelling

# Run the flow directly
python -c "from main import training_flow; training_flow('../../data/abalone.csv')"
```

#### 2. **Using Prefect CLI**
```bash
# Start Prefect server (in one terminal)
prefect server start

# Run the flow using Prefect CLI
prefect flow run "training_flow" --param trainset_path="data/abalone.csv"
```

#### 3. **Create and Run Deployments**
```bash
# Create and serve deployment (runs daily every 24 hours)
cd src/modelling
uv run python deployment.py
```

### 🖥️ **Prefect UI - Monitoring and Visualization**

The Prefect UI provides a powerful web interface to monitor your flows, view run history, and manage deployments.

#### Starting the Prefect Server and UI

1. **Start the Prefect server**:
   ```bash
   uv run prefect server start
   ```
   This will start the server on `http://localhost:4200`

2. **Open the Prefect UI**:
   - Navigate to `http://localhost:4200` in your browser
   - You'll see the Prefect dashboard with all your flows and deployments

#### Key Features in the Prefect UI

- **📊 Flow Runs**: View detailed logs and metrics for each flow execution
- **⏰ Deployments**: Manage scheduled deployments and trigger manual runs
- **📈 Flow Run History**: Track performance over time
- **🔍 Task Details**: Drill down into individual task execution
- **📋 Logs**: Real-time and historical logs for debugging

#### Work Pool Management

To run scheduled deployments, you need a work pool agent:

```bash
# Start an agent for the default work pool
uv run prefect agent start --pool default-agent-pool
```

### 📅 **Scheduled Retraining**

The project includes a deployment that runs:
- **Daily Retraining**: Runs every 24 hours automatically
- **Manual Triggers**: Can be triggered on-demand through the Prefect UI

### 🔧 **Configuration and Customization**

You can customize the deployment schedule by modifying `src/modelling/deployment.py`:

```python
# Example: Change to run every 6 hours (360 minutes)
interval=360

# Example: Change to run every 12 hours (720 minutes)
interval=720

# Example: Change to run every week (10080 minutes)
interval=10080
```

### 🐛 **Troubleshooting**

**Common Issues:**

1. **Port already in use**: If port 4200 is busy, Prefect will automatically use the next available port
2. **Agent not picking up jobs**: Make sure the agent is running and connected to the correct work pool
3. **Flow not found**: Ensure you're in the correct directory and have imported the flow properly

**Useful Commands:**
```bash
# Check Prefect server status
uv run prefect server status

# List all flows
uv run prefect flow ls

# List all deployments
uv run prefect deployment ls

# View flow run logs
uv run prefect flow-run logs <flow-run-id>
```

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies. Install or update packages with:

```bash
uv add <package>==<version>
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### Code Quality
- The pre-commit hooks will automatically format your code
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings

## 📊 Evaluation Criteria

Your project will be evaluated on:

### 🔍 **Code Quality**
- Clean, readable code structure
- Proper naming conventions
- Good use of docstrings and type hints

### 🎨 **Code Formatting**
- Consistent style (automated with pre-commit)
- Professional presentation

### ⚙️ **Functionality**
- Code runs without errors
- All requirements implemented correctly

### 📖 **Documentation & Reproducibility**
- Clear README with setup instructions
- Team member names and GitHub usernames
- Step-by-step instructions to run everything

### 🤝 **Collaboration**
- Effective use of Pull Requests
- Good teamwork and communication

---

## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [x] Prefect workflows for model training (flows and tasks)
- [x] Separate modules for training and inference
- [x] Reproducible model and encoder generation
- [x] Named tasks for better monitoring

✅ **Automated Deployment**
- [x] Prefect deployment for regular retraining (daily schedule)
- [x] Cron-based scheduling for automated retraining
- [x] Work pool management for scalable execution

✅ **Production API**
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [ ] Updated README with team info
- [ ] Clear setup and run instructions
- [ ] All TODOs removed from code

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
