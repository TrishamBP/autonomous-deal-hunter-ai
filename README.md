# The Price is Right – Autonomous AI Agent System

An experimental multi-agent AI framework that automatically detects product deals, estimates fair market prices using LLMs, and notifies users when exceptional deals are found.

This project was built as the capstone project for the AI Engineering course.

The goal is not production deployment but practicing:

- Clean Python architecture
- Object-Oriented Programming
- Agent-based system design
- Retrieval-Augmented Generation (RAG)
- LLM orchestration

---

## 🚀 Quick Start

### Option 1: Local Development (Fastest)

```bash
python launch_app.py
# Open: http://localhost:7860
```

### Option 2: Docker Deployment (Recommended)

```bash
# Copy environment file
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Start with Docker
docker-compose up -d
# Open: http://localhost:7860
```

### Option 3: Automated Docker Setup

**Windows:**

```bash
docker-start.bat
```

**macOS/Linux:**

```bash
bash docker-start.sh
```

---

## 🐳 Docker Deployment

The project is fully dockerized with persistent vector database:

### Features

✅ One-command deployment  
✅ Persistent vector database (survives restarts)  
✅ Auto-initialization on first run  
✅ Sample products auto-loaded  
✅ Production-ready configuration

### Files

- `Dockerfile` - Container image definition
- `docker-compose.yml` - Service orchestration
- `initialize_and_run.py` - Startup script with DB initialization
- `docker-start.sh` / `docker-start.bat` - Automation scripts
- `DOCKER.md` - Comprehensive Docker guide
- `DOCKER_QUICK_REF.py` - Quick reference

### Quick Commands

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```
