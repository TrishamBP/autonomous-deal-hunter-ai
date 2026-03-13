# Development Guide

This document defines the development rules and architecture for the **Price Is Right Agent System**.

This guide must be followed by all contributors and coding agents (such as Cursor).

The repository owner is the **final authority on architectural decisions**.

No architectural changes should be made without approval.

---

# Project Goal

The system builds an **autonomous agent framework** that finds undervalued products online.

The pipeline performs the following steps:

1. Discover deals from online sources
2. Retrieve product descriptions
3. Estimate real market value using AI
4. Compare estimated price with listed price
5. Store high-quality opportunities

The system combines:

- AI price estimation
- Retrieval-Augmented Generation (RAG)
- Vector search
- Agent orchestration
- Memory of past opportunities

---

# Core Architecture

The system follows a **centralized agent framework architecture**.

```text
DealAgentFramework
        │
        ▼
Planning Agent
        │
        ▼
Deal Retrieval
        │
        ▼
Price Estimation Model
        │
        ▼
Opportunity Evaluation
        │
        ▼
Memory + Vector Database
```

The framework is responsible for:

- initializing agents
- maintaining system memory
- orchestrating execution
- storing results

---

# Project Structure

The repository should follow this structure:

```
project_root/

agents/
    planning_agent.py
    deals.py
    messenger_agent.py

services/
    pricer_service.py
    retrieval_service.py

framework/
    deal_agent_framework.py

models/
    opportunity.py
    deal.py

memory/
    memory.json

utils/
    log_utils.py
    helpers.py

scripts/
    run_framework.py

ui/
    gradio_app.py
```

---

# Agent System

Agents are responsible for **decision making and orchestration**.

Agents must:

- be implemented as Python classes
- encapsulate reasoning logic
- interact with services

Example agent types:

PlanningAgent
MessengerAgent
OpportunityAgent

Agents should **not contain infrastructure code**.

---

# Services

Services perform external operations such as:

- model inference
- vector database access
- web scraping
- notifications

Example services:

PricerService
VectorStoreService

Services should contain **pure functionality**.

They should not make planning decisions.

---

# Memory System

The framework maintains a persistent memory file.

```
memory.json
```

The memory stores previously discovered opportunities.

Example memory structure:

```
{
  "deal": {...},
  "estimate": 900,
  "discount": 200
}
```

The memory allows the system to:

- track past opportunities
- prevent duplicate alerts
- analyze historical deals

The framework loads and saves this memory automatically.

---

# Vector Database

The system uses **ChromaDB** as a vector store.

The vector database stores:

- product descriptions
- embeddings
- metadata

This enables semantic retrieval during price estimation.

Example use case:

Find similar products to estimate market value.

---

# Price Estimation Model

Price estimation is handled by a fine-tuned LLM.

The model predicts the expected market price given a product description.

Example prompt:

```
What does this cost to the nearest dollar?

<product description>

Price is $
```

The model output is parsed to extract the numeric price.

The system compares:

```
Estimated Price - Listed Price
```

If the difference is large enough, the deal is considered an opportunity.

---

# Framework Responsibilities

The main framework class handles:

- logging
- agent initialization
- memory loading
- agent execution
- storing results

Typical execution flow:

```
Framework starts
      │
Load memory
      │
Initialize agents
      │
Run planning agent
      │
Evaluate opportunity
      │
Store opportunity in memory
      │
Return updated opportunities
```

---

# Logging System

Logging should follow a centralized format.

Example:

```
[2026-01-01 10:00:00] [Agents] [INFO] Starting planning agent
```

Logs should provide:

- agent activity
- reasoning steps
- pipeline progress

Logs should **not contain sensitive data**.

---

# UI Layer

The system optionally includes a **Gradio interface**.

The UI provides:

- live logs
- discovered deals
- price estimates
- visualization of vector database

The UI is not part of the core agent system.

It should remain isolated from core logic.

---

# Development Rules

All code must follow these rules.

### Follow the architecture

Do not introduce new architectural patterns without approval.

### Write modular code

Files should be small and focused.

### Use clear naming

Avoid abbreviations and unclear variable names.

### Keep agents lightweight

Agents coordinate actions but should not implement heavy logic.

### Keep services pure

Services should only perform operations.

---

# Code Review Policy

All structural changes must be approved by the project owner.

Examples of changes requiring approval:

- new agent types
- new architecture layers
- replacing the framework
- adding distributed systems

---

# Future Improvements

Potential future improvements include:

- reinforcement learning for deal ranking
- better retrieval pipelines
- multi-agent collaboration
- automated evaluation of pricing accuracy

---

# Summary

This project demonstrates an **AI agent system for detecting undervalued products online**.

The system combines:

- agent orchestration
- vector search
- LLM-based price estimation
- persistent memory
- modular Python architecture

The goal is to build a **clean and extensible AI engineering system**.
