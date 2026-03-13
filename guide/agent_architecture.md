# Agent Architecture Guide

This document defines the **multi-agent architecture** used in the project.

The system follows a **Planner → Worker → Verifier architecture**, a common pattern used in modern AI agent frameworks.

This design allows agents to collaborate, improving reliability and decision quality.

---

# 1. Why Multi-Agent Architecture?

Simple pipelines often fail because:

- LLM outputs can be unreliable
- One step may introduce incorrect reasoning
- There is no validation of results

A **multi-agent architecture** solves this by introducing different roles.

Each agent has a specific responsibility.

---

# 2. Architecture Overview

The system follows this structure:

```
Data Sources
    ↓
Planner Agent
    ↓
Worker Agents
    ↓
Verifier Agent
    ↓
Final Decision
```

Each stage improves the output before passing it forward.

---

# 3. Agent Roles

The system includes three main agent types.

### Planner Agent

The planner determines:

- what tasks need to be executed
- what data needs to be gathered
- which worker agents should run

Example responsibilities:

- parse deal feeds
- select products to evaluate
- assign price estimation tasks

Example output:

```
Task Plan:

1. Extract deals from RSS feed
2. Scrape product page
3. Estimate product value
4. Compare market price vs listed price
```

---

### Worker Agents

Worker agents perform the **actual operations**.

Examples:

**DealFinderAgent**

Responsibilities:

- monitor RSS feeds
- extract deal listings

**ScraperAgent**

Responsibilities:

- fetch product pages
- extract product details

**PriceEstimatorAgent**

Responsibilities:

- estimate fair market value using LLMs or models

**RAGAgent**

Responsibilities:

- retrieve historical price context
- enrich product information

Workers should be **small and focused**.

---

### Verifier Agent

The verifier ensures outputs are reasonable.

Responsibilities:

- validate estimated price
- check deal confidence
- remove unreliable predictions

Example verification checks:

- unrealistic price estimates
- incomplete product data
- hallucinated information

Example decision:

```
Listed Price: $300
Estimated Price: $320
Confidence: Low

Result: Reject opportunity
```

---

# 4. System Flow

The system flow becomes:

```
RSS Feeds
   ↓
Planner Agent
   ↓
DealFinderAgent
   ↓
ScraperAgent
   ↓
PriceEstimatorAgent
   ↓
RAGAgent
   ↓
VerifierAgent
   ↓
OpportunityAgent
```

The verifier acts as a **quality control layer**.

---

# 5. Agent Base Class

All agents inherit from a base class.

Example:

```
class BaseAgent:

    def run(self, input_data):
        raise NotImplementedError
```

Agents should only implement their **specific responsibility**.

---

# 6. Planner Agent Example

```
class PlannerAgent(BaseAgent):

    def run(self, rss_items):

        plan = []

        for item in rss_items:
            plan.append({
                "task": "evaluate_deal",
                "url": item.url
            })

        return plan
```

The planner determines **what work needs to happen**.

---

# 7. Worker Agent Example

```
class PriceEstimatorAgent(BaseAgent):

    def run(self, product):

        estimated_price = self.model.predict(product)

        return {
            "product": product,
            "estimated_price": estimated_price
        }
```

Workers should be **deterministic when possible**.

---

# 8. Verifier Agent Example

```
class VerifierAgent(BaseAgent):

    def run(self, deal):

        listed_price = deal["listed_price"]
        estimated_price = deal["estimated_price"]

        discount_ratio = estimated_price / listed_price

        if discount_ratio < 1.1:
            return None

        return deal
```

This removes weak opportunities.

---

# 9. Benefits of this Architecture

Advantages include:

### Reliability

Verification prevents low-quality outputs.

### Modularity

Each agent can be improved independently.

### Extensibility

New agents can easily be added.

Examples:

- competitor price agent
- review sentiment agent
- demand prediction agent

---

# 10. Future Extensions

This architecture allows advanced improvements:

### Self-reflection

Agents critique their own reasoning.

### Multi-step planning

Planner dynamically decides which agents to run.

### Tool usage

Agents call external tools such as:

- web search
- vector databases
- APIs

---

# 11. Summary

The system implements a **collaborative multi-agent architecture** consisting of:

Planner → Workers → Verifier

This pattern improves:

- reliability
- modularity
- scalability

and reflects modern AI agent system design.
