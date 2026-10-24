---
name: movement-analysis
description: Interpret raw object tracking data from the real-time camera pipeline.
---

# Movement Analysis Skill

## Purpose

Analyze live tracking data pipeline and return the analysis data from other tools. Be time sensative, only doing tool calls every 5 seconds and providing output data. 

## Workflow

1. every 5 seconds, get data from analyzeMovement2 
2. report the data in a structured format
