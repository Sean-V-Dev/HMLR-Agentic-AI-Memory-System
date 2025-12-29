---
description: How to set up and develop the LangGraph integration for HMLR
---

# LangGraph Integration Workflow

## Prerequisites
- HMLR core is working (run `python tests/test_phase_11_9_e_7b_vegetarian_conflict.py` to verify)
- OpenAI API key is set in environment

## Initial Setup

// turbo-all

1. Navigate to parent of HMLR:
```powershell
cd c:\Users\seanv
```

2. Create the integration project directory:
```powershell
mkdir langgraph-hmlr
cd langgraph-hmlr
```

3. Create a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

4. Install HMLR as editable dependency:
```powershell
pip install -e ..\HMLR
```

5. Install LangGraph and dependencies:
```powershell
pip install langgraph langchain-core langchain-openai
```

6. Verify installation:
```powershell
python -c "from hmlr import HMLRClient; from langgraph.graph import StateGraph; print('All imports OK')"
```

## Project Structure

Create the following structure:
```
langgraph-hmlr/
├── langgraph_hmlr/
│   ├── __init__.py
│   ├── nodes.py         # LangGraph node implementations
│   ├── state.py         # State schema definitions
│   └── client_pool.py   # HMLR client management
├── tests/
│   ├── __init__.py
│   ├── test_nodes.py
│   └── test_e2e.py
├── examples/
│   └── simple_agent.py
├── requirements.txt
└── setup.py
```

## Implementation Steps

1. Create the package structure:
```powershell
mkdir langgraph_hmlr
mkdir tests
mkdir examples
New-Item langgraph_hmlr\__init__.py -ItemType File
New-Item tests\__init__.py -ItemType File
```

2. Implement the state schema in `langgraph_hmlr/state.py`

3. Implement the HMLR node in `langgraph_hmlr/nodes.py`

4. Write tests in `tests/test_nodes.py`

5. Run tests:
```powershell
pytest tests/ -v
```

## Key Configuration

Set these environment variables for test isolation:
- `COGNITIVE_LATTICE_DB` - Path to HMLR database
- `USER_PROFILE_PATH` - Path to user profile JSON
- `OPENAI_API_KEY` - OpenAI API key

## Testing the Integration

Run the vegetarian constraint test to verify E2E:
```powershell
cd ..\HMLR
python tests\test_phase_11_9_e_7b_vegetarian_conflict.py
```

## Debugging

If HMLR components fail to initialize:
1. Check `ComponentBundle.is_fully_operational()`
2. Check `ComponentBundle.get_degraded_components()`
3. Verify API keys are set
4. Check database path is writable
