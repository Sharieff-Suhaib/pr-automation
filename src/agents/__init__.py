"""Agent-SWE multi-agent workflow.

One module per agent in the architecture:

    manager            -> routing / control flow decisions
    repository_agent   -> Repository Recommendation
    similar_bug_agent  -> Similar Bug Recommendation
    strategy_agent     -> Repair Strategy Recommendation
    tool_agent         -> Tool Recommendation
    testcase_agent     -> Test Case Recommendation
    codegen_agent      -> Code Generation
    testing_agent      -> Testing
    reflection_agent   -> Reflection
    feedback_agent     -> Feedback storage

Every agent is `AgentState -> partial AgentState` (see state.py).
"""
