"""The LangGraph agent that answers a question.

``build.build_graph`` wires the nodes in ``nodes/`` into this flow::

    analyze_query -> [Send per sub-query] -> retrieve_documents | superhero_agent
                  -> collect -> (resolve_dependencies -> second wave) -> synthesize
                  -> check_grounding -> (regenerate once) -> END

``analyze_query`` alone decides routing. It classifies the question, splits compound questions
into self-contained sub-queries and assigns each to a source. Every sub-query runs as its own
parallel branch. Evidence collected by the branches is the only material the synthesizer may
cite, and the grounding check verifies the answer against it.
"""
