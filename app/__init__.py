"""FastAPI service that answers questions from a text corpus and the Superhero API.

The package layout mirrors the architecture, so the folder names tell the story:

* ``api``            the HTTP surface: routes, request and response schemas, the error envelope,
                     request-id middleware, readiness checks
* ``graph``          the LangGraph agent: query analysis, the retrieval branch, the superhero
                     branch, synthesis and the grounding check
* ``retrieval``      the corpus, its dense vector index, and cross-encoder reranking
* ``tools``          the two knowledge sources exposed as plain callable functions
* ``llm``            model providers, failover between them, embeddings
* ``observability``  structured logging, tracing, metrics

``main.create_app`` assembles these into one application; ``lifespan`` owns the shared resources.
"""

__version__ = "0.1.0"
