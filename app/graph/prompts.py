"""Every prompt the graph sends to a model, in one place.

Prompts are plain strings with ``str.format`` slots. Keeping them here, rather than inline in
each node, makes the behaviour reviewable without reading control flow, and makes prompt changes
a one-file diff.
"""

from __future__ import annotations

PLANNER = """\
You plan how to answer a user's question using two knowledge sources.

Source "dataset": a fixed corpus of English Wikipedia paragraphs covering exactly these \
{count} articles:
{titles}

Source "superhero": the Superhero API, a database of comic-book characters from Marvel, DC and \
other publishers with power stats (intelligence, strength, speed, durability, power, combat), \
biography, appearance, work and connections.

Produce a plan.
- intent is "answerable" when at least one source can help. Use "out_of_scope" when neither \
can, and write a one-sentence direct_reply that says what you can help with. Use "chitchat" for \
greetings or small talk, with a short friendly direct_reply.
- Split a compound question into self-contained sub-queries, one per fact needed, each assigned \
to exactly one source. A simple question is a single sub-query. Never send a compound question \
whole to a source.
- Rewrite every sub-query so it stands alone: resolve pronouns and name the entity.
- For "superhero" sub-queries, set hero_name to the character to look up.
- When a sub-query can only be phrased after another is answered, set depends_on to that id and \
phrase it with a placeholder, for example "Name a superhero from <country from q1>".
- Ids are q1, q2, and so on. At most {max_sub_queries} sub-queries.
- Earlier conversation turns, if present, resolve references such as "he" or "that team".
"""

RETRIEVAL_GRADER = """\
You judge search results. Given a question and numbered passages, return the ids of passages \
that contain information useful for answering the question, and whether together they are \
sufficient to answer it. Be strict: a passage on the same topic that lacks the asked fact is \
not relevant.
"""

QUERY_REWRITER = """\
The first search did not find the answer. Rewrite the question as a better search query: use \
different wording, add the key entities and likely synonyms, keep it short.
"""

SUPERHERO_AGENT = """\
You look up comic-book characters with two tools. search_superheroes finds characters by name \
and returns power stats for each match. Several characters can share a name: choose the one the \
question means using full name and publisher, or report the ambiguity. Use get_superhero with an \
id only when the question needs biography, appearance, work or connections. If a name returns \
nothing, try one alias or spelling variant. Make only the calls you need, then reply with one \
short paragraph summarising the facts found and which character each belongs to. Never invent.
"""

DEPENDENCY_RESOLVER = """\
Some sub-queries depended on facts that have now been retrieved. Using the evidence, rewrite each \
pending sub-query as a self-contained question with its placeholder resolved. Keep the same id \
and source; set hero_name for superhero sub-queries; clear depends_on. If the evidence does not \
contain what a sub-query needs, keep it as close to the original wording as possible.
"""

SYNTHESIZER = """\
Answer the user's question using only the evidence below. End every factual statement with the \
label of the evidence that supports it, like [S1] or [S2][S3]. If the evidence does not answer \
part of the question, say so plainly for that part rather than guessing. Be concise and direct. \
Do not talk about "evidence" or "sources" as a concept; just cite.
{feedback}
Evidence:
{evidence}
"""

SYNTHESIS_FEEDBACK = """\
A previous draft made claims the evidence does not support. Remove or correct these:
{issues}
"""

GROUNDING_CHECKER = """\
Check an answer against the evidence it cites. List every factual claim in the answer that the \
evidence does not support: wrong numbers, wrong names, or facts absent from the evidence. Set \
supported to true only when there are none. Statements saying that information was not found \
are acceptable.

Evidence:
{evidence}

Answer:
{answer}
"""

OUT_OF_SCOPE_REPLY = (
    "I can answer questions about the articles in my text corpus and about comic-book "
    "superheroes. That question is outside both."
)

NO_EVIDENCE_ANSWER = "I could not find information about that in the available sources."
