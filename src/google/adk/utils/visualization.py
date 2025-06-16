"""
Utilities for visualizing google-adk agents.
"""

from __future__ import annotations

import itertools
from typing import Any
from typing import Tuple

from google.adk.agents import LoopAgent
from google.adk.agents import ParallelAgent
from google.adk.agents import SequentialAgent
import requests


def build_mermaid(root_agent: Any) -> Tuple[str, bytes]:
  """
  Generates a Mermaid 'flowchart LR' diagram for a google-adk
  agent tree and returns both the Mermaid source and a PNG
  image rendered via the Kroki API.

  Args:
      root_agent (Any):
          The root agent node of the google-adk agent tree.
          This should be an instance
          of SequentialAgent, LoopAgent, ParallelAgent,
          or a compatible agent class with a
          `name` attribute and an optional `sub_agents`
          attribute.

  Returns:
      Tuple[str, bytes]:
          A tuple containing:
          - The Mermaid source code as a string.
          - The PNG image bytes rendered from the Mermaid diagram.

  Raises:
      requests.RequestException: If the request to the Kroki API fails.

  Example:
      >>> mermaid_src, png_bytes = build_mermaid(my_agent_tree)
      >>> print(mermaid_src)
      >>> with open("diagram.png", "wb") as f:
      ...     f.write(png_bytes)
  """
  clusters, edges = [], []
  first_of, last_of, nodes = {}, {}, {}

  # Walk the agent tree
  def walk(node):
    nid = id(node)
    nodes[nid] = node
    name = node.name
    subs = getattr(node, "sub_agents", []) or []
    if subs:
      first_of[nid], last_of[nid] = subs[0].name, subs[-1].name
    # Create subgraph for non-root composite nodes
    if node is not root_agent and isinstance(
        node, (SequentialAgent, LoopAgent, ParallelAgent)
    ):
      block = [f'subgraph {name}["{name}"]']
      if isinstance(node, (SequentialAgent, LoopAgent)):
        for a, b in itertools.pairwise(subs):
          block.append(f"  {a.name} --> {b.name}")
        # loop-back even for single-child loops
        if isinstance(node, LoopAgent):
          if len(subs) == 1:
            block.append(f"  {subs[0].name} -.->|repeat| {subs[0].name}")
          elif len(subs) > 1:
            block.append(f"  {subs[-1].name} -.->|repeat| {subs[0].name}")
      elif isinstance(node, ParallelAgent):
        for child in subs:
          block.append(f'  {child.name}["{child.name}"]')
      block.append("end")
      clusters.append("\n".join(block))
    # Recurse
    for child in subs:
      walk(child)

  walk(root_agent)

  # Link root children
  if isinstance(root_agent, SequentialAgent):
    children = root_agent.sub_agents or []
    # Kick-off
    if children:
      first = children[0]
      if isinstance(first, ParallelAgent):
        for c in first.sub_agents:
          edges.append(f"{root_agent.name} -.-> {c.name}")
      else:
        edges.append(
            f"{root_agent.name} -.-> {first_of.get(id(first), first.name)}"
        )
    # Chain
    for prev, nxt in itertools.pairwise(children):
      prev_exits = (
          [c.name for c in prev.sub_agents]
          if isinstance(prev, ParallelAgent)
          else [last_of.get(id(prev), prev.name)]
      )
      nxt_entries = (
          [c.name for c in nxt.sub_agents]
          if isinstance(nxt, ParallelAgent)
          else [first_of.get(id(nxt), nxt.name)]
      )
      arrow = "-.->" if isinstance(nxt, ParallelAgent) else "-->"
      for src in prev_exits:
        for dst in nxt_entries:
          edges.append(f"{src} {arrow} {dst}")
  else:
    for c in getattr(root_agent, "sub_agents", []) or []:
      edges.append(f"{root_agent.name} --> {c.name}")

  # Assemble graph as mermaid code
  mermaid_src = "\n".join(
      ["flowchart LR", f'{root_agent.name}["{root_agent.name}"]']
      + clusters
      + edges
  )

  # Render via Kroki
  # note: kroki is a third party service which enables the rendering
  # of mermaid diagrams without local npm installation of mermaid.
  png = requests.post(
      "https://kroki.io/mermaid/png",
      data=mermaid_src.encode("utf-8"),
      headers={"Content-Type": "text/plain"},
  ).content

  return mermaid_src, png
