A complex evolutionary search framework designed to solve grid-based transformation tasks (like those from the Abstraction and Reasoning Corpus). Here's a flowchart and process breakdown to help visualize how it works:
  ┌────────────────────┐
 │ Load Task Examples │
 └────────┬───────────┘
          │
          ▼
 ┌──────────────────────────┐
 │ Analyze Patterns (Reasoner) ──────────┐
 └────────┬─────────────────┘           │
          ▼                             ▼
 ┌─────────────────────────────┐   ┌───────────────────────┐
 │ Try Simple Heuristic Program│   │ Suggest Ops from Hints │
 └────────┬────────────┬───────┘   └────────┬────────────────┘
          │            │                    │
     (Success?)     (Fail?)                (Ops)
       │              │                      │
       ▼              ▼                      ▼
   ✅ Return       Initialize             Create Random
   Perfect        Population             Individuals
   Program                                  
       │              │
       ▼              ▼
 ┌───────────────────────────┐
 │ Evolve Population Over Gens│
 └──────────────┬────────────┘
                │
                ▼
         Evaluate Fitness
                │
                ▼
     Select + Crossover + Mutate
                │
                ▼
         (Improved Solution?)
            ┌────┴────┐
            ▼         ▼
        Save Best   Track Stagnation
            │
            ▼
     ┌───────────────┐
     │ Return Best   │
     │ Final Program │
     └───────────────┘
