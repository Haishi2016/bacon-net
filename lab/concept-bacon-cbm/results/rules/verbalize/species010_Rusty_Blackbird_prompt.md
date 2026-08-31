System
=======
You are a report generator that produces a precise, human-readable, and insightful explanation of a Graded Logic (GL / LSP) aggregation tree. Use your knowledge of graded logic and ornithology to generate a clear, logical report.

Context
=======
This aggregation tree is a fine-grained bird-species classifier for the CUB-200 dataset. It scores how well an image matches the species **Rusty Blackbird**. The leaf "features" are fuzzy attribute-truths in [0, 1] (a value near 1 means the attribute is present); "NOT" denotes a negated attribute. Each internal node is a graded-logic aggregator whose andness (its GCD code) sets how strictly its weighted children must be satisfied. Weights are the normalized relevances used inside the power mean.

Background
==========
GL aggregators are named by their andness per this table:

| Code | Name | Andness a | Verbalization |
|------|------|-----------|---------------|
| CC  | Drastic conjunction     | 2        | "must all be completely satisfied" |
| HHC | High hyper-conjunction  | [5/4, 2) | "below the lowest" |
| CP  | Product t-norm          | 5/4      | "below the lowest" |
| LHC | Low hyper-conjunction   | [1, 5/4) | "below the lowest" |
| C   | Pure conjunction        | 1        | "decided by the lowest" |
| HC+ | High hard conjunction   | 13/14    | "must have all" |
| HC  | Medium hard conjunction | 12/14    | "must have all" |
| HC- | Low hard conjunction    | 11/14    | "must have all" |
| SC+ | High soft conjunction   | 10/14    | "nice to have most" |
| SC  | Medium soft conjunction | 9/14     | "nice to have most" |
| SC- | Low soft conjunction    | 8/14     | "nice to have most" |
| A   | Arithmetic mean         | 7/14     | "nice to have" |
| SD- | Low soft disjunction    | 6/14     | "nice to have some" |
| SD  | Medium soft disjunction | 5/14     | "nice to have some" |
| SD+ | High soft disjunction   | 4/14     | "nice to have some" |
| HD- | Low hard disjunction    | 3/14     | "enough to have any" |
| HD  | Medium hard disjunction | 2/14     | "enough to have any" |
| HD+ | High hard disjunction   | 1/14     | "enough to have any" |
| D   | Pure disjunction        | 0        | "decided by the highest" |
| LHD | Low hyper-disjunction   | (-1/4, 0]| "above the highest" |
| DP  | Product t-conorm        | -1/4     | "above the highest" |
| HHD | High hyper-disjunction  | [-1, -1/4)| "above the highest" |
| DD  | Drastic disjunction     | -1       | "true unless all are zeros" |


Instructions
============
1. Organize the report into:
   - Overview: the overall decision logic in 2-3 sentences.
   - Decision Logic Walkthrough: step by step from the root to the leaves. For each node, name the operator by its code AND verbalization (e.g. "HC+ - high hard conjunction, a strict 'must have all'"), and explain how it combines its children given their weights.
   - Ornithological Interpretation: relate the logic to the real field marks of Rusty Blackbird; note whether the rule matches known diagnostic features, and flag anything surprising, redundant, or spurious.
2. Use plain, human-friendly language.
3. Emphasize the highest-weight features and how they drive the decision.
4. Treat low-weight children as "compensatory / nice-to-have" evidence.

Input
======
The aggregation tree (pruned to its most influential children):
```json
{
  "operator": "CC",
  "name": "drastic conjunction",
  "andness": 1.98,
  "verbalization": "must all be completely satisfied",
  "children": [
    {
      "weight": 0.545,
      "operator": "HD",
      "name": "medium hard disjunction",
      "andness": 0.155,
      "verbalization": "enough to have any",
      "children": [
        {
          "weight": 0.205,
          "operator": "A",
          "name": "arithmetic mean",
          "andness": 0.474,
          "verbalization": "nice to have",
          "children": [
            {
              "weight": 0.839,
              "feature": "nape color: brown"
            },
            {
              "weight": 0.046,
              "feature": "primary color: black"
            },
            {
              "weight": 0.04,
              "feature": "bill shape: all-purpose"
            }
          ]
        },
        {
          "weight": 0.192,
          "operator": "SD-",
          "name": "low soft disjunction",
          "andness": 0.432,
          "verbalization": "nice to have some",
          "children": [
            {
              "weight": 0.712,
              "feature": "nape color: brown"
            },
            {
              "weight": 0.08,
              "feature": "leg color: black"
            },
            {
              "weight": 0.05,
              "feature": "crown color: brown"
            }
          ]
        },
        {
          "weight": 0.136,
          "operator": "HC-",
          "name": "low hard conjunction",
          "andness": 0.75,
          "verbalization": "must have all",
          "children": [
            {
              "weight": 0.208,
              "feature": "NOT breast pattern: solid"
            },
            {
              "weight": 0.182,
              "feature": "belly pattern: solid"
            },
            {
              "weight": 0.115,
              "feature": "shape: perching-like"
            }
          ]
        }
      ]
    },
    {
      "weight": 0.455,
      "operator": "HD",
      "name": "medium hard disjunction",
      "andness": 0.154,
      "verbalization": "enough to have any",
      "children": [
        {
          "weight": 0.205,
          "operator": "A",
          "name": "arithmetic mean",
          "andness": 0.474,
          "verbalization": "nice to have",
          "children": [
            {
              "weight": 0.839,
              "feature": "nape color: brown"
            },
            {
              "weight": 0.046,
              "feature": "primary color: black"
            },
            {
              "weight": 0.04,
              "feature": "bill shape: all-purpose"
            }
          ]
        },
        {
          "weight": 0.192,
          "operator": "SD-",
          "name": "low soft disjunction",
          "andness": 0.432,
          "verbalization": "nice to have some",
          "children": [
            {
              "weight": 0.712,
              "feature": "nape color: brown"
            },
            {
              "weight": 0.08,
              "feature": "leg color: black"
            },
            {
              "weight": 0.05,
              "feature": "crown color: brown"
            }
          ]
        },
        {
          "weight": 0.136,
          "operator": "HC-",
          "name": "low hard conjunction",
          "andness": 0.75,
          "verbalization": "must have all",
          "children": [
            {
              "weight": 0.208,
              "feature": "NOT breast pattern: solid"
            },
            {
              "weight": 0.182,
              "feature": "belly pattern: solid"
            },
            {
              "weight": 0.115,
              "feature": "shape: perching-like"
            }
          ]
        }
      ]
    }
  ]
}
```
