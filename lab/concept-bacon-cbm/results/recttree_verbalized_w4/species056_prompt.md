System
=======
You are a report generator that produces a precise, human-readable, and insightful explanation of a Graded Logic (GL / LSP) aggregation tree. Use your knowledge of graded logic and ornithology to generate a clear, logical report.

Context
=======
This aggregation tree is a fine-grained bird-species classifier for the CUB-200 dataset. It scores how well an image matches the species **Rose breasted Grosbeak**. The leaf "features" are fuzzy attribute-truths in [0, 1] (a value near 1 means the attribute is present); "NOT" denotes a negated attribute. Each internal node is a graded-logic aggregator whose andness (its GCD code) sets how strictly its weighted children must be satisfied. Weights are the normalized relevances used inside the power mean.

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
   - Plain-Language Statement: ONE intuitive natural-English sentence that a birder would understand, summarizing what the rule looks for -- weave in your ornithological knowledge (do NOT mention GL/andness/operators here), e.g. "It calls a bird an Rose breasted Grosbeak when it sees a mostly-blue songbird with a dark masked face and a stout bill."
   - Overview: the overall decision logic in 2-3 sentences.
   - Decision Logic Walkthrough: step by step from the root to the leaves. For each node, name the operator by its code AND verbalization (e.g. "HC+ - high hard conjunction, a strict 'must have all'"), and explain how it combines its children given their weights.
   - Ornithological Interpretation: relate the logic to the real field marks of Rose breasted Grosbeak; note whether the rule matches known diagnostic features, and flag anything surprising, redundant, or spurious.
2. Use plain, human-friendly language.
3. Emphasize the highest-weight features and how they drive the decision.
4. Treat low-weight children as "compensatory / nice-to-have" evidence.

Input
======
The aggregation tree (pruned to its most influential children):
```json
{
  "operator": "LHD",
  "name": "low hyper-disjunction",
  "andness": -0.075,
  "verbalization": "above the highest",
  "children": [
    {
      "weight": 0.374,
      "operator": "HHD",
      "name": "high hyper-disjunction",
      "andness": -0.804,
      "verbalization": "above the highest",
      "children": [
        {
          "weight": 0.335,
          "operator": "HHC",
          "name": "high hyper-conjunction",
          "andness": 1.321,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.086,
              "feature": "underparts color: red"
            },
            {
              "weight": 0.085,
              "feature": "leg color: grey"
            },
            {
              "weight": 0.084,
              "feature": "breast color: red"
            }
          ]
        },
        {
          "weight": 0.335,
          "operator": "HHC",
          "name": "high hyper-conjunction",
          "andness": 1.346,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.048,
              "feature": "leg color: grey"
            },
            {
              "weight": 0.048,
              "feature": "breast color: red"
            },
            {
              "weight": 0.048,
              "feature": "head pattern: plain"
            }
          ]
        },
        {
          "weight": 0.33,
          "operator": "HHC",
          "name": "high hyper-conjunction",
          "andness": 1.421,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.23,
              "feature": "head pattern: plain"
            },
            {
              "weight": 0.213,
              "feature": "bill shape: cone"
            },
            {
              "weight": 0.194,
              "feature": "underparts color: white"
            }
          ]
        }
      ]
    },
    {
      "weight": 0.36,
      "operator": "HHD",
      "name": "high hyper-disjunction",
      "andness": -0.8,
      "verbalization": "above the highest",
      "children": [
        {
          "weight": 0.335,
          "operator": "CP",
          "name": "product t-norm",
          "andness": 1.295,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.084,
              "feature": "underparts color: red"
            },
            {
              "weight": 0.081,
              "feature": "tail pattern: multi-colored"
            },
            {
              "weight": 0.077,
              "feature": "primary color: white"
            }
          ]
        },
        {
          "weight": 0.333,
          "operator": "HHC",
          "name": "high hyper-conjunction",
          "andness": 1.321,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.086,
              "feature": "underparts color: red"
            },
            {
              "weight": 0.085,
              "feature": "leg color: grey"
            },
            {
              "weight": 0.084,
              "feature": "breast color: red"
            }
          ]
        },
        {
          "weight": 0.332,
          "operator": "HHC",
          "name": "high hyper-conjunction",
          "andness": 1.346,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.048,
              "feature": "leg color: grey"
            },
            {
              "weight": 0.048,
              "feature": "breast color: red"
            },
            {
              "weight": 0.048,
              "feature": "head pattern: plain"
            }
          ]
        }
      ]
    },
    {
      "weight": 0.267,
      "operator": "HHD",
      "name": "high hyper-disjunction",
      "andness": -0.77,
      "verbalization": "above the highest",
      "children": [
        {
          "weight": 0.505,
          "operator": "CP",
          "name": "product t-norm",
          "andness": 1.295,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.084,
              "feature": "underparts color: red"
            },
            {
              "weight": 0.081,
              "feature": "tail pattern: multi-colored"
            },
            {
              "weight": 0.077,
              "feature": "primary color: white"
            }
          ]
        },
        {
          "weight": 0.495,
          "operator": "HHC",
          "name": "high hyper-conjunction",
          "andness": 1.421,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.23,
              "feature": "head pattern: plain"
            },
            {
              "weight": 0.213,
              "feature": "bill shape: cone"
            },
            {
              "weight": 0.194,
              "feature": "underparts color: white"
            }
          ]
        }
      ]
    }
  ]
}
```
