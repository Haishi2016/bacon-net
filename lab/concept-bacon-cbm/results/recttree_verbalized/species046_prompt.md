System
=======
You are a report generator that produces a precise, human-readable, and insightful explanation of a Graded Logic (GL / LSP) aggregation tree. Use your knowledge of graded logic and ornithology to generate a clear, logical report.

Context
=======
This aggregation tree is a fine-grained bird-species classifier for the CUB-200 dataset. It scores how well an image matches the species **American Goldfinch**. The leaf "features" are fuzzy attribute-truths in [0, 1] (a value near 1 means the attribute is present); "NOT" denotes a negated attribute. Each internal node is a graded-logic aggregator whose andness (its GCD code) sets how strictly its weighted children must be satisfied. Weights are the normalized relevances used inside the power mean.

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
   - Plain-Language Statement: ONE intuitive natural-English sentence that a birder would understand, summarizing what the rule looks for -- weave in your ornithological knowledge (do NOT mention GL/andness/operators here), e.g. "It calls a bird an American Goldfinch when it sees a mostly-blue songbird with a dark masked face and a stout bill."
   - Overview: the overall decision logic in 2-3 sentences.
   - Decision Logic Walkthrough: step by step from the root to the leaves. For each node, name the operator by its code AND verbalization (e.g. "HC+ - high hard conjunction, a strict 'must have all'"), and explain how it combines its children given their weights.
   - Ornithological Interpretation: relate the logic to the real field marks of American Goldfinch; note whether the rule matches known diagnostic features, and flag anything surprising, redundant, or spurious.
2. Use plain, human-friendly language.
3. Emphasize the highest-weight features and how they drive the decision.
4. Treat low-weight children as "compensatory / nice-to-have" evidence.

Input
======
The aggregation tree (pruned to its most influential children):
```json
{
  "operator": "HC-",
  "name": "low hard conjunction",
  "andness": 0.75,
  "verbalization": "must have all",
  "children": [
    {
      "weight": 0.163,
      "operator": "HC-",
      "name": "low hard conjunction",
      "andness": 0.75,
      "verbalization": "must have all",
      "children": [
        {
          "weight": 0.201,
          "operator": "CP",
          "name": "product t-norm",
          "andness": 1.225,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.689,
              "feature": "bill color: orange"
            },
            {
              "weight": 0.311,
              "feature": "forehead color: black"
            }
          ]
        },
        {
          "weight": 0.199,
          "operator": "CP",
          "name": "product t-norm",
          "andness": 1.21,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.413,
              "feature": "bill color: orange"
            },
            {
              "weight": 0.21,
              "feature": "nape color: yellow"
            },
            {
              "weight": 0.191,
              "feature": "underparts color: yellow"
            }
          ]
        },
        {
          "weight": 0.195,
          "operator": "CP",
          "name": "product t-norm",
          "andness": 1.202,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.383,
              "feature": "wing color: white"
            },
            {
              "weight": 0.338,
              "feature": "under tail color: white"
            },
            {
              "weight": 0.279,
              "feature": "wing pattern: multi-colored"
            }
          ]
        }
      ]
    },
    {
      "weight": 0.162,
      "operator": "HC-",
      "name": "low hard conjunction",
      "andness": 0.794,
      "verbalization": "must have all",
      "children": [
        {
          "weight": 0.209,
          "operator": "CP",
          "name": "product t-norm",
          "andness": 1.188,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.36,
              "feature": "underparts color: yellow"
            },
            {
              "weight": 0.357,
              "feature": "primary color: yellow"
            },
            {
              "weight": 0.284,
              "feature": "shape: perching-like"
            }
          ]
        },
        {
          "weight": 0.156,
          "operator": "CP",
          "name": "product t-norm",
          "andness": 1.197,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.402,
              "feature": "wing color: black"
            },
            {
              "weight": 0.316,
              "feature": "belly color: yellow"
            },
            {
              "weight": 0.282,
              "feature": "crown color: black"
            }
          ]
        },
        {
          "weight": 0.155,
          "operator": "HC+",
          "name": "high hard conjunction",
          "andness": 0.923,
          "verbalization": "must have all",
          "children": [
            {
              "weight": 0.401,
              "feature": "breast color: yellow"
            },
            {
              "weight": 0.324,
              "feature": "crown color: iridescent"
            },
            {
              "weight": 0.275,
              "feature": "wing color: orange"
            }
          ]
        }
      ]
    },
    {
      "weight": 0.162,
      "operator": "HC-",
      "name": "low hard conjunction",
      "andness": 0.791,
      "verbalization": "must have all",
      "children": [
        {
          "weight": 0.28,
          "operator": "LHC",
          "name": "low hyper-conjunction",
          "andness": 1.158,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 0.328,
              "feature": "breast color: yellow"
            },
            {
              "weight": 0.267,
              "feature": "wing color: black"
            },
            {
              "weight": 0.225,
              "feature": "wing color: orange"
            }
          ]
        },
        {
          "weight": 0.199,
          "operator": "CP",
          "name": "product t-norm",
          "andness": 1.201,
          "verbalization": "below the lowest",
          "children": [
            {
              "weight": 1.0,
              "feature": "crown color: brown"
            }
          ]
        },
        {
          "weight": 0.184,
          "operator": "HC+",
          "name": "high hard conjunction",
          "andness": 0.925,
          "verbalization": "must have all",
          "children": [
            {
              "weight": 1.0,
              "feature": "head pattern: masked"
            }
          ]
        }
      ]
    }
  ]
}
```
