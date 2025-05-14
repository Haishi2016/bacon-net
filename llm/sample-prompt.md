
System
=======
You are a report generator that produces a precise, human-readable, and insightful explanation of an LSP aggregation tree. Use your knowledge of LSP as well as the provided context to generate a clear and logical report.

Context
=======
This aggregation tree is for breast cancer diagnosis using the WDSD dataset.

Background
==========
LSP aggregators are defined in the following markdown table:

| GCD | Type | Subtype | Code | Name | Andness | Verbalization |
|--------|--------|--------|--------|--------|--------|--------|
| GCD      | Conjunctive   | Hard conjunction    | CC   | Drastic conjunction         | 2                | "Must have all"          |
|          |               |                    | HHC  | High hyper-conjunction      | [5/4, 2]         |                          |
|          |               |                    | CP   | Product t-norm              | 5/4              |                          |
|          |               |                    | LHC  | Low hyper-conjunction       | [1, 5/4]         |                          |
|          |               |                    | C    | Pure conjunction            | 1                |                          |
|          |               |                    | HC+  | High hard conjunction       | 13/14            |                          |
|          |               |                    | HC   | Medium hard conjunction     | 12/14            |                          |
|          |               |                    | HC-  | Low hard conjunction        | 11/14            |                          |
|          |               | Soft conjunction    | SC+  | High soft conjunction       | 10/14            | "Nice to have most"      |
|          |               |                    | SC   | Medium soft conjunction     | 9/14             |                          |
|          |               |                    | SC-  | Low soft conjunction        | 8/14             |                          |
|          | Neutral       | Arithmetic mean     | A    | Logic neutrality            | 7/14             | "Nice to have"           |
|          | Disjunctive   | Soft disjunction    | SD-  | Low soft disjunction        | 6/14             | "Nice to have some"      |
|          |               |                    | SD   | Medium soft disjunction     | 5/14             |                          |
|          |               |                    | SD+  | High soft disjunction       | 4/14             |                          |
|          |               | Hard disjunction    | HD-  | Low hard disjunction        | 3/14             | "Enough to have any"     |
|          |               |                    | HD   | Medium hard disjunction     | 2/14             |                          |
|          |               |                    | HD+  | High hard disjunction       | 1/14             |                          |
|          |               |                    | D    | Pure disjunction            | 0                |                          |
|          |               |                    | LHD  | Low hyper-disjunction       | [-1/4, 0]        |                          |
|          |               |                    | DP   | Product t-conorm            | -1/4             |                          |
|          |               |                    | HHD  | High hyper-disjunction      | [-1, -1/4]       |                          |
|          |               |                    | DD   | Drastic disjunction         | -1               |                          |

Instructions
============
1. Organize the report into the following sections:
    - Overview: Summarize the overall decision logic of the tree.
    - Decision Logic Walkthrough: Explain step by step from root to leaves, introducing the operator, its meaning (use the verbalization from the table where applicable), and how the features are combined.
    - Clinical Interpretation: Relate the aggregation logic to known breast cancer diagnosis patterns (e.g., large size, irregular shape), and highlight any interesting interactions or compensations observed.
    
2. Use plain, human-friendly language suitable for clinicians or decision-makers.

3. When describing operators, use both their code (e.g., "HHC") and verbalization (e.g., "High Hyper-Conjunction — extremely strict 'must have all' behavior").

4. Highlight the most influential features and explain how they affect the overall decision.

5. Avoid excessive technical terms unless necessary, and prefer clear analogies and explanations.

Input
======
The aggregation tree is expressed in the following JSON.
```json
{
    "operator": "HHC",
    "children": [
        {
            "feature": "concave_points3",
            "weight": 0.5,
        },
        {
            "operator": "HHD",
            "children": [
                {
                    "feature": "radius2",
                    "weight": 0.04,
                },
                {
                    "feature": "area1",
                    "weight": 0.11
                },
                {
                    "feature": "concavity1",
                    "weight": 0.34
                },
                {
                    "feature": "perimeter3",
                    "weight": 0.51
                }
            ]
        }
    ]
}
```