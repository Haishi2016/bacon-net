You are reasoning about a Graded Logic (GL / LSP) decision rule. A rule is a tree
of aggregators over fuzzy attribute-truths in [0,1] (1 = attribute present; "NOT"
= the attribute is absent). Each internal node combines its weighted children
with an *andness* (a GCD code) that sets how strictly the children must hold:
high-andness nodes are conjunctive ("must have all" / decided by the LOWEST
child), low/negative-andness nodes are disjunctive ("enough to have any" / decided
by the HIGHEST child), and mid values are soft averages. Weights are relative
relevances inside the weighted power mean. A node may carry an "id"; a child of
the form {"weight": w, "shared_ref": <id>} means the SAME sub-rule (already
given above under that id) is reused as another input -- evaluate/describe it
ONCE, do not treat it as a separate duplicate profile. GCD codes:

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

IMPORTANT: Reason ONLY from the information in this prompt. Do NOT use outside
knowledge, memorized facts, or long-term memory beyond what is stated here.

TASK
====
Below are 200 graded-logic decision rules, each identified only by an
opaque id (the class it detects is hidden). For EACH rule, write a concise
natural-language decision rule (1-3 sentences) that states what an input must
look like to satisfy it -- naming the key attributes and how they are logically
combined (which are required/conjunctive, which are alternatives/disjunctive,
which must be ABSENT). Describe ONLY the logic; do not name or guess any class.

OUTPUT
======
Return a CSV with a header row and one row per rule, EXACTLY:
rule_id,verbalization
Quote the verbalization field (wrap in double quotes; escape internal quotes).
Output only the CSV, nothing else.

RULES
=====
### R154
```json
{
 "id": "N1",
 "operator": "HC-",
 "name": "low hard conjunction",
 "andness": 0.752,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.956,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.881,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.188,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.284,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.489,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.429,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.037,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.023,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.01,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.007,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.167,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.077,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.282,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.251,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.178,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.165,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.125,
       "feature": "breast color: brown"
      }
     ]
    },
    {
     "weight": 0.155,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.138,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.402,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.271,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.177,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.149,
       "feature": "wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.138,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.152,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.508,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.446,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.038,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.007,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.124,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.523,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.324,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.203,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.051,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.047,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.047,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "back color: yellow"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.437,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.057,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.056,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.052,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.052,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.052,
       "feature": "NOT bill color: grey"
      },
      {
       "weight": 0.049,
       "feature": "NOT back pattern: striped"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.044,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.751,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.188,
     "shared_ref": "N3"
    },
    {
     "weight": 0.167,
     "shared_ref": "N4"
    },
    {
     "weight": 0.155,
     "shared_ref": "N5"
    },
    {
     "weight": 0.138,
     "shared_ref": "N6"
    },
    {
     "weight": 0.124,
     "shared_ref": "N7"
    },
    {
     "weight": 0.108,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R052
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.042,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.592,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.226,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.196,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.324,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.696,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.068,
       "feature": "crown color: white"
      },
      {
       "weight": 0.043,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.043,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.035,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.026,
       "feature": "wing pattern: solid"
      }
     ]
    },
    {
     "weight": 0.186,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.353,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.705,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.069,
       "feature": "crown color: white"
      },
      {
       "weight": 0.043,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.043,
       "feature": "primary color: black"
      },
      {
       "weight": 0.035,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.027,
       "feature": "wing pattern: solid"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.05,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.655,
       "feature": "primary color: black"
      },
      {
       "weight": 0.1,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.098,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.091,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.057,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.313,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.778,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.169,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.027,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.01,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.008,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.008,
       "feature": "NOT belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.149,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.678,
       "feature": "nape color: white"
      },
      {
       "weight": 0.322,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.077,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.29,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.191,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.179,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.174,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.156,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.154,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.147,
       "feature": "NOT upperparts color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.408,
   "id": "N9",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.034,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.196,
     "shared_ref": "N3"
    },
    {
     "weight": 0.186,
     "shared_ref": "N4"
    },
    {
     "weight": 0.115,
     "shared_ref": "N5"
    },
    {
     "weight": 0.103,
     "shared_ref": "N6"
    },
    {
     "weight": 0.092,
     "shared_ref": "N7"
    },
    {
     "weight": 0.077,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R074
```json
{
 "id": "N1",
 "operator": "HD-",
 "name": "low hard disjunction",
 "andness": 0.245,
 "verbalization": "enough to have any",
 "children": [
  {
   "weight": 0.576,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.169,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.515,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.052,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.142,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.131,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.092,
       "feature": "throat color: black"
      },
      {
       "weight": 0.084,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.073,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.071,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.432,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.055,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.145,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.135,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.094,
       "feature": "throat color: black"
      },
      {
       "weight": 0.086,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.075,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.073,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.008,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.126,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.38,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.368,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.252,
       "feature": "belly color: black"
      }
     ]
    },
    {
     "weight": 0.006,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.137,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.488,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.17,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.13,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.104,
       "feature": "back color: grey"
      },
      {
       "weight": 0.08,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.029,
       "feature": "forehead color: white"
      }
     ]
    },
    {
     "weight": 0.006,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.74,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.214,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.183,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.129,
       "feature": "primary color: black"
      },
      {
       "weight": 0.086,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.085,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.082,
       "feature": "nape color: black"
      }
     ]
    },
    {
     "weight": 0.006,
     "id": "N8",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.106,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.367,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.23,
       "feature": "breast color: black"
      },
      {
       "weight": 0.192,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.157,
       "feature": "back color: grey"
      },
      {
       "weight": 0.054,
       "feature": "belly color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.424,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.889,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.515,
     "shared_ref": "N3"
    },
    {
     "weight": 0.432,
     "shared_ref": "N4"
    },
    {
     "weight": 0.008,
     "shared_ref": "N5"
    },
    {
     "weight": 0.006,
     "shared_ref": "N6"
    },
    {
     "weight": 0.006,
     "shared_ref": "N7"
    },
    {
     "weight": 0.006,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R157
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.888,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.611,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.096,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.201,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.5,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.266,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.241,
       "feature": "belly color: black"
      },
      {
       "weight": 0.236,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.122,
       "feature": "breast color: black"
      },
      {
       "weight": 0.068,
       "feature": "back color: black"
      },
      {
       "weight": 0.034,
       "feature": "upperparts color: black"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.064,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.594,
       "feature": "nape color: black"
      },
      {
       "weight": 0.231,
       "feature": "crown color: black"
      },
      {
       "weight": 0.175,
       "feature": "wing color: black"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N5",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.077,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.317,
       "feature": "primary color: black"
      },
      {
       "weight": 0.243,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.225,
       "feature": "crown color: black"
      },
      {
       "weight": 0.216,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.339,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.346,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.282,
       "feature": "leg color: black"
      },
      {
       "weight": 0.109,
       "feature": "bill color: black"
      },
      {
       "weight": 0.096,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.093,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.073,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.137,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.368,
       "feature": "nape color: black"
      },
      {
       "weight": 0.362,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.27,
       "feature": "throat color: black"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.064,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.322,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.24,
       "feature": "throat color: black"
      },
      {
       "weight": 0.179,
       "feature": "primary color: black"
      },
      {
       "weight": 0.137,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.122,
       "feature": "forehead color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.389,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.367,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.201,
     "shared_ref": "N3"
    },
    {
     "weight": 0.111,
     "shared_ref": "N4"
    },
    {
     "weight": 0.111,
     "shared_ref": "N5"
    },
    {
     "weight": 0.109,
     "shared_ref": "N6"
    },
    {
     "weight": 0.108,
     "shared_ref": "N7"
    },
    {
     "weight": 0.106,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R116
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.3,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.711,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.133,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.236,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.511,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.215,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.155,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.117,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.075,
       "feature": "belly color: black"
      },
      {
       "weight": 0.071,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.047,
       "feature": "throat color: black"
      }
     ]
    },
    {
     "weight": 0.216,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.53,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.203,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.146,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.11,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.071,
       "feature": "belly color: black"
      },
      {
       "weight": 0.067,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.061,
       "feature": "upper tail color: black"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N5",
     "operator": "A",
     "name": "arithmetic mean",
     "andness": 0.5,
     "verbalization": "nice to have",
     "children": [
      {
       "weight": 0.999,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.001,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.0,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.0,
       "feature": "NOT nape color: yellow"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.092,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.709,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.288,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.002,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.0,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.0,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N7",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.091,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.461,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.381,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.155,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.001,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.0,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.0,
       "feature": "NOT underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.068,
     "id": "N8",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.009,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.528,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.258,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.213,
       "feature": "NOT back color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.289,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.021,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.236,
     "shared_ref": "N3"
    },
    {
     "weight": 0.216,
     "shared_ref": "N4"
    },
    {
     "weight": 0.096,
     "shared_ref": "N5"
    },
    {
     "weight": 0.093,
     "shared_ref": "N6"
    },
    {
     "weight": 0.085,
     "shared_ref": "N7"
    },
    {
     "weight": 0.068,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R044
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.72,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.94,
   "id": "N2",
   "operator": "HHD",
   "name": "high hyper-disjunction",
   "andness": -0.381,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.457,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.776,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.181,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.119,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.114,
       "feature": "wing color: white"
      },
      {
       "weight": 0.102,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.081,
       "feature": "breast color: black"
      },
      {
       "weight": 0.047,
       "feature": "nape color: black"
      }
     ]
    },
    {
     "weight": 0.237,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.812,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.223,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.126,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.092,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.07,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.058,
       "feature": "nape color: black"
      },
      {
       "weight": 0.057,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.124,
     "id": "N5",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.857,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.235,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.178,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.142,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.136,
       "feature": "throat color: white"
      },
      {
       "weight": 0.097,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.079,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.028,
     "id": "N6",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.742,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.611,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.389,
       "feature": "underparts color: grey"
      }
     ]
    },
    {
     "weight": 0.021,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.384,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.095,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.069,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.063,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.053,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT bill length: about the same as head"
      },
      {
       "weight": 0.049,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.018,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.467,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.08,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.058,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.058,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.053,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.049,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.044,
       "feature": "NOT back pattern: solid"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.06,
   "id": "N9",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.239,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.457,
     "shared_ref": "N3"
    },
    {
     "weight": 0.237,
     "shared_ref": "N4"
    },
    {
     "weight": 0.124,
     "shared_ref": "N5"
    },
    {
     "weight": 0.028,
     "shared_ref": "N6"
    },
    {
     "weight": 0.021,
     "shared_ref": "N7"
    },
    {
     "weight": 0.018,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R011
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.678,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.617,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.056,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.144,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.551,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.192,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.11,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.096,
       "feature": "primary color: white"
      },
      {
       "weight": 0.078,
       "feature": "throat color: black"
      },
      {
       "weight": 0.061,
       "feature": "throat color: white"
      },
      {
       "weight": 0.056,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.095,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.946,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.013,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.009,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.008,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.007,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.007,
       "feature": "NOT head pattern: plain"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N5",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.066,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.458,
       "feature": "primary color: white"
      },
      {
       "weight": 0.172,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.14,
       "feature": "nape color: black"
      },
      {
       "weight": 0.088,
       "feature": "wing color: black"
      },
      {
       "weight": 0.015,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.014,
       "feature": "NOT bill shape: cone"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.056,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.287,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.226,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.198,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.171,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.118,
       "feature": "NOT back color: yellow"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.389,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.227,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.204,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.203,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.19,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.177,
       "feature": "NOT tail pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.077,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.58,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.049,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.048,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.048,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.048,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.046,
       "feature": "NOT wing pattern: spotted"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.383,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.283,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.144,
     "shared_ref": "N3"
    },
    {
     "weight": 0.108,
     "shared_ref": "N4"
    },
    {
     "weight": 0.102,
     "shared_ref": "N5"
    },
    {
     "weight": 0.1,
     "shared_ref": "N6"
    },
    {
     "weight": 0.087,
     "shared_ref": "N7"
    },
    {
     "weight": 0.077,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R066
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.718,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.58,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.096,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.186,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.502,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.404,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.176,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.074,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.055,
       "feature": "nape color: black"
      },
      {
       "weight": 0.055,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.051,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.142,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.547,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.391,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.181,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.17,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.07,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.053,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.037,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.135,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.556,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.207,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.181,
       "feature": "breast color: black"
      },
      {
       "weight": 0.118,
       "feature": "throat color: black"
      },
      {
       "weight": 0.101,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.075,
       "feature": "nape color: black"
      },
      {
       "weight": 0.064,
       "feature": "upper tail color: black"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.256,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.331,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.267,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.234,
       "feature": "breast color: black"
      },
      {
       "weight": 0.127,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.031,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.01,
       "feature": "tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.061,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.41,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.088,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.076,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.069,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.059,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.051,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.048,
       "feature": "NOT forehead color: yellow"
      }
     ]
    },
    {
     "weight": 0.061,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.344,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.214,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.071,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.07,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.064,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.06,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.059,
       "feature": "NOT size: small (5 - 9 in)"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.42,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.309,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.186,
     "shared_ref": "N3"
    },
    {
     "weight": 0.142,
     "shared_ref": "N4"
    },
    {
     "weight": 0.135,
     "shared_ref": "N5"
    },
    {
     "weight": 0.1,
     "shared_ref": "N6"
    },
    {
     "weight": 0.061,
     "shared_ref": "N7"
    },
    {
     "weight": 0.061,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R111
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.71,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.59,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.125,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.165,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.441,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.195,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.193,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.145,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.117,
       "feature": "leg color: black"
      },
      {
       "weight": 0.103,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.096,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N4",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.128,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.397,
       "feature": "belly color: black"
      },
      {
       "weight": 0.26,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.19,
       "feature": "throat color: black"
      },
      {
       "weight": 0.146,
       "feature": "nape color: black"
      },
      {
       "weight": 0.007,
       "feature": "size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N5",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.187,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.32,
       "feature": "belly color: black"
      },
      {
       "weight": 0.299,
       "feature": "breast color: black"
      },
      {
       "weight": 0.21,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.163,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.009,
       "feature": "NOT breast color: grey"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N6",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.039,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.603,
       "feature": "breast color: black"
      },
      {
       "weight": 0.308,
       "feature": "throat color: black"
      },
      {
       "weight": 0.013,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.013,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.012,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.011,
       "feature": "NOT back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.083,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.116,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.669,
       "feature": "nape color: black"
      },
      {
       "weight": 0.041,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.039,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.038,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.036,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.036,
       "feature": "NOT throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.071,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.369,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.044,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.036,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.034,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.033,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.032,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.032,
       "feature": "NOT throat color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.41,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.976,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.165,
     "shared_ref": "N3"
    },
    {
     "weight": 0.116,
     "shared_ref": "N4"
    },
    {
     "weight": 0.11,
     "shared_ref": "N5"
    },
    {
     "weight": 0.1,
     "shared_ref": "N6"
    },
    {
     "weight": 0.083,
     "shared_ref": "N7"
    },
    {
     "weight": 0.071,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R021
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.596,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.822,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.804,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.746,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.053,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.148,
       "feature": "belly color: black"
      },
      {
       "weight": 0.106,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.1,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.096,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.065,
       "feature": "nape color: black"
      },
      {
       "weight": 0.064,
       "feature": "breast color: black"
      }
     ]
    },
    {
     "weight": 0.143,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.151,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.152,
       "feature": "belly color: black"
      },
      {
       "weight": 0.109,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.103,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.099,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.066,
       "feature": "breast color: black"
      },
      {
       "weight": 0.057,
       "feature": "wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.024,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.108,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.226,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.2,
       "feature": "primary color: black"
      },
      {
       "weight": 0.194,
       "feature": "leg color: black"
      },
      {
       "weight": 0.172,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.124,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.085,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.022,
     "id": "N6",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.036,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.273,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.266,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.241,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.138,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.081,
       "feature": "underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.02,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.057,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.702,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.174,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.124,
       "feature": "throat color: buff"
      }
     ]
    },
    {
     "weight": 0.012,
     "id": "N8",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.751,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.276,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.215,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.21,
       "feature": "bill color: black"
      },
      {
       "weight": 0.183,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.048,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.036,
       "feature": "back pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.178,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.774,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.746,
     "shared_ref": "N3"
    },
    {
     "weight": 0.143,
     "shared_ref": "N4"
    },
    {
     "weight": 0.024,
     "shared_ref": "N5"
    },
    {
     "weight": 0.022,
     "shared_ref": "N6"
    },
    {
     "weight": 0.02,
     "shared_ref": "N7"
    },
    {
     "weight": 0.012,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R197
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.98,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.545,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.155,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.205,
     "id": "N3",
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
      },
      {
       "weight": 0.034,
       "feature": "wing color: black"
      },
      {
       "weight": 0.015,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.014,
       "feature": "NOT breast color: grey"
      }
     ]
    },
    {
     "weight": 0.192,
     "id": "N4",
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
      },
      {
       "weight": 0.05,
       "feature": "crown color: black"
      },
      {
       "weight": 0.043,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.039,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.136,
     "id": "N5",
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
      },
      {
       "weight": 0.097,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.08,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.072,
       "feature": "NOT wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N6",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.13,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.11,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.089,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.082,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.069,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.063,
       "feature": "primary color: black"
      }
     ]
    },
    {
     "weight": 0.037,
     "id": "N7",
     "operator": "SD-",
     "name": "low soft disjunction",
     "andness": 0.4,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.283,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.271,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.25,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.196,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.037,
     "id": "N8",
     "operator": "SD",
     "name": "medium soft disjunction",
     "andness": 0.366,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.34,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.307,
       "feature": "crown color: black"
      },
      {
       "weight": 0.234,
       "feature": "bill color: black"
      },
      {
       "weight": 0.119,
       "feature": "upperparts color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.455,
   "id": "N9",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.154,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.205,
     "shared_ref": "N3"
    },
    {
     "weight": 0.192,
     "shared_ref": "N4"
    },
    {
     "weight": 0.136,
     "shared_ref": "N5"
    },
    {
     "weight": 0.108,
     "shared_ref": "N6"
    },
    {
     "weight": 0.037,
     "shared_ref": "N7"
    },
    {
     "weight": 0.037,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R110
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.108,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.656,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.259,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.132,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.469,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.114,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.09,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.084,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.079,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.078,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.078,
       "feature": "NOT shape: duck-like"
      }
     ]
    },
    {
     "weight": 0.121,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.736,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.102,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.058,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.056,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.051,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.048,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.046,
       "feature": "NOT head pattern: plain"
      }
     ]
    },
    {
     "weight": 0.12,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.694,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.079,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.066,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.06,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.058,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.057,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.052,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.532,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.08,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.068,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.053,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.052,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.051,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.05,
       "feature": "NOT belly color: brown"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.446,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.124,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.092,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.089,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.085,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.084,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.082,
       "feature": "NOT bill color: grey"
      }
     ]
    },
    {
     "weight": 0.099,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.181,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.278,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.264,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.249,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.208,
       "feature": "NOT back color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.344,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.347,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.132,
     "shared_ref": "N3"
    },
    {
     "weight": 0.121,
     "shared_ref": "N4"
    },
    {
     "weight": 0.12,
     "shared_ref": "N5"
    },
    {
     "weight": 0.115,
     "shared_ref": "N6"
    },
    {
     "weight": 0.102,
     "shared_ref": "N7"
    },
    {
     "weight": 0.099,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R065
```json
{
 "id": "N1",
 "operator": "HC-",
 "name": "low hard conjunction",
 "andness": 0.813,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.681,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.852,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.246,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.457,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.359,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.124,
       "feature": "belly color: black"
      },
      {
       "weight": 0.078,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.075,
       "feature": "back color: black"
      },
      {
       "weight": 0.06,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.048,
       "feature": "wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.12,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.362,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.212,
       "feature": "back color: white"
      },
      {
       "weight": 0.152,
       "feature": "throat color: black"
      },
      {
       "weight": 0.147,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.07,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.057,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.245,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.283,
       "feature": "belly color: black"
      },
      {
       "weight": 0.18,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.138,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.104,
       "feature": "wing color: white"
      },
      {
       "weight": 0.102,
       "feature": "nape color: white"
      },
      {
       "weight": 0.077,
       "feature": "primary color: black"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.507,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.117,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.098,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.097,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.085,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.084,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.066,
       "feature": "NOT wing color: grey"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.495,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.137,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.137,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.132,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.122,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.075,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.064,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.533,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.179,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.107,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.104,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.101,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.085,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.085,
       "feature": "NOT wing shape: pointed-wings"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.319,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.814,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.246,
     "shared_ref": "N3"
    },
    {
     "weight": 0.11,
     "shared_ref": "N4"
    },
    {
     "weight": 0.109,
     "shared_ref": "N5"
    },
    {
     "weight": 0.084,
     "shared_ref": "N6"
    },
    {
     "weight": 0.084,
     "shared_ref": "N7"
    },
    {
     "weight": 0.084,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R034
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.896,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.944,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.982,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.157,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.194,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.252,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.212,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.188,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.182,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.166,
       "feature": "bill color: grey"
      }
     ]
    },
    {
     "weight": 0.138,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.438,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.253,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.189,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.182,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.167,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.069,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.034,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N5",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.03,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.506,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.266,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.143,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.055,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.03,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.563,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.098,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.078,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.054,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.048,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.04,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.037,
       "feature": "NOT primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.596,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.21,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.05,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.041,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.04,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.036,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.036,
       "feature": "NOT crown color: black"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.464,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.137,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.127,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.095,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.061,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.059,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.056,
       "feature": "NOT wing color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.056,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.831,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.157,
     "shared_ref": "N3"
    },
    {
     "weight": 0.138,
     "shared_ref": "N4"
    },
    {
     "weight": 0.093,
     "shared_ref": "N5"
    },
    {
     "weight": 0.084,
     "shared_ref": "N6"
    },
    {
     "weight": 0.081,
     "shared_ref": "N7"
    },
    {
     "weight": 0.081,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R080
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.553,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.562,
   "id": "N2",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.383,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.092,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.508,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.143,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.134,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.102,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.099,
       "feature": "belly color: white"
      },
      {
       "weight": 0.097,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.079,
       "feature": "bill shape: cone"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.228,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.212,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.198,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.15,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.147,
       "feature": "belly color: white"
      },
      {
       "weight": 0.144,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.112,
       "feature": "wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.132,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.497,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.321,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.11,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.072,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.492,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.106,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.092,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.053,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.045,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.044,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.043,
       "feature": "nape color: grey"
      }
     ]
    },
    {
     "weight": 0.079,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.362,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.078,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.062,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.057,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.056,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.055,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.053,
       "feature": "NOT upperparts color: white"
      }
     ]
    },
    {
     "weight": 0.079,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.472,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.087,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.058,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.045,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.044,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.044,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.039,
       "feature": "NOT forehead color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.438,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.476,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.092,
     "shared_ref": "N3"
    },
    {
     "weight": 0.085,
     "shared_ref": "N4"
    },
    {
     "weight": 0.082,
     "shared_ref": "N5"
    },
    {
     "weight": 0.08,
     "shared_ref": "N6"
    },
    {
     "weight": 0.079,
     "shared_ref": "N7"
    },
    {
     "weight": 0.079,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R159
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.87,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.612,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.874,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.114,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.354,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.231,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.171,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.161,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.156,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.144,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.138,
       "feature": "NOT nape color: white"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.439,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.083,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.078,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.073,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.072,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.07,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.065,
       "feature": "NOT bill shape: dagger"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.645,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.036,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.035,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.032,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.027,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.026,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.026,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.513,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.06,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.058,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.049,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.048,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.046,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.045,
       "feature": "NOT belly color: brown"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.354,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.398,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.349,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.253,
       "feature": "NOT wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.621,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.051,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.046,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.035,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.032,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.03,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.026,
       "feature": "NOT forehead color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.388,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.482,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.114,
     "shared_ref": "N3"
    },
    {
     "weight": 0.106,
     "shared_ref": "N4"
    },
    {
     "weight": 0.106,
     "shared_ref": "N5"
    },
    {
     "weight": 0.105,
     "shared_ref": "N6"
    },
    {
     "weight": 0.103,
     "shared_ref": "N7"
    },
    {
     "weight": 0.102,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R149
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.151,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.628,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.915,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.162,
     "id": "N3",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.249,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 1.0,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.0,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.0,
       "feature": "under tail color: buff"
      }
     ]
    },
    {
     "weight": 0.157,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.791,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.228,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.163,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.154,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.136,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.09,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.079,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.123,
     "id": "N5",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.25,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.998,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.001,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.0,
       "feature": "throat color: black"
      },
      {
       "weight": 0.0,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.0,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.0,
       "feature": "belly color: grey"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.173,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.183,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.135,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.122,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.081,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.079,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.074,
       "feature": "wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.113,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.283,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.105,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.095,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.063,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.06,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.058,
       "feature": "wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.083,
     "id": "N8",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.068,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.349,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.264,
       "feature": "crown color: white"
      },
      {
       "weight": 0.209,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.179,
       "feature": "bill shape: dagger"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.372,
   "id": "N9",
   "operator": "D",
   "name": "pure disjunction",
   "andness": -0.029,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.162,
     "shared_ref": "N3"
    },
    {
     "weight": 0.157,
     "shared_ref": "N4"
    },
    {
     "weight": 0.123,
     "shared_ref": "N5"
    },
    {
     "weight": 0.085,
     "shared_ref": "N6"
    },
    {
     "weight": 0.084,
     "shared_ref": "N7"
    },
    {
     "weight": 0.083,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R031
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.079,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.667,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.238,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.894,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.216,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.409,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.339,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.252,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.031,
     "id": "N4",
     "operator": "SD+",
     "name": "high soft disjunction",
     "andness": 0.26,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.959,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.011,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.01,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.006,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.005,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.004,
       "feature": "NOT back color: grey"
      }
     ]
    },
    {
     "weight": 0.025,
     "id": "N5",
     "operator": "SD+",
     "name": "high soft disjunction",
     "andness": 0.251,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.369,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.078,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.065,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.054,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.051,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.041,
       "feature": "NOT belly color: white"
      }
     ]
    },
    {
     "weight": 0.017,
     "id": "N6",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.063,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.158,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.151,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.082,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.081,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.077,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.071,
       "feature": "NOT primary color: buff"
      }
     ]
    },
    {
     "weight": 0.008,
     "id": "N7",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.769,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.528,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.372,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.043,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.024,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.015,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.006,
       "feature": "under tail color: white"
      }
     ]
    },
    {
     "weight": 0.007,
     "id": "N8",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.842,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.676,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.124,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.096,
       "feature": "eye color: black"
      },
      {
       "weight": 0.084,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.021,
       "feature": "primary color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.333,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.813,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.894,
     "shared_ref": "N3"
    },
    {
     "weight": 0.031,
     "shared_ref": "N4"
    },
    {
     "weight": 0.025,
     "shared_ref": "N5"
    },
    {
     "weight": 0.017,
     "shared_ref": "N6"
    },
    {
     "weight": 0.008,
     "shared_ref": "N7"
    },
    {
     "weight": 0.007,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R175
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.871,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.596,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.226,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.338,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.359,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.304,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.182,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.166,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.145,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.089,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.046,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.175,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.131,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.356,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.243,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.122,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.087,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.069,
       "feature": "back color: grey"
      },
      {
       "weight": 0.048,
       "feature": "back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.17,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.28,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.495,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.236,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.096,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.093,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.081,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.274,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.245,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.203,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.126,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.1,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.066,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.056,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.079,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.175,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.953,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.016,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.011,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.01,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.009,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.046,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.356,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.226,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.109,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.108,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.073,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.052,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.043,
       "feature": "NOT belly color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.404,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.845,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.338,
     "shared_ref": "N3"
    },
    {
     "weight": 0.175,
     "shared_ref": "N4"
    },
    {
     "weight": 0.17,
     "shared_ref": "N5"
    },
    {
     "weight": 0.109,
     "shared_ref": "N6"
    },
    {
     "weight": 0.079,
     "shared_ref": "N7"
    },
    {
     "weight": 0.046,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R106
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.123,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.581,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.925,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.117,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.335,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.142,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.119,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.112,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.112,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.11,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.108,
       "feature": "NOT bill color: buff"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N4",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.115,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.495,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.435,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.061,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.006,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.003,
       "feature": "primary color: buff"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.777,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.053,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.039,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.031,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.029,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.025,
       "feature": "NOT wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.586,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.044,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.036,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.035,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.031,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.029,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.029,
       "feature": "NOT under tail color: grey"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.037,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.572,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.428,
       "feature": "NOT primary color: brown"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.31,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.506,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.217,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.199,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.043,
       "feature": "bill color: black"
      },
      {
       "weight": 0.035,
       "feature": "breast pattern: solid"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.419,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.92,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.117,
     "shared_ref": "N3"
    },
    {
     "weight": 0.115,
     "shared_ref": "N4"
    },
    {
     "weight": 0.111,
     "shared_ref": "N5"
    },
    {
     "weight": 0.106,
     "shared_ref": "N6"
    },
    {
     "weight": 0.096,
     "shared_ref": "N7"
    },
    {
     "weight": 0.095,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R140
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.733,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.943,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.277,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.839,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.905,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.132,
       "feature": "wing color: white"
      },
      {
       "weight": 0.125,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.12,
       "feature": "breast color: black"
      },
      {
       "weight": 0.092,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.085,
       "feature": "belly color: white"
      },
      {
       "weight": 0.076,
       "feature": "wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.054,
     "id": "N4",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.029,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.172,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.116,
       "feature": "belly color: white"
      },
      {
       "weight": 0.105,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.088,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.084,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.063,
       "feature": "back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.019,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.098,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.344,
       "feature": "breast color: black"
      },
      {
       "weight": 0.264,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.183,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.102,
       "feature": "back color: black"
      },
      {
       "weight": 0.054,
       "feature": "wing color: black"
      },
      {
       "weight": 0.053,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.016,
     "id": "N6",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.858,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.325,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.258,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.213,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.204,
       "feature": "bill shape: dagger"
      }
     ]
    },
    {
     "weight": 0.013,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.075,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.214,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.199,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.169,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.14,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.083,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.056,
       "feature": "belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.009,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.593,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.063,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.06,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.056,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.055,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.05,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.049,
       "feature": "wing color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.057,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.135,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.839,
     "shared_ref": "N3"
    },
    {
     "weight": 0.054,
     "shared_ref": "N4"
    },
    {
     "weight": 0.019,
     "shared_ref": "N5"
    },
    {
     "weight": 0.016,
     "shared_ref": "N6"
    },
    {
     "weight": 0.013,
     "shared_ref": "N7"
    },
    {
     "weight": 0.009,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R069
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.558,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.719,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.037,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.163,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.384,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.418,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.183,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.126,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.125,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.055,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.048,
       "feature": "upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.153,
     "id": "N4",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.061,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.659,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.337,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.004,
       "feature": "wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.127,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.259,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.676,
       "feature": "leg color: black"
      },
      {
       "weight": 0.256,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.047,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.02,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.119,
     "id": "N6",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.014,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.351,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.321,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.289,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.025,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.003,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.002,
       "feature": "crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.118,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.067,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.605,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.339,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.046,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.002,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.002,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.002,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.209,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.827,
       "feature": "leg color: black"
      },
      {
       "weight": 0.121,
       "feature": "back color: brown"
      },
      {
       "weight": 0.052,
       "feature": "bill length: shorter than head"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.281,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.067,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.163,
     "shared_ref": "N3"
    },
    {
     "weight": 0.153,
     "shared_ref": "N4"
    },
    {
     "weight": 0.127,
     "shared_ref": "N5"
    },
    {
     "weight": 0.119,
     "shared_ref": "N6"
    },
    {
     "weight": 0.118,
     "shared_ref": "N7"
    },
    {
     "weight": 0.085,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R003
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.052,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.618,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.101,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.42,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.056,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.507,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.159,
       "feature": "belly color: black"
      },
      {
       "weight": 0.098,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.078,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.071,
       "feature": "breast color: black"
      },
      {
       "weight": 0.055,
       "feature": "throat color: black"
      }
     ]
    },
    {
     "weight": 0.269,
     "id": "N4",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.023,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.532,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.166,
       "feature": "belly color: black"
      },
      {
       "weight": 0.102,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.082,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.067,
       "feature": "bill color: black"
      },
      {
       "weight": 0.051,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.041,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.587,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.413,
       "feature": "back color: black"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.268,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.803,
       "feature": "breast color: black"
      },
      {
       "weight": 0.161,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.021,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.015,
       "feature": "forehead color: grey"
      }
     ]
    },
    {
     "weight": 0.038,
     "id": "N7",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.117,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.441,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.192,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.013,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.013,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.012,
       "feature": "back color: white"
      },
      {
       "weight": 0.012,
       "feature": "upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.021,
     "id": "N8",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.974,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.194,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.19,
       "feature": "crown color: black"
      },
      {
       "weight": 0.148,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.135,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.124,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.112,
       "feature": "belly pattern: solid"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.382,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.107,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.42,
     "shared_ref": "N3"
    },
    {
     "weight": 0.269,
     "shared_ref": "N4"
    },
    {
     "weight": 0.091,
     "shared_ref": "N5"
    },
    {
     "weight": 0.089,
     "shared_ref": "N6"
    },
    {
     "weight": 0.038,
     "shared_ref": "N7"
    },
    {
     "weight": 0.021,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R156
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.721,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.828,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.782,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.468,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.035,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.278,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.114,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.106,
       "feature": "belly color: black"
      },
      {
       "weight": 0.06,
       "feature": "breast color: black"
      },
      {
       "weight": 0.048,
       "feature": "throat color: black"
      },
      {
       "weight": 0.047,
       "feature": "underparts color: black"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.357,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.048,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.043,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.041,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.04,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.039,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.038,
       "feature": "underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.075,
     "id": "N5",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.205,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.322,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.048,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.037,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.031,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.031,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.029,
       "feature": "upperparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.071,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.334,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.627,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.257,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.071,
       "feature": "wing color: black"
      },
      {
       "weight": 0.046,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.07,
     "id": "N7",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.839,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.093,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.092,
       "feature": "breast color: black"
      },
      {
       "weight": 0.076,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.073,
       "feature": "throat color: black"
      },
      {
       "weight": 0.071,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.066,
       "feature": "wing pattern: solid"
      }
     ]
    },
    {
     "weight": 0.057,
     "id": "N8",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.18,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.188,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.12,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.046,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.036,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.031,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.031,
       "feature": "back pattern: striped"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.172,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.879,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.468,
     "shared_ref": "N3"
    },
    {
     "weight": 0.08,
     "shared_ref": "N4"
    },
    {
     "weight": 0.075,
     "shared_ref": "N5"
    },
    {
     "weight": 0.071,
     "shared_ref": "N6"
    },
    {
     "weight": 0.07,
     "shared_ref": "N7"
    },
    {
     "weight": 0.057,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R183
```json
{
 "id": "N1",
 "operator": "SC+",
 "name": "high soft conjunction",
 "andness": 0.745,
 "verbalization": "nice to have most",
 "children": [
  {
   "weight": 0.732,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.752,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.184,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.606,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.276,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.2,
       "feature": "breast color: black"
      },
      {
       "weight": 0.086,
       "feature": "leg color: black"
      },
      {
       "weight": 0.055,
       "feature": "nape color: black"
      },
      {
       "weight": 0.053,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.051,
       "feature": "throat color: black"
      }
     ]
    },
    {
     "weight": 0.151,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.555,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.311,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.174,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.097,
       "feature": "leg color: black"
      },
      {
       "weight": 0.087,
       "feature": "belly color: black"
      },
      {
       "weight": 0.061,
       "feature": "nape color: black"
      },
      {
       "weight": 0.059,
       "feature": "underparts color: black"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.545,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.135,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.041,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.041,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.038,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.036,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.035,
       "feature": "size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.334,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.118,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.112,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.106,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.103,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.103,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.096,
       "feature": "nape color: grey"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.448,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.13,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.069,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.053,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.041,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.041,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.037,
       "feature": "forehead color: grey"
      }
     ]
    },
    {
     "weight": 0.077,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.06,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.241,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.165,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.133,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.103,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.09,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.072,
       "feature": "forehead color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.268,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.049,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.184,
     "shared_ref": "N3"
    },
    {
     "weight": 0.151,
     "shared_ref": "N4"
    },
    {
     "weight": 0.087,
     "shared_ref": "N5"
    },
    {
     "weight": 0.085,
     "shared_ref": "N6"
    },
    {
     "weight": 0.082,
     "shared_ref": "N7"
    },
    {
     "weight": 0.077,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R176
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.744,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.549,
   "id": "N2",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.338,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.118,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.323,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.423,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.251,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.128,
       "feature": "throat color: black"
      },
      {
       "weight": 0.105,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.092,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.313,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.148,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.146,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.145,
       "feature": "belly color: black"
      },
      {
       "weight": 0.098,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.083,
       "feature": "breast color: black"
      },
      {
       "weight": 0.082,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.428,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.206,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.16,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.158,
       "feature": "belly color: black"
      },
      {
       "weight": 0.122,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.098,
       "feature": "nape color: black"
      },
      {
       "weight": 0.062,
       "feature": "throat color: black"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.211,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.172,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.155,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.131,
       "feature": "crown color: white"
      },
      {
       "weight": 0.125,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.119,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.113,
       "feature": "belly color: brown"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.128,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.166,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.157,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.142,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.131,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.109,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.105,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.309,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.134,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.11,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.079,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.05,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.047,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.045,
       "feature": "crown color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.451,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.348,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.118,
     "shared_ref": "N3"
    },
    {
     "weight": 0.102,
     "shared_ref": "N4"
    },
    {
     "weight": 0.098,
     "shared_ref": "N5"
    },
    {
     "weight": 0.094,
     "shared_ref": "N6"
    },
    {
     "weight": 0.094,
     "shared_ref": "N7"
    },
    {
     "weight": 0.088,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R027
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.187,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.611,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.038,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.1,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.355,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.236,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.108,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.085,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.069,
       "feature": "belly color: black"
      },
      {
       "weight": 0.058,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.049,
       "feature": "breast color: black"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.351,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.3,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.087,
       "feature": "belly color: black"
      },
      {
       "weight": 0.074,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.063,
       "feature": "breast color: black"
      },
      {
       "weight": 0.053,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.047,
       "feature": "throat color: black"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.418,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.114,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.107,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.101,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.097,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.091,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.091,
       "feature": "NOT upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.434,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.187,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.16,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.157,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.135,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.127,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.121,
       "feature": "NOT back color: white"
      }
     ]
    },
    {
     "weight": 0.083,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.496,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.046,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.035,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.034,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.033,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.031,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.031,
       "feature": "NOT forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.572,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.037,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.037,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.033,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.032,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.032,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.031,
       "feature": "NOT breast color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.389,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.491,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.1,
     "shared_ref": "N3"
    },
    {
     "weight": 0.098,
     "shared_ref": "N4"
    },
    {
     "weight": 0.097,
     "shared_ref": "N5"
    },
    {
     "weight": 0.096,
     "shared_ref": "N6"
    },
    {
     "weight": 0.083,
     "shared_ref": "N7"
    },
    {
     "weight": 0.08,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R136
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 0.974,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.588,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.961,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.233,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.248,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.621,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.212,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.119,
       "feature": "breast color: white"
      },
      {
       "weight": 0.048,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.221,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.357,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.347,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.197,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.104,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.096,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.084,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.054,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.147,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.255,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.321,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.259,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.182,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.096,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.078,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.041,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.304,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 1.0,
       "feature": "forehead color: brown"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.049,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.732,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.214,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.051,
       "feature": "throat color: white"
      },
      {
       "weight": 0.002,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.002,
       "feature": "NOT forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.069,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.035,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.768,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.079,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.071,
       "feature": "back color: brown"
      },
      {
       "weight": 0.049,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.023,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.004,
       "feature": "NOT underparts color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.412,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.053,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.233,
     "shared_ref": "N3"
    },
    {
     "weight": 0.221,
     "shared_ref": "N4"
    },
    {
     "weight": 0.147,
     "shared_ref": "N5"
    },
    {
     "weight": 0.094,
     "shared_ref": "N6"
    },
    {
     "weight": 0.073,
     "shared_ref": "N7"
    },
    {
     "weight": 0.069,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R071
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.574,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.81,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.225,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.129,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.47,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.361,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.201,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.193,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.062,
       "feature": "nape color: black"
      },
      {
       "weight": 0.043,
       "feature": "back color: black"
      },
      {
       "weight": 0.039,
       "feature": "upper tail color: black"
      }
     ]
    },
    {
     "weight": 0.122,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.479,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.348,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.194,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.187,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.072,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.07,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.065,
       "feature": "breast color: black"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N5",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.098,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.25,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.234,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.222,
       "feature": "belly color: black"
      },
      {
       "weight": 0.158,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.135,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N6",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.112,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.306,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.272,
       "feature": "belly color: black"
      },
      {
       "weight": 0.198,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.174,
       "feature": "breast color: black"
      },
      {
       "weight": 0.05,
       "feature": "breast color: grey"
      }
     ]
    },
    {
     "weight": 0.099,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.072,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.777,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.223,
       "feature": "throat color: grey"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.537,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.154,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.154,
       "feature": "throat color: black"
      },
      {
       "weight": 0.125,
       "feature": "nape color: black"
      },
      {
       "weight": 0.091,
       "feature": "bill color: black"
      },
      {
       "weight": 0.088,
       "feature": "crown color: black"
      },
      {
       "weight": 0.086,
       "feature": "back color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.19,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.753,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.129,
     "shared_ref": "N3"
    },
    {
     "weight": 0.122,
     "shared_ref": "N4"
    },
    {
     "weight": 0.11,
     "shared_ref": "N5"
    },
    {
     "weight": 0.103,
     "shared_ref": "N6"
    },
    {
     "weight": 0.099,
     "shared_ref": "N7"
    },
    {
     "weight": 0.095,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R019
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.845,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.522,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.139,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.191,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.635,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.179,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.138,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.103,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.085,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.061,
       "feature": "breast color: black"
      },
      {
       "weight": 0.057,
       "feature": "underparts color: black"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N4",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.222,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.088,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.079,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.074,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.065,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.065,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.065,
       "feature": "NOT wing color: brown"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.175,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.091,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.089,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.084,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.077,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.077,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.072,
       "feature": "NOT primary color: grey"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.372,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.072,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.065,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.061,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.061,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.059,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.059,
       "feature": "size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.142,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.23,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.173,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.168,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.152,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.149,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.126,
       "feature": "leg color: buff"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.296,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.047,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.045,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.04,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.039,
       "feature": "crown color: white"
      },
      {
       "weight": 0.038,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.037,
       "feature": "NOT back pattern: striped"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.478,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.219,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.191,
     "shared_ref": "N3"
    },
    {
     "weight": 0.107,
     "shared_ref": "N4"
    },
    {
     "weight": 0.105,
     "shared_ref": "N5"
    },
    {
     "weight": 0.104,
     "shared_ref": "N6"
    },
    {
     "weight": 0.103,
     "shared_ref": "N7"
    },
    {
     "weight": 0.098,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R122
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.098,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.502,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.002,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.114,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.279,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.268,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.231,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.225,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.139,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.069,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.069,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.69,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.102,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.059,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.054,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.054,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.052,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "NOT nape color: buff"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.638,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.144,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.129,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.1,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.096,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.094,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.093,
       "feature": "NOT upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.633,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.177,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.087,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.077,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.066,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.066,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.064,
       "feature": "NOT size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.7,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.082,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.071,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.071,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.071,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.07,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.066,
       "feature": "NOT bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.625,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.13,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.122,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.121,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.076,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.069,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.058,
       "feature": "NOT back pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.498,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.131,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.114,
     "shared_ref": "N3"
    },
    {
     "weight": 0.093,
     "shared_ref": "N4"
    },
    {
     "weight": 0.091,
     "shared_ref": "N5"
    },
    {
     "weight": 0.09,
     "shared_ref": "N6"
    },
    {
     "weight": 0.09,
     "shared_ref": "N7"
    },
    {
     "weight": 0.089,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R133
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.146,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.563,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.152,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.949,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.228,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.297,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.165,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.164,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.107,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.093,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.09,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.014,
     "id": "N4",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.119,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.711,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.279,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.005,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.005,
       "feature": "NOT wing pattern: spotted"
      }
     ]
    },
    {
     "weight": 0.012,
     "id": "N5",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.068,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.606,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.367,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.013,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.005,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.005,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.004,
       "feature": "NOT back color: buff"
      }
     ]
    },
    {
     "weight": 0.009,
     "id": "N6",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.751,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.209,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.202,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.187,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.105,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.075,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.073,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.004,
     "id": "N7",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.666,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.218,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.116,
       "feature": "NOT underparts color: white"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N8",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.246,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.799,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.067,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.067,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.051,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.016,
       "feature": "crown color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.437,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.89,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.949,
     "shared_ref": "N3"
    },
    {
     "weight": 0.014,
     "shared_ref": "N4"
    },
    {
     "weight": 0.012,
     "shared_ref": "N5"
    },
    {
     "weight": 0.009,
     "shared_ref": "N6"
    },
    {
     "weight": 0.004,
     "shared_ref": "N7"
    },
    {
     "weight": 0.002,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R028
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.382,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.763,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.845,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.172,
     "id": "N3",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.079,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.288,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.252,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.2,
       "feature": "back color: brown"
      },
      {
       "weight": 0.154,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.103,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.001,
       "feature": "NOT upperparts color: white"
      }
     ]
    },
    {
     "weight": 0.166,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.09,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 1.0,
       "feature": "primary color: white"
      }
     ]
    },
    {
     "weight": 0.155,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.1,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.275,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.229,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.186,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.166,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.144,
       "feature": "NOT forehead color: brown"
      }
     ]
    },
    {
     "weight": 0.127,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.474,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.036,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.034,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.031,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.027,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.026,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.026,
       "feature": "NOT wing color: buff"
      }
     ]
    },
    {
     "weight": 0.119,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.536,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.036,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.033,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.032,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.031,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.031,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.029,
       "feature": "NOT leg color: buff"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.322,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.313,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.148,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.13,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.122,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.098,
       "feature": "primary color: white"
      },
      {
       "weight": 0.067,
       "feature": "wing pattern: solid"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.237,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.906,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.172,
     "shared_ref": "N3"
    },
    {
     "weight": 0.166,
     "shared_ref": "N4"
    },
    {
     "weight": 0.155,
     "shared_ref": "N5"
    },
    {
     "weight": 0.127,
     "shared_ref": "N6"
    },
    {
     "weight": 0.119,
     "shared_ref": "N7"
    },
    {
     "weight": 0.104,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R113
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.212,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.561,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.153,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.175,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.332,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.205,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.124,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.123,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.102,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.091,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.074,
       "feature": "underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.046,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.358,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.214,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.183,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.07,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.067,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.029,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.239,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.368,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.319,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.11,
       "feature": "leg color: black"
      },
      {
       "weight": 0.093,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.074,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.036,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.362,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.651,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.08,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.054,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.051,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.049,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.043,
       "feature": "NOT primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.404,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.656,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.08,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.078,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.053,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.052,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.043,
       "feature": "NOT primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.416,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.13,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.128,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.123,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.11,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.1,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.091,
       "feature": "NOT back color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.439,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.206,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.175,
     "shared_ref": "N3"
    },
    {
     "weight": 0.108,
     "shared_ref": "N4"
    },
    {
     "weight": 0.105,
     "shared_ref": "N5"
    },
    {
     "weight": 0.1,
     "shared_ref": "N6"
    },
    {
     "weight": 0.098,
     "shared_ref": "N7"
    },
    {
     "weight": 0.09,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R073
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.984,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.523,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.169,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.201,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.213,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.208,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.149,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.096,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.077,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.057,
       "feature": "NOT back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N4",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.221,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.882,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.038,
       "feature": "leg color: black"
      },
      {
       "weight": 0.035,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.03,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.015,
       "feature": "bill color: grey"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N5",
     "operator": "SD",
     "name": "medium soft disjunction",
     "andness": 0.331,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 1.0,
       "feature": "back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N6",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.247,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.791,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.15,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.033,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.021,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.004,
       "feature": "NOT upper tail color: buff"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.833,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.064,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.058,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.047,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.04,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.037,
       "feature": "eye color: black"
      },
      {
       "weight": 0.035,
       "feature": "throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.063,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.68,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.057,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.056,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.054,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.052,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.05,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "bill shape: dagger"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.477,
   "id": "N9",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.159,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.201,
     "shared_ref": "N3"
    },
    {
     "weight": 0.109,
     "shared_ref": "N4"
    },
    {
     "weight": 0.107,
     "shared_ref": "N5"
    },
    {
     "weight": 0.104,
     "shared_ref": "N6"
    },
    {
     "weight": 0.064,
     "shared_ref": "N7"
    },
    {
     "weight": 0.063,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R186
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.959,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.699,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.08,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.17,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.271,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.223,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.166,
       "feature": "belly color: black"
      },
      {
       "weight": 0.159,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.134,
       "feature": "back color: buff"
      },
      {
       "weight": 0.078,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.073,
       "feature": "underparts color: black"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.269,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.155,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.154,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.115,
       "feature": "belly color: black"
      },
      {
       "weight": 0.11,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.086,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.05,
       "feature": "underparts color: black"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.364,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.412,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.268,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.175,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.146,
       "feature": "NOT size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.475,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.253,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.189,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.137,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.134,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.121,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.091,
       "feature": "NOT forehead color: white"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.452,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.473,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.113,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.04,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.036,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.027,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.025,
       "feature": "NOT upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.659,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.243,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.217,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.216,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.202,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.122,
       "feature": "NOT size: very small (3 - 5 in)"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.301,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.119,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.17,
     "shared_ref": "N3"
    },
    {
     "weight": 0.105,
     "shared_ref": "N4"
    },
    {
     "weight": 0.103,
     "shared_ref": "N5"
    },
    {
     "weight": 0.103,
     "shared_ref": "N6"
    },
    {
     "weight": 0.098,
     "shared_ref": "N7"
    },
    {
     "weight": 0.097,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R144
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.958,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.745,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.992,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.121,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.632,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.083,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.074,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.067,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.058,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.058,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.052,
       "feature": "NOT upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.119,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.724,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.044,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.043,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.043,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.041,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.04,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.037,
       "feature": "NOT back pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.604,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.158,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.118,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.107,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.075,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.074,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.071,
       "feature": "NOT forehead color: white"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.377,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.243,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.204,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.175,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.123,
       "feature": "throat color: white"
      },
      {
       "weight": 0.081,
       "feature": "breast color: white"
      },
      {
       "weight": 0.08,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.677,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.065,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.063,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.047,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.046,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.045,
       "feature": "NOT back color: black"
      }
     ]
    },
    {
     "weight": 0.099,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.395,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.223,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.192,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.135,
       "feature": "throat color: white"
      },
      {
       "weight": 0.098,
       "feature": "belly color: white"
      },
      {
       "weight": 0.088,
       "feature": "breast color: white"
      },
      {
       "weight": 0.087,
       "feature": "tail shape: notched tail"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.255,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.322,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.121,
     "shared_ref": "N3"
    },
    {
     "weight": 0.119,
     "shared_ref": "N4"
    },
    {
     "weight": 0.117,
     "shared_ref": "N5"
    },
    {
     "weight": 0.116,
     "shared_ref": "N6"
    },
    {
     "weight": 0.115,
     "shared_ref": "N7"
    },
    {
     "weight": 0.099,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R178
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.593,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.798,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.268,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.564,
     "id": "N3",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.836,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.265,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.117,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.106,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.099,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.095,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.067,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.282,
     "id": "N4",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.823,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.289,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.128,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.116,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.104,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.095,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.073,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.03,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.372,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.281,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.09,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.069,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.066,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.066,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.065,
       "feature": "NOT crown color: blue"
      }
     ]
    },
    {
     "weight": 0.027,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.516,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.074,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.062,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.058,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.048,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.045,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.041,
       "feature": "NOT belly color: brown"
      }
     ]
    },
    {
     "weight": 0.023,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.481,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.094,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.091,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.076,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.054,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.053,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.052,
       "feature": "NOT breast color: brown"
      }
     ]
    },
    {
     "weight": 0.02,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.26,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.138,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.121,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.081,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.071,
       "feature": "back color: brown"
      },
      {
       "weight": 0.066,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.064,
       "feature": "primary color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.202,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.078,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.564,
     "shared_ref": "N3"
    },
    {
     "weight": 0.282,
     "shared_ref": "N4"
    },
    {
     "weight": 0.03,
     "shared_ref": "N5"
    },
    {
     "weight": 0.027,
     "shared_ref": "N6"
    },
    {
     "weight": 0.023,
     "shared_ref": "N7"
    },
    {
     "weight": 0.02,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R055
```json
{
 "id": "N1",
 "operator": "HD",
 "name": "medium hard disjunction",
 "andness": 0.129,
 "verbalization": "enough to have any",
 "children": [
  {
   "weight": 0.962,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.054,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.453,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.311,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.171,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.135,
       "feature": "throat color: white"
      },
      {
       "weight": 0.122,
       "feature": "breast color: white"
      },
      {
       "weight": 0.107,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.099,
       "feature": "back color: grey"
      },
      {
       "weight": 0.089,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.267,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.288,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.184,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.145,
       "feature": "throat color: white"
      },
      {
       "weight": 0.131,
       "feature": "breast color: white"
      },
      {
       "weight": 0.115,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.107,
       "feature": "back color: grey"
      },
      {
       "weight": 0.084,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.058,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.396,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.77,
       "feature": "leg color: black"
      },
      {
       "weight": 0.2,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.03,
       "feature": "NOT wing color: brown"
      }
     ]
    },
    {
     "weight": 0.039,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.394,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.252,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.185,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.132,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.104,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.102,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.059,
       "feature": "NOT size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.031,
     "id": "N7",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.987,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.556,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.295,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.148,
       "feature": "nape color: grey"
      }
     ]
    },
    {
     "weight": 0.03,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.512,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.285,
       "feature": "leg color: black"
      },
      {
       "weight": 0.158,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.111,
       "feature": "bill color: black"
      },
      {
       "weight": 0.061,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.053,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.043,
       "feature": "NOT wing color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.038,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.758,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.453,
     "shared_ref": "N3"
    },
    {
     "weight": 0.267,
     "shared_ref": "N4"
    },
    {
     "weight": 0.058,
     "shared_ref": "N5"
    },
    {
     "weight": 0.039,
     "shared_ref": "N6"
    },
    {
     "weight": 0.031,
     "shared_ref": "N7"
    },
    {
     "weight": 0.03,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R032
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.711,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.682,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.122,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.202,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.764,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 1.0,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.128,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.805,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.534,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.25,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.215,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N5",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.148,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.684,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.262,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.048,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.004,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.002,
       "feature": "NOT primary color: brown"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N6",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.774,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.556,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.26,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.184,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N7",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.148,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.987,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.004,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.004,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.003,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.002,
       "feature": "NOT throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N8",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.009,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.032,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.032,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.032,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.031,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.03,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.03,
       "feature": "NOT nape color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.318,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.929,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.202,
     "shared_ref": "N3"
    },
    {
     "weight": 0.128,
     "shared_ref": "N4"
    },
    {
     "weight": 0.117,
     "shared_ref": "N5"
    },
    {
     "weight": 0.109,
     "shared_ref": "N6"
    },
    {
     "weight": 0.107,
     "shared_ref": "N7"
    },
    {
     "weight": 0.103,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R124
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.408,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.679,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.118,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.164,
     "id": "N3",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.175,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.269,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.182,
       "feature": "bill color: black"
      },
      {
       "weight": 0.144,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.134,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.066,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.062,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.134,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.645,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.19,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.138,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.089,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.082,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.073,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.068,
       "feature": "NOT under tail color: brown"
      }
     ]
    },
    {
     "weight": 0.131,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.709,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.099,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.093,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.064,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.052,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.051,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.048,
       "feature": "NOT primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.123,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.725,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.103,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.061,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.05,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.046,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.045,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.043,
       "feature": "NOT back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.762,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.071,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.066,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.055,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.053,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.037,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.036,
       "feature": "NOT upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.232,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 1.0,
       "feature": "NOT breast color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.321,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.353,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.164,
     "shared_ref": "N3"
    },
    {
     "weight": 0.134,
     "shared_ref": "N4"
    },
    {
     "weight": 0.131,
     "shared_ref": "N5"
    },
    {
     "weight": 0.123,
     "shared_ref": "N6"
    },
    {
     "weight": 0.116,
     "shared_ref": "N7"
    },
    {
     "weight": 0.082,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R082
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.855,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.596,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.041,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.412,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.472,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.25,
       "feature": "leg color: black"
      },
      {
       "weight": 0.246,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.211,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.081,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.055,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.045,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.069,
     "id": "N4",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.059,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.372,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.243,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.211,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.174,
       "feature": "under tail color: buff"
      }
     ]
    },
    {
     "weight": 0.067,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.081,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.299,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.211,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.161,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.125,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.105,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.1,
       "feature": "tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.066,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.213,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.223,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.217,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.16,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.118,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.117,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.114,
       "feature": "underparts color: grey"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.299,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.217,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.138,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.134,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.103,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.103,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.1,
       "feature": "crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.063,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.354,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.213,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.206,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.125,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.097,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.09,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.076,
       "feature": "breast pattern: striped"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.404,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.318,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.412,
     "shared_ref": "N3"
    },
    {
     "weight": 0.069,
     "shared_ref": "N4"
    },
    {
     "weight": 0.067,
     "shared_ref": "N5"
    },
    {
     "weight": 0.066,
     "shared_ref": "N6"
    },
    {
     "weight": 0.064,
     "shared_ref": "N7"
    },
    {
     "weight": 0.063,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R077
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.935,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.644,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.099,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.235,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.121,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.268,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.256,
       "feature": "leg color: black"
      },
      {
       "weight": 0.145,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.09,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.079,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.058,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.132,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.337,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.119,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.091,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.058,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.051,
       "feature": "nape color: white"
      },
      {
       "weight": 0.048,
       "feature": "NOT back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.083,
     "id": "N5",
     "operator": "A",
     "name": "arithmetic mean",
     "andness": 0.531,
     "verbalization": "nice to have",
     "children": [
      {
       "weight": 0.462,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.37,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.127,
       "feature": "leg color: black"
      },
      {
       "weight": 0.041,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.077,
     "id": "N6",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.719,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.508,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.437,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.056,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.066,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.355,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.092,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.092,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.083,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.082,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.074,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.071,
       "feature": "NOT belly color: white"
      }
     ]
    },
    {
     "weight": 0.06,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.388,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.166,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.106,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.054,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.042,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.041,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.04,
       "feature": "crown color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.356,
   "id": "N9",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.128,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.235,
     "shared_ref": "N3"
    },
    {
     "weight": 0.114,
     "shared_ref": "N4"
    },
    {
     "weight": 0.083,
     "shared_ref": "N5"
    },
    {
     "weight": 0.077,
     "shared_ref": "N6"
    },
    {
     "weight": 0.066,
     "shared_ref": "N7"
    },
    {
     "weight": 0.06,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R054
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.17,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.997,
   "id": "N2",
   "operator": "HHD",
   "name": "high hyper-disjunction",
   "andness": -0.689,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.718,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.814,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.252,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.132,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.057,
       "feature": "nape color: black"
      },
      {
       "weight": 0.052,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.046,
       "feature": "leg color: black"
      },
      {
       "weight": 0.045,
       "feature": "underparts color: black"
      }
     ]
    },
    {
     "weight": 0.272,
     "id": "N4",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.886,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.252,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.132,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.057,
       "feature": "nape color: black"
      },
      {
       "weight": 0.052,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.046,
       "feature": "leg color: black"
      },
      {
       "weight": 0.045,
       "feature": "underparts color: black"
      }
     ]
    },
    {
     "weight": 0.001,
     "id": "N5",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.282,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.16,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.152,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.152,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.117,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.079,
       "feature": "back color: yellow"
      }
     ]
    },
    {
     "weight": 0.001,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.224,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.388,
       "feature": "eye color: black"
      },
      {
       "weight": 0.295,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.184,
       "feature": "nape color: white"
      },
      {
       "weight": 0.133,
       "feature": "crown color: blue"
      }
     ]
    },
    {
     "weight": 0.001,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.125,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.522,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.293,
       "feature": "NOT bill length: about the same as head"
      },
      {
       "weight": 0.185,
       "feature": "belly color: black"
      }
     ]
    },
    {
     "weight": 0.001,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.188,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.266,
       "feature": "nape color: white"
      },
      {
       "weight": 0.251,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.248,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.234,
       "feature": "forehead color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.003,
   "id": "N9",
   "operator": "A",
   "name": "arithmetic mean",
   "andness": 0.5,
   "verbalization": "nice to have",
   "children": [
    {
     "weight": 0.718,
     "shared_ref": "N3"
    },
    {
     "weight": 0.272,
     "shared_ref": "N4"
    },
    {
     "weight": 0.001,
     "shared_ref": "N5"
    },
    {
     "weight": 0.001,
     "shared_ref": "N6"
    },
    {
     "weight": 0.001,
     "shared_ref": "N7"
    },
    {
     "weight": 0.001,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R062
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.601,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.555,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.109,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.187,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.458,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.218,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.161,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.098,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.082,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.071,
       "feature": "crown color: white"
      },
      {
       "weight": 0.054,
       "feature": "forehead color: white"
      }
     ]
    },
    {
     "weight": 0.126,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.493,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.238,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.144,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.08,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.066,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.061,
       "feature": "nape color: white"
      },
      {
       "weight": 0.058,
       "feature": "primary color: grey"
      }
     ]
    },
    {
     "weight": 0.112,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.117,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.523,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.17,
       "feature": "crown color: white"
      },
      {
       "weight": 0.106,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.094,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.09,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.018,
       "feature": "shape: duck-like"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N6",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.009,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.305,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.165,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.111,
       "feature": "back color: grey"
      },
      {
       "weight": 0.086,
       "feature": "breast color: white"
      },
      {
       "weight": 0.086,
       "feature": "belly color: white"
      },
      {
       "weight": 0.083,
       "feature": "throat color: white"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.416,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.077,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.064,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.056,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.055,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.051,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.05,
       "feature": "NOT belly color: buff"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.278,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.063,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.063,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.056,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.056,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.054,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.054,
       "feature": "under tail color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.445,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.115,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.187,
     "shared_ref": "N3"
    },
    {
     "weight": 0.126,
     "shared_ref": "N4"
    },
    {
     "weight": 0.112,
     "shared_ref": "N5"
    },
    {
     "weight": 0.091,
     "shared_ref": "N6"
    },
    {
     "weight": 0.088,
     "shared_ref": "N7"
    },
    {
     "weight": 0.084,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R084
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.383,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.802,
   "id": "N2",
   "operator": "HHD",
   "name": "high hyper-disjunction",
   "andness": -0.32,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.963,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.916,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.229,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.207,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.201,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.175,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.143,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.022,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.007,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.06,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.324,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.264,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.138,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.104,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.069,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.06,
       "feature": "upperparts color: brown"
      }
     ]
    },
    {
     "weight": 0.006,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.204,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.374,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.329,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.298,
       "feature": "back color: buff"
      }
     ]
    },
    {
     "weight": 0.006,
     "id": "N6",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.032,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.878,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.122,
       "feature": "nape color: brown"
      }
     ]
    },
    {
     "weight": 0.004,
     "id": "N7",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.159,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.531,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.395,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.074,
       "feature": "nape color: brown"
      }
     ]
    },
    {
     "weight": 0.003,
     "id": "N8",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.752,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.204,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.177,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.175,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.11,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.107,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.101,
       "feature": "back color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.198,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.778,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.963,
     "shared_ref": "N3"
    },
    {
     "weight": 0.007,
     "shared_ref": "N4"
    },
    {
     "weight": 0.006,
     "shared_ref": "N5"
    },
    {
     "weight": 0.006,
     "shared_ref": "N6"
    },
    {
     "weight": 0.004,
     "shared_ref": "N7"
    },
    {
     "weight": 0.003,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R020
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.475,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.586,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.051,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.144,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.291,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.234,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.196,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.139,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.11,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.071,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.069,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.118,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.367,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.763,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.237,
       "feature": "crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.404,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.178,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.1,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.074,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.072,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.063,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.06,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.223,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.987,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.004,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.003,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.002,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.002,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.002,
       "feature": "NOT crown color: white"
      }
     ]
    },
    {
     "weight": 0.074,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.298,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.305,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.121,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.112,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.112,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.096,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.089,
       "feature": "NOT forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.501,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.097,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.088,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.083,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.073,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.069,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.064,
       "feature": "NOT size: medium (9 - 16 in)"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.414,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.012,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.144,
     "shared_ref": "N3"
    },
    {
     "weight": 0.118,
     "shared_ref": "N4"
    },
    {
     "weight": 0.085,
     "shared_ref": "N5"
    },
    {
     "weight": 0.08,
     "shared_ref": "N6"
    },
    {
     "weight": 0.074,
     "shared_ref": "N7"
    },
    {
     "weight": 0.073,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R070
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.46,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.7,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.943,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.112,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.242,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.189,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.165,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.121,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.099,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.08,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.072,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N4",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.03,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.313,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.251,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.241,
       "feature": "nape color: black"
      },
      {
       "weight": 0.196,
       "feature": "crown color: black"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.483,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.159,
       "feature": "back color: buff"
      },
      {
       "weight": 0.1,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.092,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.091,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.07,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.065,
       "feature": "upper tail color: black"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.4,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.244,
       "feature": "back color: buff"
      },
      {
       "weight": 0.203,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.154,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.142,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.099,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.09,
       "feature": "leg color: buff"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.442,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.059,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.049,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.041,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.04,
       "feature": "NOT bill color: grey"
      },
      {
       "weight": 0.039,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.039,
       "feature": "NOT crown color: blue"
      }
     ]
    },
    {
     "weight": 0.079,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.371,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.086,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.05,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.049,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.043,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.041,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.04,
       "feature": "NOT nape color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.3,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.109,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.112,
     "shared_ref": "N3"
    },
    {
     "weight": 0.105,
     "shared_ref": "N4"
    },
    {
     "weight": 0.09,
     "shared_ref": "N5"
    },
    {
     "weight": 0.082,
     "shared_ref": "N6"
    },
    {
     "weight": 0.08,
     "shared_ref": "N7"
    },
    {
     "weight": 0.079,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R148
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.553,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.752,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.967,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.906,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.156,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.132,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.117,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.086,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.08,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.078,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.073,
       "feature": "belly color: black"
      }
     ]
    },
    {
     "weight": 0.014,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.072,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.281,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.074,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.02,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.02,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.018,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.017,
       "feature": "size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.013,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.128,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.027,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.027,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.025,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.025,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.025,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.024,
       "feature": "NOT shape: duck-like"
      }
     ]
    },
    {
     "weight": 0.012,
     "id": "N6",
     "operator": "SD",
     "name": "medium soft disjunction",
     "andness": 0.373,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.406,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.135,
       "feature": "primary color: black"
      },
      {
       "weight": 0.108,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.106,
       "feature": "wing color: black"
      },
      {
       "weight": 0.105,
       "feature": "throat color: black"
      },
      {
       "weight": 0.104,
       "feature": "crown color: black"
      }
     ]
    },
    {
     "weight": 0.011,
     "id": "N7",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.249,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.145,
       "feature": "primary color: black"
      },
      {
       "weight": 0.13,
       "feature": "breast color: black"
      },
      {
       "weight": 0.116,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.114,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.113,
       "feature": "wing color: black"
      },
      {
       "weight": 0.111,
       "feature": "crown color: black"
      }
     ]
    },
    {
     "weight": 0.009,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.321,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.745,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.255,
       "feature": "back pattern: solid"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.248,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.005,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.906,
     "shared_ref": "N3"
    },
    {
     "weight": 0.014,
     "shared_ref": "N4"
    },
    {
     "weight": 0.013,
     "shared_ref": "N5"
    },
    {
     "weight": 0.012,
     "shared_ref": "N6"
    },
    {
     "weight": 0.011,
     "shared_ref": "N7"
    },
    {
     "weight": 0.009,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R125
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.585,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.659,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.199,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.14,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.485,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.14,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.138,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.124,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.073,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.065,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.063,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.12,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.222,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.26,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.225,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.138,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.137,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.072,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.069,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.119,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.392,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.157,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.136,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.093,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.092,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.083,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.049,
       "feature": "size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.462,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.085,
       "feature": "NOT eye color: black"
      },
      {
       "weight": 0.054,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.052,
       "feature": "NOT upper tail color: black"
      },
      {
       "weight": 0.05,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.048,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.041,
       "feature": "NOT belly color: white"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.462,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.107,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.055,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.049,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.048,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.045,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.04,
       "feature": "NOT forehead color: white"
      }
     ]
    },
    {
     "weight": 0.072,
     "id": "N8",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.963,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.262,
       "feature": "leg color: black"
      },
      {
       "weight": 0.25,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.185,
       "feature": "wing color: black"
      },
      {
       "weight": 0.163,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.141,
       "feature": "upperparts color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.341,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.427,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.14,
     "shared_ref": "N3"
    },
    {
     "weight": 0.12,
     "shared_ref": "N4"
    },
    {
     "weight": 0.119,
     "shared_ref": "N5"
    },
    {
     "weight": 0.076,
     "shared_ref": "N6"
    },
    {
     "weight": 0.073,
     "shared_ref": "N7"
    },
    {
     "weight": 0.072,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R100
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.424,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.768,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": -0.008,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.428,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.518,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.347,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.152,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.146,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.103,
       "feature": "bill color: black"
      },
      {
       "weight": 0.084,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.079,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N4",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.208,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.988,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.004,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.003,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.003,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.001,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.001,
       "feature": "NOT wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.369,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.291,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.28,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.162,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.153,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.114,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N6",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.146,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.879,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.095,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.012,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.008,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.002,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.001,
       "feature": "NOT breast color: grey"
      }
     ]
    },
    {
     "weight": 0.075,
     "id": "N7",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.155,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 1.0,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N8",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.089,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.254,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.236,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.12,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.114,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.108,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.094,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.232,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.865,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.428,
     "shared_ref": "N3"
    },
    {
     "weight": 0.107,
     "shared_ref": "N4"
    },
    {
     "weight": 0.106,
     "shared_ref": "N5"
    },
    {
     "weight": 0.105,
     "shared_ref": "N6"
    },
    {
     "weight": 0.075,
     "shared_ref": "N7"
    },
    {
     "weight": 0.064,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R039
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.183,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.573,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.879,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.971,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.119,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.385,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.123,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.118,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.103,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.1,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.079,
       "feature": "upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.012,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.361,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.239,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.23,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.201,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.195,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.046,
       "feature": "back color: brown"
      },
      {
       "weight": 0.03,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.59,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.233,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.107,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.097,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.096,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.084,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.067,
       "feature": "NOT upperparts color: black"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.626,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.212,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.076,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.072,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.068,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.061,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.057,
       "feature": "head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.647,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.294,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.168,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.114,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.079,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.078,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.073,
       "feature": "back color: white"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.797,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.109,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.091,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.09,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.079,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.07,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.058,
       "feature": "underparts color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.427,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.15,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.971,
     "shared_ref": "N3"
    },
    {
     "weight": 0.012,
     "shared_ref": "N4"
    },
    {
     "weight": 0.002,
     "shared_ref": "N5"
    },
    {
     "weight": 0.002,
     "shared_ref": "N6"
    },
    {
     "weight": 0.002,
     "shared_ref": "N7"
    },
    {
     "weight": 0.002,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R165
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.931,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.73,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.962,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.117,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.709,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.065,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.064,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.055,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.048,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.042,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.037,
       "feature": "NOT upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.667,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.071,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.059,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.05,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.046,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.039,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.037,
       "feature": "NOT wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.36,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.783,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.029,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.027,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.024,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.021,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.019,
       "feature": "NOT crown color: grey"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.542,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.121,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.117,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.103,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.101,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.057,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.039,
       "feature": "NOT back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.497,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.086,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.084,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.071,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.064,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.045,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.044,
       "feature": "NOT breast color: brown"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.326,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.25,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.119,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.091,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.056,
       "feature": "back color: grey"
      },
      {
       "weight": 0.052,
       "feature": "crown color: black"
      },
      {
       "weight": 0.051,
       "feature": "upperparts color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.27,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.244,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.117,
     "shared_ref": "N3"
    },
    {
     "weight": 0.115,
     "shared_ref": "N4"
    },
    {
     "weight": 0.107,
     "shared_ref": "N5"
    },
    {
     "weight": 0.104,
     "shared_ref": "N6"
    },
    {
     "weight": 0.102,
     "shared_ref": "N7"
    },
    {
     "weight": 0.101,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R091
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.084,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.957,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.095,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.158,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.187,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.384,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.192,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.146,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.13,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.04,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.027,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.12,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.099,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.708,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.145,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.059,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.034,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.023,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.014,
       "feature": "NOT nape color: white"
      }
     ]
    },
    {
     "weight": 0.119,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.144,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.427,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.214,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.162,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.065,
       "feature": "wing color: black"
      },
      {
       "weight": 0.036,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.028,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.131,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.406,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.215,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.136,
       "feature": "eye color: black"
      },
      {
       "weight": 0.123,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.05,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.039,
       "feature": "NOT primary color: brown"
      }
     ]
    },
    {
     "weight": 0.065,
     "id": "N7",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.858,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.643,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.357,
       "feature": "bill shape: cone"
      }
     ]
    },
    {
     "weight": 0.062,
     "id": "N8",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.814,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.552,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.285,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.158,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.002,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.002,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.001,
       "feature": "NOT underparts color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.043,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.833,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.158,
     "shared_ref": "N3"
    },
    {
     "weight": 0.12,
     "shared_ref": "N4"
    },
    {
     "weight": 0.119,
     "shared_ref": "N5"
    },
    {
     "weight": 0.116,
     "shared_ref": "N6"
    },
    {
     "weight": 0.065,
     "shared_ref": "N7"
    },
    {
     "weight": 0.062,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R067
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.178,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.62,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.886,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.13,
     "id": "N3",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.268,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.468,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.204,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.2,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.127,
       "feature": "NOT primary color: grey"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.489,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.287,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.158,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.152,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.14,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.135,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.128,
       "feature": "NOT forehead color: white"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.414,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.375,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.328,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.182,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.115,
       "feature": "NOT primary color: grey"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.662,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.075,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.07,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.064,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.056,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.049,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.048,
       "feature": "NOT throat color: white"
      }
     ]
    },
    {
     "weight": 0.099,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.687,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.08,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.068,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.061,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.055,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.054,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.051,
       "feature": "NOT bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.678,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.074,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.07,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.067,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.062,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.057,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.053,
       "feature": "NOT throat color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.38,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.532,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.13,
     "shared_ref": "N3"
    },
    {
     "weight": 0.105,
     "shared_ref": "N4"
    },
    {
     "weight": 0.104,
     "shared_ref": "N5"
    },
    {
     "weight": 0.102,
     "shared_ref": "N6"
    },
    {
     "weight": 0.099,
     "shared_ref": "N7"
    },
    {
     "weight": 0.097,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R188
```json
{
 "id": "N1",
 "operator": "HD-",
 "name": "low hard disjunction",
 "andness": 0.214,
 "verbalization": "enough to have any",
 "children": [
  {
   "weight": 0.929,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.125,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.6,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.354,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.156,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.122,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.107,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.102,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.088,
       "feature": "back color: grey"
      },
      {
       "weight": 0.083,
       "feature": "under tail color: grey"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.378,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.266,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.208,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.183,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.127,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.121,
       "feature": "leg color: black"
      },
      {
       "weight": 0.059,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.043,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.804,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.077,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.076,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.075,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.068,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.065,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.063,
       "feature": "nape color: buff"
      }
     ]
    },
    {
     "weight": 0.043,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.853,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.054,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.053,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.05,
       "feature": "belly color: black"
      },
      {
       "weight": 0.048,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.046,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.046,
       "feature": "NOT belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.043,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.874,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.044,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.042,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.041,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.04,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.037,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.036,
       "feature": "under tail color: buff"
      }
     ]
    },
    {
     "weight": 0.043,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.864,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.18,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.079,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.028,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.027,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.026,
       "feature": "belly color: black"
      },
      {
       "weight": 0.025,
       "feature": "nape color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.071,
   "id": "N9",
   "operator": "SC+",
   "name": "high soft conjunction",
   "andness": 0.698,
   "verbalization": "nice to have most",
   "children": [
    {
     "weight": 0.6,
     "shared_ref": "N3"
    },
    {
     "weight": 0.1,
     "shared_ref": "N4"
    },
    {
     "weight": 0.043,
     "shared_ref": "N5"
    },
    {
     "weight": 0.043,
     "shared_ref": "N6"
    },
    {
     "weight": 0.043,
     "shared_ref": "N7"
    },
    {
     "weight": 0.043,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R146
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.69,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.76,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.204,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.13,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.417,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.179,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.177,
       "feature": "primary color: white"
      },
      {
       "weight": 0.152,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.073,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.065,
       "feature": "back color: black"
      },
      {
       "weight": 0.061,
       "feature": "wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.12,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.432,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.167,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.158,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.123,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.121,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.094,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.05,
       "feature": "belly color: white"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.532,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.141,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.112,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.103,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.102,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.079,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.054,
       "feature": "wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.166,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.595,
       "feature": "primary color: white"
      },
      {
       "weight": 0.187,
       "feature": "throat color: black"
      },
      {
       "weight": 0.112,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.107,
       "feature": "wing color: black"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.525,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.072,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.058,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.057,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.047,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.046,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.046,
       "feature": "NOT under tail color: brown"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.464,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.064,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.059,
       "feature": "NOT bill length: about the same as head"
      },
      {
       "weight": 0.059,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.058,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.058,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.057,
       "feature": "NOT breast color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.24,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.449,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.13,
     "shared_ref": "N3"
    },
    {
     "weight": 0.12,
     "shared_ref": "N4"
    },
    {
     "weight": 0.114,
     "shared_ref": "N5"
    },
    {
     "weight": 0.111,
     "shared_ref": "N6"
    },
    {
     "weight": 0.093,
     "shared_ref": "N7"
    },
    {
     "weight": 0.091,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R088
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.486,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.572,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.944,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.126,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.651,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.086,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.083,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.083,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.081,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.081,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.08,
       "feature": "NOT wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.119,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.596,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.203,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.156,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.133,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.131,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.129,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.125,
       "feature": "NOT wing color: buff"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.726,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.052,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.05,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.049,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.049,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.048,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.741,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.045,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.038,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.037,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.037,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.037,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.036,
       "feature": "NOT bill shape: dagger"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.712,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.051,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.051,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.047,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.047,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.046,
       "feature": "NOT crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.26,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.124,
       "feature": "wing color: white"
      },
      {
       "weight": 0.12,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.12,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.076,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.076,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.056,
       "feature": "head pattern: plain"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.428,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.249,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.126,
     "shared_ref": "N3"
    },
    {
     "weight": 0.119,
     "shared_ref": "N4"
    },
    {
     "weight": 0.117,
     "shared_ref": "N5"
    },
    {
     "weight": 0.114,
     "shared_ref": "N6"
    },
    {
     "weight": 0.108,
     "shared_ref": "N7"
    },
    {
     "weight": 0.089,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R009
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.902,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.721,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.192,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.115,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.612,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.067,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.059,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.057,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.056,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.055,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.055,
       "feature": "NOT primary color: brown"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.625,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.046,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.045,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.045,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.041,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.041,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.041,
       "feature": "NOT bill color: buff"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.146,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.401,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.327,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.271,
       "feature": "NOT upperparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.638,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.036,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.036,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.035,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.034,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.033,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.032,
       "feature": "NOT belly color: buff"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.348,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.145,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.132,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.13,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.124,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.122,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.121,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.446,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.157,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.09,
       "feature": "crown color: white"
      },
      {
       "weight": 0.074,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.071,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.07,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.067,
       "feature": "wing color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.279,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.263,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.115,
     "shared_ref": "N3"
    },
    {
     "weight": 0.105,
     "shared_ref": "N4"
    },
    {
     "weight": 0.1,
     "shared_ref": "N5"
    },
    {
     "weight": 0.097,
     "shared_ref": "N6"
    },
    {
     "weight": 0.092,
     "shared_ref": "N7"
    },
    {
     "weight": 0.09,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R041
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.602,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.666,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.032,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.18,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.568,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.211,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.109,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.107,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.098,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.087,
       "feature": "back color: white"
      },
      {
       "weight": 0.062,
       "feature": "crown color: white"
      }
     ]
    },
    {
     "weight": 0.138,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.401,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.285,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.279,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.174,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.163,
       "feature": "crown color: white"
      },
      {
       "weight": 0.099,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.426,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.552,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.256,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.102,
       "feature": "back color: grey"
      },
      {
       "weight": 0.08,
       "feature": "breast color: white"
      },
      {
       "weight": 0.01,
       "feature": "bill color: buff"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N6",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.016,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.164,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.159,
       "feature": "back color: white"
      },
      {
       "weight": 0.122,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.094,
       "feature": "wing color: white"
      },
      {
       "weight": 0.077,
       "feature": "primary color: white"
      },
      {
       "weight": 0.066,
       "feature": "primary color: grey"
      }
     ]
    },
    {
     "weight": 0.06,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.308,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.208,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.139,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.09,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.081,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.067,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.062,
       "feature": "NOT primary color: brown"
      }
     ]
    },
    {
     "weight": 0.059,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.287,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.176,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.094,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.086,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.072,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.07,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.062,
       "feature": "NOT belly color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.334,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.269,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.18,
     "shared_ref": "N3"
    },
    {
     "weight": 0.138,
     "shared_ref": "N4"
    },
    {
     "weight": 0.108,
     "shared_ref": "N5"
    },
    {
     "weight": 0.064,
     "shared_ref": "N6"
    },
    {
     "weight": 0.06,
     "shared_ref": "N7"
    },
    {
     "weight": 0.059,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R112
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.545,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.727,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.31,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.11,
     "id": "N3",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.179,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.765,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.131,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.101,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.002,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.001,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.001,
       "feature": "NOT belly color: black"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.279,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.424,
       "feature": "crown color: white"
      },
      {
       "weight": 0.182,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.106,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.077,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.061,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.039,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N5",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.202,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.855,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.03,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.018,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.017,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.016,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.015,
       "feature": "crown color: blue"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.211,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.734,
       "feature": "crown color: white"
      },
      {
       "weight": 0.127,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.087,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.031,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.021,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N7",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.783,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.476,
       "feature": "leg color: black"
      },
      {
       "weight": 0.277,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.247,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.214,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.289,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.221,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.191,
       "feature": "nape color: white"
      },
      {
       "weight": 0.096,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.059,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.052,
       "feature": "throat color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.273,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.309,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.11,
     "shared_ref": "N3"
    },
    {
     "weight": 0.105,
     "shared_ref": "N4"
    },
    {
     "weight": 0.094,
     "shared_ref": "N5"
    },
    {
     "weight": 0.093,
     "shared_ref": "N6"
    },
    {
     "weight": 0.087,
     "shared_ref": "N7"
    },
    {
     "weight": 0.076,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R145
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.116,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.655,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.754,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.129,
     "id": "N3",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.246,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.555,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.445,
       "feature": "NOT bill color: buff"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.628,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.085,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.057,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.055,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.054,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.053,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.053,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.638,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.046,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.046,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.045,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.045,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.045,
       "feature": "NOT back color: buff"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.587,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.098,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.09,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.086,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.082,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.082,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.082,
       "feature": "NOT breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.663,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.048,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.046,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.045,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.044,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.043,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.043,
       "feature": "NOT upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.543,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.105,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.098,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.097,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.083,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.071,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.066,
       "feature": "NOT under tail color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.345,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.251,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.129,
     "shared_ref": "N3"
    },
    {
     "weight": 0.102,
     "shared_ref": "N4"
    },
    {
     "weight": 0.1,
     "shared_ref": "N5"
    },
    {
     "weight": 0.098,
     "shared_ref": "N6"
    },
    {
     "weight": 0.095,
     "shared_ref": "N7"
    },
    {
     "weight": 0.095,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R151
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.914,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.789,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.248,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.108,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.372,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.18,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.153,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.138,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.135,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.134,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.131,
       "feature": "NOT crown color: blue"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.562,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.089,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.083,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.073,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.061,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.06,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.056,
       "feature": "NOT wing color: black"
      }
     ]
    },
    {
     "weight": 0.099,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.613,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.113,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.102,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.065,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.044,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.042,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.04,
       "feature": "NOT bill color: black"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.4,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.146,
       "feature": "leg color: black"
      },
      {
       "weight": 0.121,
       "feature": "crown color: white"
      },
      {
       "weight": 0.107,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.105,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.073,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.071,
       "feature": "upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.38,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.122,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.071,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.069,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.068,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.066,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.063,
       "feature": "NOT under tail color: brown"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.419,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.095,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.073,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.072,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.07,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.069,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.066,
       "feature": "NOT belly color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.211,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.435,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.108,
     "shared_ref": "N3"
    },
    {
     "weight": 0.105,
     "shared_ref": "N4"
    },
    {
     "weight": 0.099,
     "shared_ref": "N5"
    },
    {
     "weight": 0.096,
     "shared_ref": "N6"
    },
    {
     "weight": 0.094,
     "shared_ref": "N7"
    },
    {
     "weight": 0.094,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R087
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.401,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.695,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.142,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.519,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.222,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.324,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.31,
       "feature": "bill color: black"
      },
      {
       "weight": 0.123,
       "feature": "back color: white"
      },
      {
       "weight": 0.111,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.071,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.061,
       "feature": "nape color: white"
      }
     ]
    },
    {
     "weight": 0.127,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.468,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.191,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.183,
       "feature": "bill color: black"
      },
      {
       "weight": 0.095,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.073,
       "feature": "back color: white"
      },
      {
       "weight": 0.066,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.049,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.075,
     "id": "N5",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.19,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.774,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.146,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.08,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.074,
     "id": "N6",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.189,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.422,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.318,
       "feature": "crown color: white"
      },
      {
       "weight": 0.177,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.068,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.008,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.004,
       "feature": "throat color: grey"
      }
     ]
    },
    {
     "weight": 0.071,
     "id": "N7",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.189,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.68,
       "feature": "crown color: white"
      },
      {
       "weight": 0.173,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.146,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.001,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.063,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.353,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.191,
       "feature": "back color: grey"
      },
      {
       "weight": 0.159,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.14,
       "feature": "nape color: white"
      },
      {
       "weight": 0.105,
       "feature": "primary color: white"
      },
      {
       "weight": 0.08,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.078,
       "feature": "breast color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.305,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.921,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.519,
     "shared_ref": "N3"
    },
    {
     "weight": 0.127,
     "shared_ref": "N4"
    },
    {
     "weight": 0.075,
     "shared_ref": "N5"
    },
    {
     "weight": 0.074,
     "shared_ref": "N6"
    },
    {
     "weight": 0.071,
     "shared_ref": "N7"
    },
    {
     "weight": 0.063,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R181
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.586,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.698,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.074,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.195,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.37,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.37,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.202,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.099,
       "feature": "wing color: white"
      },
      {
       "weight": 0.075,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.061,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.058,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.146,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.355,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.312,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.215,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.118,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.058,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.05,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.047,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.125,
     "id": "N5",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.025,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.549,
       "feature": "crown color: white"
      },
      {
       "weight": 0.373,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.077,
       "feature": "breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.17,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.67,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.289,
       "feature": "crown color: white"
      },
      {
       "weight": 0.041,
       "feature": "breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.257,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.171,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.109,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.108,
       "feature": "wing color: white"
      },
      {
       "weight": 0.094,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.088,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.082,
       "feature": "primary color: grey"
      }
     ]
    },
    {
     "weight": 0.072,
     "id": "N8",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.027,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.391,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.26,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.245,
       "feature": "primary color: white"
      },
      {
       "weight": 0.031,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.026,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.024,
       "feature": "NOT underparts color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.302,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.966,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.195,
     "shared_ref": "N3"
    },
    {
     "weight": 0.146,
     "shared_ref": "N4"
    },
    {
     "weight": 0.125,
     "shared_ref": "N5"
    },
    {
     "weight": 0.116,
     "shared_ref": "N6"
    },
    {
     "weight": 0.085,
     "shared_ref": "N7"
    },
    {
     "weight": 0.072,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R051
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.507,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.873,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.162,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.31,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.635,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.197,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.105,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.058,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.058,
       "feature": "back color: white"
      },
      {
       "weight": 0.054,
       "feature": "crown color: white"
      },
      {
       "weight": 0.048,
       "feature": "wing color: white"
      }
     ]
    },
    {
     "weight": 0.295,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.639,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.222,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.118,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.065,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.065,
       "feature": "back color: white"
      },
      {
       "weight": 0.061,
       "feature": "crown color: white"
      },
      {
       "weight": 0.054,
       "feature": "wing color: white"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.334,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.994,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.003,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.002,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.0,
       "feature": "NOT crown color: brown"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.3,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.986,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.006,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.002,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.001,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.001,
       "feature": "back color: buff"
      },
      {
       "weight": 0.001,
       "feature": "NOT belly color: buff"
      }
     ]
    },
    {
     "weight": 0.057,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.367,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.473,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.426,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.101,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.032,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.59,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.939,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.061,
       "feature": "wing pattern: spotted"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.127,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.754,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.31,
     "shared_ref": "N3"
    },
    {
     "weight": 0.295,
     "shared_ref": "N4"
    },
    {
     "weight": 0.11,
     "shared_ref": "N5"
    },
    {
     "weight": 0.092,
     "shared_ref": "N6"
    },
    {
     "weight": 0.057,
     "shared_ref": "N7"
    },
    {
     "weight": 0.032,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R196
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.856,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.681,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.818,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.192,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.469,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.838,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.162,
       "feature": "bill color: black"
      }
     ]
    },
    {
     "weight": 0.191,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.523,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.467,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.421,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.091,
       "feature": "bill color: black"
      },
      {
       "weight": 0.021,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.129,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.226,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.664,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.332,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.004,
       "feature": "back color: yellow"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.315,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.223,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.188,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.171,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.168,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.106,
       "feature": "breast color: black"
      },
      {
       "weight": 0.094,
       "feature": "crown color: black"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N7",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.019,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.728,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.267,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.005,
       "feature": "wing pattern: spotted"
      }
     ]
    },
    {
     "weight": 0.048,
     "id": "N8",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.219,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.8,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.042,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.04,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.035,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.022,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.019,
       "feature": "NOT bill color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.319,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.959,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.192,
     "shared_ref": "N3"
    },
    {
     "weight": 0.191,
     "shared_ref": "N4"
    },
    {
     "weight": 0.129,
     "shared_ref": "N5"
    },
    {
     "weight": 0.101,
     "shared_ref": "N6"
    },
    {
     "weight": 0.098,
     "shared_ref": "N7"
    },
    {
     "weight": 0.048,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R098
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.843,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.903,
   "id": "N2",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.361,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.211,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.595,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.557,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.157,
       "feature": "leg color: black"
      },
      {
       "weight": 0.129,
       "feature": "bill color: black"
      },
      {
       "weight": 0.097,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.059,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.183,
     "id": "N4",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.246,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.581,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.113,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.061,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.053,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.048,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.046,
       "feature": "NOT primary color: grey"
      }
     ]
    },
    {
     "weight": 0.131,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.052,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.197,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.169,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.139,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.101,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.086,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.08,
       "feature": "crown color: blue"
      }
     ]
    },
    {
     "weight": 0.124,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.253,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.108,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.105,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.097,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.092,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.087,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.083,
       "feature": "NOT upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.531,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.043,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.043,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.042,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.04,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.038,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.037,
       "feature": "bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.418,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.079,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.065,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.063,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.062,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.056,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.056,
       "feature": "head pattern: eyebrow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.097,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.044,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.211,
     "shared_ref": "N3"
    },
    {
     "weight": 0.183,
     "shared_ref": "N4"
    },
    {
     "weight": 0.131,
     "shared_ref": "N5"
    },
    {
     "weight": 0.124,
     "shared_ref": "N6"
    },
    {
     "weight": 0.111,
     "shared_ref": "N7"
    },
    {
     "weight": 0.097,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R014
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.648,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.723,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.222,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.221,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.225,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.335,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.302,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.199,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.08,
       "feature": "leg color: black"
      },
      {
       "weight": 0.043,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.042,
       "feature": "bill color: black"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N4",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.178,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.856,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.105,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.004,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.004,
       "feature": "NOT leg color: grey"
      },
      {
       "weight": 0.004,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.004,
       "feature": "NOT breast color: buff"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N5",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.06,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.974,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.004,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.003,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.003,
       "feature": "crown color: white"
      },
      {
       "weight": 0.003,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.002,
       "feature": "NOT wing color: grey"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N6",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.087,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.331,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.284,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.208,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.177,
       "feature": "NOT underparts color: black"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.173,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.269,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.243,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.16,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.136,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.064,
       "feature": "leg color: black"
      },
      {
       "weight": 0.05,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.066,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.255,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.155,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.122,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.12,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.107,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.098,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.084,
       "feature": "throat color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.277,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.206,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.221,
     "shared_ref": "N3"
    },
    {
     "weight": 0.107,
     "shared_ref": "N4"
    },
    {
     "weight": 0.102,
     "shared_ref": "N5"
    },
    {
     "weight": 0.093,
     "shared_ref": "N6"
    },
    {
     "weight": 0.09,
     "shared_ref": "N7"
    },
    {
     "weight": 0.066,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R102
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.464,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.817,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.264,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.281,
     "id": "N3",
     "operator": "A",
     "name": "arithmetic mean",
     "andness": 0.5,
     "verbalization": "nice to have",
     "children": [
      {
       "weight": 1.0,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.0,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.0,
       "feature": "back color: white"
      }
     ]
    },
    {
     "weight": 0.266,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.397,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.542,
       "feature": "bill color: black"
      },
      {
       "weight": 0.458,
       "feature": "wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.206,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.174,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.858,
       "feature": "leg color: black"
      },
      {
       "weight": 0.077,
       "feature": "bill color: black"
      },
      {
       "weight": 0.065,
       "feature": "wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.137,
     "id": "N6",
     "operator": "A",
     "name": "arithmetic mean",
     "andness": 0.5,
     "verbalization": "nice to have",
     "children": [
      {
       "weight": 0.999,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.0,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.0,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.0,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.0,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.0,
       "feature": "nape color: brown"
      }
     ]
    },
    {
     "weight": 0.045,
     "id": "N7",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.828,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.378,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.157,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.146,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.093,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.043,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.029,
       "feature": "NOT underparts color: white"
      }
     ]
    },
    {
     "weight": 0.026,
     "id": "N8",
     "operator": "SC",
     "name": "medium soft conjunction",
     "andness": 0.618,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.971,
       "feature": "leg color: black"
      },
      {
       "weight": 0.029,
       "feature": "wing color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.183,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.155,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.281,
     "shared_ref": "N3"
    },
    {
     "weight": 0.266,
     "shared_ref": "N4"
    },
    {
     "weight": 0.206,
     "shared_ref": "N5"
    },
    {
     "weight": 0.137,
     "shared_ref": "N6"
    },
    {
     "weight": 0.045,
     "shared_ref": "N7"
    },
    {
     "weight": 0.026,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R012
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.319,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.676,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.012,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.483,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.28,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.144,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.109,
       "feature": "nape color: white"
      },
      {
       "weight": 0.095,
       "feature": "bill color: black"
      },
      {
       "weight": 0.063,
       "feature": "breast color: white"
      },
      {
       "weight": 0.062,
       "feature": "primary color: black"
      },
      {
       "weight": 0.061,
       "feature": "upper tail color: black"
      }
     ]
    },
    {
     "weight": 0.37,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.277,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.151,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.115,
       "feature": "nape color: white"
      },
      {
       "weight": 0.1,
       "feature": "bill color: black"
      },
      {
       "weight": 0.066,
       "feature": "breast color: white"
      },
      {
       "weight": 0.065,
       "feature": "primary color: black"
      },
      {
       "weight": 0.064,
       "feature": "upper tail color: black"
      }
     ]
    },
    {
     "weight": 0.027,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.052,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.282,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.266,
       "feature": "crown color: black"
      },
      {
       "weight": 0.172,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.163,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.117,
       "feature": "head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.026,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.311,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.317,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.193,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.184,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.167,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.139,
       "feature": "bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.017,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.118,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.351,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.33,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.179,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.141,
       "feature": "crown color: white"
      }
     ]
    },
    {
     "weight": 0.012,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.424,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.277,
       "feature": "belly color: black"
      },
      {
       "weight": 0.24,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.13,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.095,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.078,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.078,
       "feature": "crown color: blue"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.324,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.218,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.483,
     "shared_ref": "N3"
    },
    {
     "weight": 0.37,
     "shared_ref": "N4"
    },
    {
     "weight": 0.027,
     "shared_ref": "N5"
    },
    {
     "weight": 0.026,
     "shared_ref": "N6"
    },
    {
     "weight": 0.017,
     "shared_ref": "N7"
    },
    {
     "weight": 0.012,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R061
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.888,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.813,
   "id": "N2",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.399,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.155,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.53,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.195,
       "feature": "leg color: black"
      },
      {
       "weight": 0.166,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.161,
       "feature": "nape color: white"
      },
      {
       "weight": 0.093,
       "feature": "wing color: black"
      },
      {
       "weight": 0.093,
       "feature": "belly color: white"
      },
      {
       "weight": 0.089,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N4",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.281,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.482,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.182,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.177,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.159,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.368,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.218,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.207,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.204,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.197,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.173,
       "feature": "NOT crown color: blue"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.424,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.285,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.176,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.1,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.092,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.092,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.089,
       "feature": "NOT throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.391,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.241,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.212,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.188,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.182,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.178,
       "feature": "NOT wing pattern: spotted"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.574,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.052,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.052,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.047,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.047,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.046,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.045,
       "feature": "NOT upperparts color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.187,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.22,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.155,
     "shared_ref": "N3"
    },
    {
     "weight": 0.116,
     "shared_ref": "N4"
    },
    {
     "weight": 0.116,
     "shared_ref": "N5"
    },
    {
     "weight": 0.108,
     "shared_ref": "N6"
    },
    {
     "weight": 0.107,
     "shared_ref": "N7"
    },
    {
     "weight": 0.107,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R185
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.228,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.622,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.011,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.163,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.271,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.499,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.359,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.067,
       "feature": "throat color: white"
      },
      {
       "weight": 0.05,
       "feature": "back color: grey"
      },
      {
       "weight": 0.025,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.145,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.312,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.274,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.203,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.197,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.139,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.077,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.037,
       "feature": "throat color: white"
      }
     ]
    },
    {
     "weight": 0.14,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.333,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.402,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.223,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.199,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.176,
       "feature": "breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.134,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.205,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.51,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.252,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.223,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.015,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.114,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.217,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.1,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.093,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.076,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.068,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.068,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.274,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.585,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.221,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.078,
       "feature": "back color: grey"
      },
      {
       "weight": 0.07,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.045,
       "feature": "bill length: shorter than head"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.378,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.263,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.163,
     "shared_ref": "N3"
    },
    {
     "weight": 0.145,
     "shared_ref": "N4"
    },
    {
     "weight": 0.14,
     "shared_ref": "N5"
    },
    {
     "weight": 0.134,
     "shared_ref": "N6"
    },
    {
     "weight": 0.09,
     "shared_ref": "N7"
    },
    {
     "weight": 0.088,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R130
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.821,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.754,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.26,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.141,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.497,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.286,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.282,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.132,
       "feature": "leg color: black"
      },
      {
       "weight": 0.096,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.064,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.05,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.141,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.292,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.271,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.267,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.132,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.125,
       "feature": "leg color: black"
      },
      {
       "weight": 0.091,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.061,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.129,
     "id": "N5",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.145,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.242,
       "feature": "bill color: black"
      },
      {
       "weight": 0.213,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.177,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.122,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.098,
       "feature": "eye color: black"
      },
      {
       "weight": 0.074,
       "feature": "bill color: grey"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.188,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.693,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.155,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.152,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.522,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.099,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.064,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.051,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.041,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.036,
       "feature": "NOT primary color: white"
      },
      {
       "weight": 0.032,
       "feature": "wing color: buff"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.488,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.102,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.056,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.046,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.042,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.041,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.04,
       "feature": "under tail color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.246,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.294,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.141,
     "shared_ref": "N3"
    },
    {
     "weight": 0.141,
     "shared_ref": "N4"
    },
    {
     "weight": 0.129,
     "shared_ref": "N5"
    },
    {
     "weight": 0.116,
     "shared_ref": "N6"
    },
    {
     "weight": 0.096,
     "shared_ref": "N7"
    },
    {
     "weight": 0.093,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R134
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.901,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.821,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.089,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.118,
     "id": "N3",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.29,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.297,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.268,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.22,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.214,
       "feature": "NOT back color: buff"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.546,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.097,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.09,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.085,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.081,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.078,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.075,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.507,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.137,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.124,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.099,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.096,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.082,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.079,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.21,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.274,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.207,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.195,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.162,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.161,
       "feature": "NOT back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.73,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.036,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.034,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.033,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.032,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.031,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.031,
       "feature": "NOT back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.112,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.803,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.148,
       "feature": "eye color: black"
      },
      {
       "weight": 0.025,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.012,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.011,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.179,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.267,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.118,
     "shared_ref": "N3"
    },
    {
     "weight": 0.117,
     "shared_ref": "N4"
    },
    {
     "weight": 0.114,
     "shared_ref": "N5"
    },
    {
     "weight": 0.106,
     "shared_ref": "N6"
    },
    {
     "weight": 0.103,
     "shared_ref": "N7"
    },
    {
     "weight": 0.097,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R092
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.973,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.63,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.243,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.185,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.121,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.308,
       "feature": "crown color: black"
      },
      {
       "weight": 0.211,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.094,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.087,
       "feature": "eye color: black"
      },
      {
       "weight": 0.084,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.071,
       "feature": "NOT primary color: black"
      }
     ]
    },
    {
     "weight": 0.13,
     "id": "N4",
     "operator": "SC",
     "name": "medium soft conjunction",
     "andness": 0.628,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.579,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.136,
       "feature": "nape color: black"
      },
      {
       "weight": 0.107,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.097,
       "feature": "belly color: white"
      },
      {
       "weight": 0.072,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.008,
       "feature": "NOT wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N5",
     "operator": "SC",
     "name": "medium soft conjunction",
     "andness": 0.617,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.48,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.24,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.197,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.082,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N6",
     "operator": "SD-",
     "name": "low soft disjunction",
     "andness": 0.439,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.637,
       "feature": "nape color: black"
      },
      {
       "weight": 0.25,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.06,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.053,
       "feature": "primary color: grey"
      }
     ]
    },
    {
     "weight": 0.069,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.165,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.206,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.176,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.156,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.076,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.067,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.052,
       "feature": "NOT bill color: grey"
      }
     ]
    },
    {
     "weight": 0.063,
     "id": "N8",
     "operator": "SD-",
     "name": "low soft disjunction",
     "andness": 0.412,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.895,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.068,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.02,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.017,
       "feature": "primary color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.37,
   "id": "N9",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.243,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.185,
     "shared_ref": "N3"
    },
    {
     "weight": 0.13,
     "shared_ref": "N4"
    },
    {
     "weight": 0.108,
     "shared_ref": "N5"
    },
    {
     "weight": 0.076,
     "shared_ref": "N6"
    },
    {
     "weight": 0.069,
     "shared_ref": "N7"
    },
    {
     "weight": 0.063,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R127
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.599,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.63,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.003,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.824,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.064,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.104,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.104,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.092,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.089,
       "feature": "bill color: black"
      },
      {
       "weight": 0.082,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.073,
       "feature": "breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.068,
     "id": "N4",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.985,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.122,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.122,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.108,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.104,
       "feature": "bill color: black"
      },
      {
       "weight": 0.086,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.084,
       "feature": "belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.017,
     "id": "N5",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.807,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.39,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.191,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.151,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.119,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.098,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.051,
       "feature": "back color: yellow"
      }
     ]
    },
    {
     "weight": 0.014,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.433,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.685,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.031,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.029,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.029,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.029,
       "feature": "NOT primary color: white"
      },
      {
       "weight": 0.026,
       "feature": "NOT breast color: white"
      }
     ]
    },
    {
     "weight": 0.013,
     "id": "N7",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.21,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.109,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.106,
       "feature": "NOT leg color: grey"
      },
      {
       "weight": 0.094,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.089,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.089,
       "feature": "crown color: white"
      },
      {
       "weight": 0.082,
       "feature": "forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.01,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.318,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.372,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.217,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.187,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.034,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.026,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.021,
       "feature": "NOT upperparts color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.37,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.246,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.824,
     "shared_ref": "N3"
    },
    {
     "weight": 0.068,
     "shared_ref": "N4"
    },
    {
     "weight": 0.017,
     "shared_ref": "N5"
    },
    {
     "weight": 0.014,
     "shared_ref": "N6"
    },
    {
     "weight": 0.013,
     "shared_ref": "N7"
    },
    {
     "weight": 0.01,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R192
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.664,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.588,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.291,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.136,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.411,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.139,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.104,
       "feature": "primary color: white"
      },
      {
       "weight": 0.094,
       "feature": "leg color: black"
      },
      {
       "weight": 0.082,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.075,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.073,
       "feature": "wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.479,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.133,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.12,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.099,
       "feature": "primary color: white"
      },
      {
       "weight": 0.092,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.079,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.072,
       "feature": "nape color: grey"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.522,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.172,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.132,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.129,
       "feature": "leg color: black"
      },
      {
       "weight": 0.085,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.068,
       "feature": "belly color: white"
      },
      {
       "weight": 0.067,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.46,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.055,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.052,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.05,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.05,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.047,
       "feature": "NOT shape: duck-like"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.056,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.413,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.275,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.162,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.149,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.08,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.221,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.148,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.138,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.132,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.122,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.122,
       "feature": "NOT forehead color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.412,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.375,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.136,
     "shared_ref": "N3"
    },
    {
     "weight": 0.109,
     "shared_ref": "N4"
    },
    {
     "weight": 0.109,
     "shared_ref": "N5"
    },
    {
     "weight": 0.081,
     "shared_ref": "N6"
    },
    {
     "weight": 0.08,
     "shared_ref": "N7"
    },
    {
     "weight": 0.08,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R115
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.708,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.687,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.007,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.151,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.499,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.433,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.233,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.151,
       "feature": "nape color: white"
      },
      {
       "weight": 0.118,
       "feature": "back color: grey"
      },
      {
       "weight": 0.047,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.018,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.132,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.486,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.218,
       "feature": "back color: grey"
      },
      {
       "weight": 0.192,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.141,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.132,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.118,
       "feature": "primary color: white"
      },
      {
       "weight": 0.075,
       "feature": "wing color: black"
      }
     ]
    },
    {
     "weight": 0.126,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.411,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.389,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.248,
       "feature": "crown color: black"
      },
      {
       "weight": 0.237,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.126,
       "feature": "belly color: white"
      }
     ]
    },
    {
     "weight": 0.126,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.561,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.316,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.17,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.11,
       "feature": "nape color: white"
      },
      {
       "weight": 0.056,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.051,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.048,
       "feature": "crown color: black"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.074,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.294,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.172,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.145,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.112,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.1,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.098,
       "feature": "wing color: black"
      }
     ]
    },
    {
     "weight": 0.062,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.245,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.08,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.074,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.071,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.06,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.058,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.055,
       "feature": "shape: duck-like"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.313,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.18,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.151,
     "shared_ref": "N3"
    },
    {
     "weight": 0.132,
     "shared_ref": "N4"
    },
    {
     "weight": 0.126,
     "shared_ref": "N5"
    },
    {
     "weight": 0.126,
     "shared_ref": "N6"
    },
    {
     "weight": 0.087,
     "shared_ref": "N7"
    },
    {
     "weight": 0.062,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R177
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.585,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.716,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.016,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.158,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.333,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.593,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.088,
       "feature": "throat color: white"
      },
      {
       "weight": 0.087,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.066,
       "feature": "primary color: black"
      },
      {
       "weight": 0.061,
       "feature": "belly color: white"
      },
      {
       "weight": 0.044,
       "feature": "bill length: about the same as head"
      }
     ]
    },
    {
     "weight": 0.144,
     "id": "N4",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.152,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.998,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.002,
       "feature": "head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.126,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.107,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.992,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.003,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.002,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.002,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.001,
       "feature": "throat color: buff"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.285,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.55,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.189,
       "feature": "leg color: black"
      },
      {
       "weight": 0.081,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.061,
       "feature": "primary color: black"
      },
      {
       "weight": 0.047,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.039,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.384,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.34,
       "feature": "leg color: black"
      },
      {
       "weight": 0.264,
       "feature": "nape color: white"
      },
      {
       "weight": 0.142,
       "feature": "crown color: black"
      },
      {
       "weight": 0.109,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.084,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.061,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.171,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.196,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.169,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.122,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.114,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.113,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.106,
       "feature": "bill color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.284,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.953,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.158,
     "shared_ref": "N3"
    },
    {
     "weight": 0.144,
     "shared_ref": "N4"
    },
    {
     "weight": 0.126,
     "shared_ref": "N5"
    },
    {
     "weight": 0.106,
     "shared_ref": "N6"
    },
    {
     "weight": 0.096,
     "shared_ref": "N7"
    },
    {
     "weight": 0.081,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R168
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 1.027,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.515,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.145,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.169,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.278,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.148,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.142,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.099,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.083,
       "feature": "breast color: black"
      },
      {
       "weight": 0.062,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.057,
       "feature": "upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.548,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.278,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.273,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.26,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.189,
       "feature": "NOT shape: duck-like"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.526,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.081,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.06,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.058,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.054,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.054,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.053,
       "feature": "NOT underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.49,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.163,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.131,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.122,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.121,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.11,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.1,
       "feature": "NOT upperparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N7",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.004,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.195,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.187,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.131,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.11,
       "feature": "breast color: black"
      },
      {
       "weight": 0.083,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.052,
       "feature": "nape color: white"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N8",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.029,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.257,
       "feature": "leg color: black"
      },
      {
       "weight": 0.176,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.164,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.155,
       "feature": "throat color: white"
      },
      {
       "weight": 0.136,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.112,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.485,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.362,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.169,
     "shared_ref": "N3"
    },
    {
     "weight": 0.107,
     "shared_ref": "N4"
    },
    {
     "weight": 0.103,
     "shared_ref": "N5"
    },
    {
     "weight": 0.102,
     "shared_ref": "N6"
    },
    {
     "weight": 0.095,
     "shared_ref": "N7"
    },
    {
     "weight": 0.092,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R139
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.884,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.927,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.948,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.599,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.943,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.264,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.182,
       "feature": "back color: grey"
      },
      {
       "weight": 0.157,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.111,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.094,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.046,
       "feature": "bill color: black"
      }
     ]
    },
    {
     "weight": 0.058,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.128,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.174,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.168,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.139,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.125,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.115,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.108,
       "feature": "NOT primary color: white"
      }
     ]
    },
    {
     "weight": 0.047,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.645,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.353,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.042,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.023,
       "feature": "belly color: black"
      },
      {
       "weight": 0.022,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.021,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.02,
       "feature": "NOT breast color: brown"
      }
     ]
    },
    {
     "weight": 0.045,
     "id": "N6",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.172,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.932,
       "feature": "eye color: black"
      },
      {
       "weight": 0.025,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.024,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.019,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.041,
     "id": "N7",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.011,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.364,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.322,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.314,
       "feature": "NOT forehead color: yellow"
      }
     ]
    },
    {
     "weight": 0.039,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.297,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.925,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.008,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.006,
       "feature": "NOT leg color: grey"
      },
      {
       "weight": 0.006,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.005,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.005,
       "feature": "NOT back pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.073,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.871,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.599,
     "shared_ref": "N3"
    },
    {
     "weight": 0.058,
     "shared_ref": "N4"
    },
    {
     "weight": 0.047,
     "shared_ref": "N5"
    },
    {
     "weight": 0.045,
     "shared_ref": "N6"
    },
    {
     "weight": 0.041,
     "shared_ref": "N7"
    },
    {
     "weight": 0.039,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R085
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.883,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.596,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.883,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.113,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.722,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.056,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.055,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.05,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.05,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.048,
       "feature": "NOT forehead color: yellow"
      }
     ]
    },
    {
     "weight": 0.112,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.675,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.088,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.064,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.061,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.059,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.057,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.057,
       "feature": "NOT breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.65,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.063,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.063,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.058,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.058,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.057,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.056,
       "feature": "NOT bill color: buff"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.719,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.056,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.054,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.049,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.047,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.046,
       "feature": "NOT nape color: buff"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N7",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.062,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.315,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.194,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.11,
       "feature": "belly color: white"
      },
      {
       "weight": 0.102,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.097,
       "feature": "breast color: white"
      },
      {
       "weight": 0.093,
       "feature": "throat color: white"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.199,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.278,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.206,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.134,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.118,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.072,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.067,
       "feature": "wing pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.404,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.298,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.113,
     "shared_ref": "N3"
    },
    {
     "weight": 0.112,
     "shared_ref": "N4"
    },
    {
     "weight": 0.11,
     "shared_ref": "N5"
    },
    {
     "weight": 0.11,
     "shared_ref": "N6"
    },
    {
     "weight": 0.105,
     "shared_ref": "N7"
    },
    {
     "weight": 0.103,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R150
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.612,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.647,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.102,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.142,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.51,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.184,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.093,
       "feature": "crown color: white"
      },
      {
       "weight": 0.078,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.07,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.067,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.061,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.122,
     "id": "N4",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.207,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.93,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.027,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.004,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.004,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.004,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.003,
       "feature": "NOT crown color: brown"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.408,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.162,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.15,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.137,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.125,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.111,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.107,
       "feature": "NOT back color: yellow"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.4,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.153,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.121,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.116,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.115,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.11,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.102,
       "feature": "tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.493,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.936,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.007,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.005,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.005,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.005,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.005,
       "feature": "belly color: grey"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.381,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.243,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.17,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.156,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.155,
       "feature": "back color: buff"
      },
      {
       "weight": 0.142,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.135,
       "feature": "NOT belly color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.353,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.17,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.142,
     "shared_ref": "N3"
    },
    {
     "weight": 0.122,
     "shared_ref": "N4"
    },
    {
     "weight": 0.107,
     "shared_ref": "N5"
    },
    {
     "weight": 0.106,
     "shared_ref": "N6"
    },
    {
     "weight": 0.105,
     "shared_ref": "N7"
    },
    {
     "weight": 0.101,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R174
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.97,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.821,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.157,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.263,
     "id": "N3",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.859,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.274,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.231,
       "feature": "belly color: white"
      },
      {
       "weight": 0.092,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.074,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.062,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.052,
       "feature": "NOT tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.167,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.058,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.269,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.206,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.111,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.067,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.05,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.048,
       "feature": "NOT breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N5",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.748,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.34,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.199,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.111,
       "feature": "back color: buff"
      },
      {
       "weight": 0.091,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.072,
       "feature": "bill color: black"
      },
      {
       "weight": 0.058,
       "feature": "wing color: buff"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.06,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.162,
       "feature": "bill color: black"
      },
      {
       "weight": 0.143,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.102,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.077,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.05,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.049,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.067,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.238,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.209,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.197,
       "feature": "back color: brown"
      },
      {
       "weight": 0.099,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.071,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.064,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.042,
       "feature": "NOT back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.053,
     "id": "N8",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.698,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.272,
       "feature": "leg color: black"
      },
      {
       "weight": 0.24,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.115,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.113,
       "feature": "back color: buff"
      },
      {
       "weight": 0.092,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.072,
       "feature": "upper tail color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.179,
   "id": "N9",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.029,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.263,
     "shared_ref": "N3"
    },
    {
     "weight": 0.167,
     "shared_ref": "N4"
    },
    {
     "weight": 0.098,
     "shared_ref": "N5"
    },
    {
     "weight": 0.084,
     "shared_ref": "N6"
    },
    {
     "weight": 0.067,
     "shared_ref": "N7"
    },
    {
     "weight": 0.053,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R152
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 0.985,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.543,
   "id": "N2",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.421,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.15,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.242,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.355,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.307,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.1,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.063,
       "feature": "back color: black"
      },
      {
       "weight": 0.039,
       "feature": "primary color: black"
      },
      {
       "weight": 0.036,
       "feature": "wing color: black"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.281,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.225,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.2,
       "feature": "back color: white"
      },
      {
       "weight": 0.195,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.096,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.074,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.051,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.155,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.287,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.246,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.154,
       "feature": "back color: black"
      },
      {
       "weight": 0.096,
       "feature": "primary color: black"
      },
      {
       "weight": 0.082,
       "feature": "throat color: white"
      },
      {
       "weight": 0.073,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.13,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.646,
       "feature": "back color: white"
      },
      {
       "weight": 0.164,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.115,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.075,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.512,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.2,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.135,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.121,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.118,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.093,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.056,
       "feature": "NOT bill color: buff"
      }
     ]
    },
    {
     "weight": 0.072,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.612,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.126,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.08,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.079,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.076,
       "feature": "NOT upper tail color: black"
      },
      {
       "weight": 0.065,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.059,
       "feature": "NOT bill shape: all-purpose"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.457,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.074,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.15,
     "shared_ref": "N3"
    },
    {
     "weight": 0.117,
     "shared_ref": "N4"
    },
    {
     "weight": 0.086,
     "shared_ref": "N5"
    },
    {
     "weight": 0.076,
     "shared_ref": "N6"
    },
    {
     "weight": 0.073,
     "shared_ref": "N7"
    },
    {
     "weight": 0.072,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R109
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.139,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.629,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.12,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.352,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.569,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.319,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.067,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.036,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.009,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.336,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.758,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.45,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.252,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.058,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.051,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.05,
       "feature": "wing color: white"
      },
      {
       "weight": 0.041,
       "feature": "under tail color: white"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N5",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.787,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.224,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.204,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.194,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.192,
       "feature": "wing color: white"
      },
      {
       "weight": 0.157,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.028,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.044,
     "id": "N6",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.011,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.383,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.237,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.221,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.159,
       "feature": "NOT back color: white"
      }
     ]
    },
    {
     "weight": 0.04,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.139,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.523,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.477,
       "feature": "NOT primary color: buff"
      }
     ]
    },
    {
     "weight": 0.038,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.102,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.523,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.477,
       "feature": "NOT primary color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.371,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.2,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.352,
     "shared_ref": "N3"
    },
    {
     "weight": 0.336,
     "shared_ref": "N4"
    },
    {
     "weight": 0.1,
     "shared_ref": "N5"
    },
    {
     "weight": 0.044,
     "shared_ref": "N6"
    },
    {
     "weight": 0.04,
     "shared_ref": "N7"
    },
    {
     "weight": 0.038,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R101
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.551,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.728,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.881,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.121,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.219,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.297,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.215,
       "feature": "breast color: black"
      },
      {
       "weight": 0.159,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.073,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.063,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.055,
       "feature": "back color: brown"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.232,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.515,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.485,
       "feature": "breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.381,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.196,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.17,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.159,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.141,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.096,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.086,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.312,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.282,
       "feature": "breast color: black"
      },
      {
       "weight": 0.211,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.159,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.152,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.103,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.093,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.441,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.235,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.126,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.111,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.108,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.104,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.102,
       "feature": "breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.448,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.17,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.153,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.094,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.075,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.067,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.062,
       "feature": "NOT wing pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.272,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.001,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.121,
     "shared_ref": "N3"
    },
    {
     "weight": 0.107,
     "shared_ref": "N4"
    },
    {
     "weight": 0.098,
     "shared_ref": "N5"
    },
    {
     "weight": 0.088,
     "shared_ref": "N6"
    },
    {
     "weight": 0.081,
     "shared_ref": "N7"
    },
    {
     "weight": 0.08,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R024
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.954,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.575,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.072,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.424,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.652,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.258,
       "feature": "wing color: white"
      },
      {
       "weight": 0.2,
       "feature": "throat color: black"
      },
      {
       "weight": 0.165,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.09,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.072,
       "feature": "belly color: white"
      },
      {
       "weight": 0.055,
       "feature": "nape color: black"
      }
     ]
    },
    {
     "weight": 0.2,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.422,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.283,
       "feature": "wing color: white"
      },
      {
       "weight": 0.219,
       "feature": "throat color: black"
      },
      {
       "weight": 0.194,
       "feature": "breast color: white"
      },
      {
       "weight": 0.182,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.079,
       "feature": "belly color: white"
      },
      {
       "weight": 0.043,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.509,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.205,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.125,
       "feature": "nape color: black"
      },
      {
       "weight": 0.124,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.108,
       "feature": "bill color: black"
      },
      {
       "weight": 0.103,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.089,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.048,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.216,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.758,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.086,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.047,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.044,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.041,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.025,
       "feature": "NOT crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.048,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.526,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.248,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.15,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.147,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.119,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.119,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.109,
       "feature": "NOT crown color: grey"
      }
     ]
    },
    {
     "weight": 0.048,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.376,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.162,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.161,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.135,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.116,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.102,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.089,
       "feature": "NOT under tail color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.425,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.981,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.424,
     "shared_ref": "N3"
    },
    {
     "weight": 0.2,
     "shared_ref": "N4"
    },
    {
     "weight": 0.076,
     "shared_ref": "N5"
    },
    {
     "weight": 0.048,
     "shared_ref": "N6"
    },
    {
     "weight": 0.048,
     "shared_ref": "N7"
    },
    {
     "weight": 0.048,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R103
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.937,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.693,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.172,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.107,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.681,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.111,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.076,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.074,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.073,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.062,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.047,
       "feature": "NOT belly color: buff"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.663,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.093,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.087,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.073,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.053,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.051,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.051,
       "feature": "NOT upperparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.649,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.107,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.1,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.068,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.056,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.056,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.055,
       "feature": "NOT back color: buff"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.678,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.13,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.092,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.089,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.061,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.055,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.046,
       "feature": "NOT underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.515,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.224,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.178,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.173,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.147,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.14,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.139,
       "feature": "NOT primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.507,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.19,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.115,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.115,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.104,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.103,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.096,
       "feature": "NOT under tail color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.307,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.475,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.107,
     "shared_ref": "N3"
    },
    {
     "weight": 0.103,
     "shared_ref": "N4"
    },
    {
     "weight": 0.102,
     "shared_ref": "N5"
    },
    {
     "weight": 0.102,
     "shared_ref": "N6"
    },
    {
     "weight": 0.1,
     "shared_ref": "N7"
    },
    {
     "weight": 0.098,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R170
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.984,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.558,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.25,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.182,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.932,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.366,
       "feature": "wing color: black"
      },
      {
       "weight": 0.111,
       "feature": "eye color: black"
      },
      {
       "weight": 0.091,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.074,
       "feature": "bill color: black"
      },
      {
       "weight": 0.073,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.065,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.165,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.783,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.326,
       "feature": "wing color: black"
      },
      {
       "weight": 0.154,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.099,
       "feature": "eye color: black"
      },
      {
       "weight": 0.088,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.081,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.065,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N5",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.708,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.565,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.369,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.066,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N6",
     "operator": "SC-",
     "name": "low soft conjunction",
     "andness": 0.597,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.65,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.188,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.076,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.04,
       "feature": "back color: grey"
      },
      {
       "weight": 0.032,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.015,
       "feature": "primary color: grey"
      }
     ]
    },
    {
     "weight": 0.077,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.102,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.167,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.138,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.126,
       "feature": "bill color: black"
      },
      {
       "weight": 0.108,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.09,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.088,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.062,
     "id": "N8",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.018,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.441,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.36,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.199,
       "feature": "nape color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.442,
   "id": "N9",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.248,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.182,
     "shared_ref": "N3"
    },
    {
     "weight": 0.165,
     "shared_ref": "N4"
    },
    {
     "weight": 0.094,
     "shared_ref": "N5"
    },
    {
     "weight": 0.087,
     "shared_ref": "N6"
    },
    {
     "weight": 0.077,
     "shared_ref": "N7"
    },
    {
     "weight": 0.062,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R190
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.258,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.608,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.163,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.156,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.399,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.192,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.15,
       "feature": "back color: buff"
      },
      {
       "weight": 0.143,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.122,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.106,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.071,
       "feature": "underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.144,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.416,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.37,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.17,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.108,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.063,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.059,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.051,
       "feature": "upper tail color: buff"
      }
     ]
    },
    {
     "weight": 0.121,
     "id": "N5",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.254,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.622,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.213,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.144,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.021,
       "feature": "nape color: brown"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.143,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.312,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.228,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.19,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.1,
       "feature": "bill color: black"
      },
      {
       "weight": 0.094,
       "feature": "throat color: white"
      },
      {
       "weight": 0.076,
       "feature": "wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.27,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.841,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.121,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.037,
       "feature": "head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.119,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.809,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.109,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.083,
       "feature": "NOT upper tail color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.392,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.954,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.156,
     "shared_ref": "N3"
    },
    {
     "weight": 0.144,
     "shared_ref": "N4"
    },
    {
     "weight": 0.121,
     "shared_ref": "N5"
    },
    {
     "weight": 0.106,
     "shared_ref": "N6"
    },
    {
     "weight": 0.104,
     "shared_ref": "N7"
    },
    {
     "weight": 0.09,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R006
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.262,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.577,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.92,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.157,
     "id": "N3",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.013,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.647,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.33,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.007,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.006,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.005,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.002,
       "feature": "throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.502,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 1.0,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.113,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.648,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.124,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.074,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.073,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.071,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.068,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.054,
       "feature": "NOT wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.703,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.07,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.068,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.068,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.063,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.062,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.06,
       "feature": "NOT shape: duck-like"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.65,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.087,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.085,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.084,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.079,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.072,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.07,
       "feature": "NOT size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N8",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.163,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.411,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.127,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.089,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.08,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.068,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.055,
       "feature": "upperparts color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.423,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.368,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.157,
     "shared_ref": "N3"
    },
    {
     "weight": 0.117,
     "shared_ref": "N4"
    },
    {
     "weight": 0.113,
     "shared_ref": "N5"
    },
    {
     "weight": 0.107,
     "shared_ref": "N6"
    },
    {
     "weight": 0.104,
     "shared_ref": "N7"
    },
    {
     "weight": 0.094,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R010
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.856,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.792,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.167,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.776,
     "id": "N3",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.995,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.264,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.238,
       "feature": "nape color: white"
      },
      {
       "weight": 0.171,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.098,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.044,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.039,
       "feature": "primary color: white"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.31,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.261,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.236,
       "feature": "nape color: white"
      },
      {
       "weight": 0.17,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.097,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.043,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.039,
       "feature": "primary color: white"
      }
     ]
    },
    {
     "weight": 0.017,
     "id": "N5",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.064,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.372,
       "feature": "crown color: black"
      },
      {
       "weight": 0.133,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.096,
       "feature": "crown color: white"
      },
      {
       "weight": 0.044,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.044,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.038,
       "feature": "belly color: grey"
      }
     ]
    },
    {
     "weight": 0.017,
     "id": "N6",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.204,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.492,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.304,
       "feature": "crown color: black"
      },
      {
       "weight": 0.04,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.023,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.021,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.018,
       "feature": "forehead color: brown"
      }
     ]
    },
    {
     "weight": 0.014,
     "id": "N7",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.078,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.245,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.204,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.127,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.054,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.039,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.036,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.012,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.087,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.202,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.187,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.171,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.163,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.138,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.138,
       "feature": "belly color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.208,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.794,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.776,
     "shared_ref": "N3"
    },
    {
     "weight": 0.107,
     "shared_ref": "N4"
    },
    {
     "weight": 0.017,
     "shared_ref": "N5"
    },
    {
     "weight": 0.017,
     "shared_ref": "N6"
    },
    {
     "weight": 0.014,
     "shared_ref": "N7"
    },
    {
     "weight": 0.012,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R035
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.468,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.793,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.932,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.965,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.965,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.135,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.099,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.086,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.082,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.068,
       "feature": "wing color: white"
      },
      {
       "weight": 0.058,
       "feature": "wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.005,
     "id": "N4",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.01,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.069,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.069,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.058,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.058,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.056,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.055,
       "feature": "NOT size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.004,
     "id": "N5",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.009,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.064,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.063,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.062,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.061,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.06,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.06,
       "feature": "forehead color: white"
      }
     ]
    },
    {
     "weight": 0.004,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.095,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.087,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.046,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.041,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.041,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.038,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.036,
       "feature": "NOT back color: buff"
      }
     ]
    },
    {
     "weight": 0.004,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.373,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.229,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.125,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.113,
       "feature": "nape color: black"
      },
      {
       "weight": 0.113,
       "feature": "throat color: black"
      },
      {
       "weight": 0.111,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.085,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.004,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.25,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.5,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.22,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.151,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.129,
       "feature": "bill length: shorter than head"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.207,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.911,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.965,
     "shared_ref": "N3"
    },
    {
     "weight": 0.005,
     "shared_ref": "N4"
    },
    {
     "weight": 0.004,
     "shared_ref": "N5"
    },
    {
     "weight": 0.004,
     "shared_ref": "N6"
    },
    {
     "weight": 0.004,
     "shared_ref": "N7"
    },
    {
     "weight": 0.004,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R063
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.392,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.704,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.073,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.165,
     "id": "N3",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.039,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.984,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.002,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.002,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.002,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.002,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.002,
       "feature": "NOT bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.144,
     "id": "N4",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.152,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.992,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.003,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.002,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.002,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.001,
       "feature": "NOT breast color: grey"
      }
     ]
    },
    {
     "weight": 0.121,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.325,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.121,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.091,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.09,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.088,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.083,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.082,
       "feature": "NOT wing pattern: spotted"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.233,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.297,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.241,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.234,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.227,
       "feature": "NOT breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.328,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.185,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.175,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.141,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.134,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.098,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.095,
       "feature": "crown color: blue"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.601,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.039,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.034,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.03,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.03,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.029,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.029,
       "feature": "NOT crown color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.296,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.153,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.165,
     "shared_ref": "N3"
    },
    {
     "weight": 0.144,
     "shared_ref": "N4"
    },
    {
     "weight": 0.121,
     "shared_ref": "N5"
    },
    {
     "weight": 0.11,
     "shared_ref": "N6"
    },
    {
     "weight": 0.106,
     "shared_ref": "N7"
    },
    {
     "weight": 0.103,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R108
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 0.989,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.544,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.069,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.091,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.407,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 1.0,
       "feature": "NOT underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.669,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.085,
       "feature": "NOT upper tail color: black"
      },
      {
       "weight": 0.073,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.068,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.057,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.045,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.044,
       "feature": "NOT breast color: white"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.6,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.087,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.087,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.078,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.076,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.066,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.049,
       "feature": "NOT wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.598,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.138,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.103,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.08,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.079,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.068,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.064,
       "feature": "NOT under tail color: grey"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.28,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.202,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.144,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.141,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.117,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.079,
       "feature": "throat color: black"
      },
      {
       "weight": 0.066,
       "feature": "wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.684,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.049,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "NOT primary color: white"
      },
      {
       "weight": 0.046,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.036,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.036,
       "feature": "NOT nape color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.456,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.438,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.091,
     "shared_ref": "N3"
    },
    {
     "weight": 0.088,
     "shared_ref": "N4"
    },
    {
     "weight": 0.087,
     "shared_ref": "N5"
    },
    {
     "weight": 0.087,
     "shared_ref": "N6"
    },
    {
     "weight": 0.086,
     "shared_ref": "N7"
    },
    {
     "weight": 0.085,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R007
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.696,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.557,
   "id": "N2",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.364,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.14,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.339,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.27,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.172,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.151,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.128,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.097,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.084,
       "feature": "leg color: grey"
      }
     ]
    },
    {
     "weight": 0.134,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.285,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.231,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.188,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.136,
       "feature": "throat color: black"
      },
      {
       "weight": 0.131,
       "feature": "primary color: black"
      },
      {
       "weight": 0.095,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.091,
       "feature": "wing color: white"
      }
     ]
    },
    {
     "weight": 0.132,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.461,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.154,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.098,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.093,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.086,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.075,
       "feature": "nape color: black"
      },
      {
       "weight": 0.073,
       "feature": "breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.123,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.201,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.199,
       "feature": "nape color: black"
      },
      {
       "weight": 0.145,
       "feature": "throat color: black"
      },
      {
       "weight": 0.119,
       "feature": "back color: black"
      },
      {
       "weight": 0.108,
       "feature": "crown color: black"
      },
      {
       "weight": 0.072,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.422,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.138,
       "feature": "belly color: black"
      },
      {
       "weight": 0.115,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.111,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.11,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.09,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.078,
       "feature": "NOT breast color: grey"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.522,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.177,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.144,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.141,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.11,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.106,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.094,
       "feature": "NOT wing shape: pointed-wings"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.443,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.269,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.14,
     "shared_ref": "N3"
    },
    {
     "weight": 0.134,
     "shared_ref": "N4"
    },
    {
     "weight": 0.132,
     "shared_ref": "N5"
    },
    {
     "weight": 0.108,
     "shared_ref": "N6"
    },
    {
     "weight": 0.088,
     "shared_ref": "N7"
    },
    {
     "weight": 0.087,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R200
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.873,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.764,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.905,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.193,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.818,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 1.0,
       "feature": "wing shape: pointed-wings"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N4",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.281,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.216,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.213,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.211,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.211,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.148,
       "feature": "NOT back color: white"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.142,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.946,
       "feature": "belly color: white"
      },
      {
       "weight": 0.022,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.021,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.012,
       "feature": "NOT belly color: grey"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.341,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.433,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.131,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.113,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.082,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.081,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.072,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.558,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.063,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.054,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.054,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.053,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.047,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.046,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.351,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.226,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.172,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.148,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.108,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.095,
       "feature": "breast color: white"
      },
      {
       "weight": 0.089,
       "feature": "throat color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.236,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.933,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.193,
     "shared_ref": "N3"
    },
    {
     "weight": 0.115,
     "shared_ref": "N4"
    },
    {
     "weight": 0.111,
     "shared_ref": "N5"
    },
    {
     "weight": 0.108,
     "shared_ref": "N6"
    },
    {
     "weight": 0.107,
     "shared_ref": "N7"
    },
    {
     "weight": 0.098,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R046
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.867,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.924,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.813,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.743,
     "id": "N3",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.996,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.369,
       "feature": "nape color: white"
      },
      {
       "weight": 0.297,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.212,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.122,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.057,
     "id": "N4",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.176,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.656,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.344,
       "feature": "leg color: buff"
      }
     ]
    },
    {
     "weight": 0.056,
     "id": "N5",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.192,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.322,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.247,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.184,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.17,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.077,
       "feature": "wing shape: pointed-wings"
      }
     ]
    },
    {
     "weight": 0.038,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.371,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.812,
       "feature": "nape color: white"
      },
      {
       "weight": 0.133,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.055,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.027,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.493,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.82,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.18,
       "feature": "wing color: buff"
      }
     ]
    },
    {
     "weight": 0.025,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.53,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.712,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.11,
       "feature": "crown color: white"
      },
      {
       "weight": 0.104,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.074,
       "feature": "belly color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.076,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.989,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.743,
     "shared_ref": "N3"
    },
    {
     "weight": 0.057,
     "shared_ref": "N4"
    },
    {
     "weight": 0.056,
     "shared_ref": "N5"
    },
    {
     "weight": 0.038,
     "shared_ref": "N6"
    },
    {
     "weight": 0.027,
     "shared_ref": "N7"
    },
    {
     "weight": 0.025,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R047
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.821,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.61,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.174,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.128,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.557,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.256,
       "feature": "crown color: white"
      },
      {
       "weight": 0.24,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.187,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.122,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.078,
       "feature": "nape color: white"
      },
      {
       "weight": 0.046,
       "feature": "primary color: white"
      }
     ]
    },
    {
     "weight": 0.121,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.074,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.237,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.184,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.156,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.155,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.142,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.127,
       "feature": "wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.523,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.48,
       "feature": "crown color: white"
      },
      {
       "weight": 0.45,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.071,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.476,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.189,
       "feature": "wing color: white"
      },
      {
       "weight": 0.181,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.118,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.099,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.082,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.075,
       "feature": "nape color: white"
      }
     ]
    },
    {
     "weight": 0.079,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.329,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.102,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.082,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.074,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.073,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.064,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.064,
       "feature": "throat color: grey"
      }
     ]
    },
    {
     "weight": 0.078,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.363,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.094,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.093,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.08,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.064,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.061,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.057,
       "feature": "breast color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.39,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.215,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.128,
     "shared_ref": "N3"
    },
    {
     "weight": 0.121,
     "shared_ref": "N4"
    },
    {
     "weight": 0.094,
     "shared_ref": "N5"
    },
    {
     "weight": 0.094,
     "shared_ref": "N6"
    },
    {
     "weight": 0.079,
     "shared_ref": "N7"
    },
    {
     "weight": 0.078,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R040
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.119,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.726,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.049,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.564,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.033,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.399,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.189,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.121,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.115,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.077,
       "feature": "leg color: black"
      },
      {
       "weight": 0.058,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.238,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.057,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.376,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.179,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.114,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.093,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.073,
       "feature": "leg color: black"
      },
      {
       "weight": 0.055,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.087,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.21,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.173,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.166,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.112,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.105,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.082,
       "feature": "primary color: grey"
      }
     ]
    },
    {
     "weight": 0.018,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.22,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.241,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.172,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.165,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.104,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.077,
       "feature": "bill color: black"
      },
      {
       "weight": 0.074,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.017,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.314,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.333,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.158,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.141,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.134,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.068,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.068,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.009,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.691,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.051,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.048,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.032,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.03,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.029,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.028,
       "feature": "forehead color: blue"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.274,
   "id": "N9",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.133,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.564,
     "shared_ref": "N3"
    },
    {
     "weight": 0.238,
     "shared_ref": "N4"
    },
    {
     "weight": 0.097,
     "shared_ref": "N5"
    },
    {
     "weight": 0.018,
     "shared_ref": "N6"
    },
    {
     "weight": 0.017,
     "shared_ref": "N7"
    },
    {
     "weight": 0.009,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R155
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.091,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.815,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.872,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.525,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.373,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.169,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.166,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.145,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.117,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.114,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.098,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.324,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.404,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.225,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.222,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.193,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.152,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.067,
       "feature": "bill color: black"
      },
      {
       "weight": 0.06,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.065,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.596,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.441,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.37,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.189,
       "feature": "bill color: black"
      }
     ]
    },
    {
     "weight": 0.03,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.612,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.455,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.235,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.232,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.079,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.009,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.621,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.098,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.097,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.096,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.092,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.075,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.049,
       "feature": "NOT belly color: white"
      }
     ]
    },
    {
     "weight": 0.008,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.579,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.118,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.093,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.08,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.079,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.067,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.057,
       "feature": "NOT belly color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.185,
   "id": "N9",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.187,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.525,
     "shared_ref": "N3"
    },
    {
     "weight": 0.324,
     "shared_ref": "N4"
    },
    {
     "weight": 0.065,
     "shared_ref": "N5"
    },
    {
     "weight": 0.03,
     "shared_ref": "N6"
    },
    {
     "weight": 0.009,
     "shared_ref": "N7"
    },
    {
     "weight": 0.008,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R193
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.897,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.734,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.003,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.133,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.406,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.207,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.177,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.16,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.137,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.064,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.058,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.273,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.234,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.201,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.182,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.155,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.071,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.066,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.491,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.214,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.175,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.169,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.163,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.159,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.12,
       "feature": "NOT wing shape: pointed-wings"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.449,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.658,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.192,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.15,
       "feature": "NOT back color: yellow"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.529,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.194,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.188,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.173,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.158,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.152,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.135,
       "feature": "NOT size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.362,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.332,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.125,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.124,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.111,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.107,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.102,
       "feature": "NOT forehead color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.266,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.41,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.133,
     "shared_ref": "N3"
    },
    {
     "weight": 0.105,
     "shared_ref": "N4"
    },
    {
     "weight": 0.091,
     "shared_ref": "N5"
    },
    {
     "weight": 0.089,
     "shared_ref": "N6"
    },
    {
     "weight": 0.089,
     "shared_ref": "N7"
    },
    {
     "weight": 0.086,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R053
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.956,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.959,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.097,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.146,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.035,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.159,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.128,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.085,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.073,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.066,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.065,
       "feature": "NOT crown color: brown"
      }
     ]
    },
    {
     "weight": 0.144,
     "id": "N4",
     "operator": "A",
     "name": "arithmetic mean",
     "andness": 0.5,
     "verbalization": "nice to have",
     "children": [
      {
       "weight": 0.361,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.33,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.218,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.045,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.006,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.005,
       "feature": "NOT crown color: white"
      }
     ]
    },
    {
     "weight": 0.125,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.184,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.303,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.216,
       "feature": "bill color: black"
      },
      {
       "weight": 0.166,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.156,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.13,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.03,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N6",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.721,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.76,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.132,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.021,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.02,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.02,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.018,
       "feature": "NOT primary color: buff"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N7",
     "operator": "SC",
     "name": "medium soft conjunction",
     "andness": 0.672,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.364,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.311,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.098,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.049,
       "feature": "primary color: white"
      },
      {
       "weight": 0.047,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.038,
       "feature": "NOT wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N8",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.905,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.624,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.376,
       "feature": "upper tail color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.041,
   "id": "N9",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.212,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.146,
     "shared_ref": "N3"
    },
    {
     "weight": 0.144,
     "shared_ref": "N4"
    },
    {
     "weight": 0.125,
     "shared_ref": "N5"
    },
    {
     "weight": 0.092,
     "shared_ref": "N6"
    },
    {
     "weight": 0.076,
     "shared_ref": "N7"
    },
    {
     "weight": 0.073,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R118
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.113,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.558,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.981,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.193,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.471,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.199,
       "feature": "throat color: black"
      },
      {
       "weight": 0.143,
       "feature": "primary color: white"
      },
      {
       "weight": 0.137,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.099,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.083,
       "feature": "breast color: white"
      },
      {
       "weight": 0.08,
       "feature": "back color: black"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.479,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.178,
       "feature": "throat color: black"
      },
      {
       "weight": 0.128,
       "feature": "primary color: white"
      },
      {
       "weight": 0.122,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.088,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.074,
       "feature": "breast color: white"
      },
      {
       "weight": 0.072,
       "feature": "back color: black"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.454,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.1,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.097,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.088,
       "feature": "NOT bill color: grey"
      },
      {
       "weight": 0.081,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.079,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.078,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.604,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.066,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.05,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.047,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.046,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.045,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.044,
       "feature": "NOT shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.376,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.169,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.157,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.15,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.139,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.138,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.126,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.083,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.426,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.374,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.184,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.154,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.146,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.143,
       "feature": "NOT back color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.442,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.929,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.193,
     "shared_ref": "N3"
    },
    {
     "weight": 0.104,
     "shared_ref": "N4"
    },
    {
     "weight": 0.091,
     "shared_ref": "N5"
    },
    {
     "weight": 0.089,
     "shared_ref": "N6"
    },
    {
     "weight": 0.087,
     "shared_ref": "N7"
    },
    {
     "weight": 0.083,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R013
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.922,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.507,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.095,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.206,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.361,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.565,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.25,
       "feature": "nape color: black"
      },
      {
       "weight": 0.165,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.009,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.006,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.003,
       "feature": "NOT size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.202,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.7,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.119,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.107,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.074,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.244,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.61,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.192,
       "feature": "bill color: black"
      },
      {
       "weight": 0.104,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.093,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.306,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.439,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.183,
       "feature": "leg color: black"
      },
      {
       "weight": 0.176,
       "feature": "back color: black"
      },
      {
       "weight": 0.105,
       "feature": "bill color: black"
      },
      {
       "weight": 0.096,
       "feature": "wing color: black"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N7",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.211,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.696,
       "feature": "belly color: black"
      },
      {
       "weight": 0.267,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.032,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.006,
       "feature": "crown color: blue"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N8",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.25,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.513,
       "feature": "belly color: black"
      },
      {
       "weight": 0.267,
       "feature": "breast color: black"
      },
      {
       "weight": 0.197,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.023,
       "feature": "throat color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.493,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.868,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.206,
     "shared_ref": "N3"
    },
    {
     "weight": 0.117,
     "shared_ref": "N4"
    },
    {
     "weight": 0.114,
     "shared_ref": "N5"
    },
    {
     "weight": 0.108,
     "shared_ref": "N6"
    },
    {
     "weight": 0.092,
     "shared_ref": "N7"
    },
    {
     "weight": 0.091,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R198
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 0.974,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.507,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.014,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.205,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.547,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.383,
       "feature": "nape color: white"
      },
      {
       "weight": 0.068,
       "feature": "belly color: black"
      },
      {
       "weight": 0.067,
       "feature": "throat color: black"
      },
      {
       "weight": 0.063,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.042,
       "feature": "breast color: black"
      },
      {
       "weight": 0.04,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.148,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.447,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.424,
       "feature": "nape color: white"
      },
      {
       "weight": 0.076,
       "feature": "belly color: black"
      },
      {
       "weight": 0.075,
       "feature": "throat color: black"
      },
      {
       "weight": 0.047,
       "feature": "breast color: black"
      },
      {
       "weight": 0.045,
       "feature": "leg color: black"
      },
      {
       "weight": 0.044,
       "feature": "wing pattern: solid"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.181,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.109,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.106,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.097,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.096,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.093,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.09,
       "feature": "NOT upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.283,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.221,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.198,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.112,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.099,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.098,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.095,
       "feature": "underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.436,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.074,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.065,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.063,
       "feature": "NOT leg color: grey"
      },
      {
       "weight": 0.058,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.057,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.052,
       "feature": "NOT crown color: white"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.217,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.094,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.062,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.061,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.058,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.055,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.055,
       "feature": "wing pattern: spotted"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.493,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.485,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.205,
     "shared_ref": "N3"
    },
    {
     "weight": 0.148,
     "shared_ref": "N4"
    },
    {
     "weight": 0.089,
     "shared_ref": "N5"
    },
    {
     "weight": 0.088,
     "shared_ref": "N6"
    },
    {
     "weight": 0.082,
     "shared_ref": "N7"
    },
    {
     "weight": 0.08,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R093
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.045,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.519,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.139,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.235,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.347,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.356,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.248,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.166,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.078,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.069,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.058,
       "feature": "upper tail color: black"
      }
     ]
    },
    {
     "weight": 0.134,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.377,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.228,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.152,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.129,
       "feature": "breast color: black"
      },
      {
       "weight": 0.081,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.072,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.064,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.12,
     "id": "N5",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.02,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 1.0,
       "feature": "throat color: black"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.37,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.344,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.136,
       "feature": "breast color: black"
      },
      {
       "weight": 0.086,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.07,
       "feature": "throat color: black"
      },
      {
       "weight": 0.061,
       "feature": "back color: black"
      },
      {
       "weight": 0.056,
       "feature": "upper tail color: black"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N7",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.003,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.285,
       "feature": "nape color: black"
      },
      {
       "weight": 0.201,
       "feature": "wing color: black"
      },
      {
       "weight": 0.189,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.151,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.131,
       "feature": "bill color: black"
      },
      {
       "weight": 0.024,
       "feature": "NOT breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.134,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.978,
       "feature": "nape color: black"
      },
      {
       "weight": 0.022,
       "feature": "NOT underparts color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.481,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.135,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.235,
     "shared_ref": "N3"
    },
    {
     "weight": 0.134,
     "shared_ref": "N4"
    },
    {
     "weight": 0.12,
     "shared_ref": "N5"
    },
    {
     "weight": 0.108,
     "shared_ref": "N6"
    },
    {
     "weight": 0.108,
     "shared_ref": "N7"
    },
    {
     "weight": 0.108,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R037
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.144,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.589,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.953,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.101,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.663,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.142,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.082,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.075,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.055,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.051,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.048,
       "feature": "NOT back pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.676,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.068,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.064,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.063,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.06,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.06,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.058,
       "feature": "NOT crown color: grey"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.514,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.166,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.15,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.128,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.101,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.096,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.096,
       "feature": "NOT nape color: yellow"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.504,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.848,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.019,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.017,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.017,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.016,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.015,
       "feature": "NOT bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.524,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.184,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.161,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.13,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.091,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.087,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.074,
       "feature": "NOT primary color: buff"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.488,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.149,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.132,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.121,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.121,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.116,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.101,
       "feature": "NOT throat color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.411,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.242,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.101,
     "shared_ref": "N3"
    },
    {
     "weight": 0.094,
     "shared_ref": "N4"
    },
    {
     "weight": 0.087,
     "shared_ref": "N5"
    },
    {
     "weight": 0.086,
     "shared_ref": "N6"
    },
    {
     "weight": 0.084,
     "shared_ref": "N7"
    },
    {
     "weight": 0.084,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R026
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.885,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.761,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.931,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.132,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.292,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.296,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.133,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.096,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.078,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.055,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.052,
       "feature": "crown color: grey"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.642,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.056,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.054,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.047,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.044,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.044,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.043,
       "feature": "NOT upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.628,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.075,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.061,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.059,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.059,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.052,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.051,
       "feature": "NOT upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.232,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.25,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.177,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.142,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.103,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.081,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.069,
       "feature": "primary color: black"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.657,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.057,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.053,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.052,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.046,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.043,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.041,
       "feature": "NOT wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.617,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.091,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.061,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.044,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.043,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.04,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.037,
       "feature": "NOT crown color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.239,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.884,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.132,
     "shared_ref": "N3"
    },
    {
     "weight": 0.101,
     "shared_ref": "N4"
    },
    {
     "weight": 0.101,
     "shared_ref": "N5"
    },
    {
     "weight": 0.097,
     "shared_ref": "N6"
    },
    {
     "weight": 0.096,
     "shared_ref": "N7"
    },
    {
     "weight": 0.095,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R169
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.038,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.517,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.964,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.182,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.509,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.29,
       "feature": "primary color: white"
      },
      {
       "weight": 0.178,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.087,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.084,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.076,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.063,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.154,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.459,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.158,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.135,
       "feature": "leg color: black"
      },
      {
       "weight": 0.123,
       "feature": "wing color: black"
      },
      {
       "weight": 0.105,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.095,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.079,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.143,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.434,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.468,
       "feature": "primary color: white"
      },
      {
       "weight": 0.203,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.173,
       "feature": "leg color: black"
      },
      {
       "weight": 0.105,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.051,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.404,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.404,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.222,
       "feature": "wing color: black"
      },
      {
       "weight": 0.197,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.126,
       "feature": "throat color: white"
      },
      {
       "weight": 0.05,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.286,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.501,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.187,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.16,
       "feature": "back color: grey"
      },
      {
       "weight": 0.152,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.222,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 1.0,
       "feature": "primary color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.483,
   "id": "N9",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.175,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.182,
     "shared_ref": "N3"
    },
    {
     "weight": 0.154,
     "shared_ref": "N4"
    },
    {
     "weight": 0.143,
     "shared_ref": "N5"
    },
    {
     "weight": 0.108,
     "shared_ref": "N6"
    },
    {
     "weight": 0.086,
     "shared_ref": "N7"
    },
    {
     "weight": 0.085,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R030
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.614,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.669,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.003,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.141,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.609,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.171,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.13,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.12,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.103,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.102,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.072,
       "feature": "primary color: brown"
      }
     ]
    },
    {
     "weight": 0.134,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.572,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.19,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.113,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.075,
       "feature": "throat color: white"
      },
      {
       "weight": 0.072,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.069,
       "feature": "back color: buff"
      },
      {
       "weight": 0.066,
       "feature": "leg color: buff"
      }
     ]
    },
    {
     "weight": 0.128,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.48,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.199,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.196,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.158,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.092,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.087,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.082,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.113,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.314,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.183,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.171,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.103,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.092,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.089,
       "feature": "back color: buff"
      },
      {
       "weight": 0.081,
       "feature": "upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.069,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.356,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.102,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.094,
       "feature": "crown color: white"
      },
      {
       "weight": 0.084,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.083,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.071,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.07,
       "feature": "NOT wing color: black"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.124,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.115,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.114,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.109,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.105,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.103,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.083,
       "feature": "NOT under tail color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.331,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.437,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.141,
     "shared_ref": "N3"
    },
    {
     "weight": 0.134,
     "shared_ref": "N4"
    },
    {
     "weight": 0.128,
     "shared_ref": "N5"
    },
    {
     "weight": 0.113,
     "shared_ref": "N6"
    },
    {
     "weight": 0.069,
     "shared_ref": "N7"
    },
    {
     "weight": 0.064,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R162
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.819,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.929,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.306,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.866,
     "id": "N3",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.746,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.173,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.161,
       "feature": "throat color: black"
      },
      {
       "weight": 0.129,
       "feature": "breast color: black"
      },
      {
       "weight": 0.119,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.078,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.044,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.013,
     "id": "N4",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.83,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.352,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.104,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.082,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.074,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.07,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.068,
       "feature": "NOT throat color: grey"
      }
     ]
    },
    {
     "weight": 0.013,
     "id": "N5",
     "operator": "SD-",
     "name": "low soft disjunction",
     "andness": 0.394,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.497,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.296,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.11,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.097,
       "feature": "nape color: grey"
      }
     ]
    },
    {
     "weight": 0.012,
     "id": "N6",
     "operator": "SC-",
     "name": "low soft conjunction",
     "andness": 0.601,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.309,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.288,
       "feature": "throat color: black"
      },
      {
       "weight": 0.214,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.105,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.083,
       "feature": "nape color: grey"
      }
     ]
    },
    {
     "weight": 0.011,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.209,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.186,
       "feature": "belly color: white"
      },
      {
       "weight": 0.052,
       "feature": "NOT primary color: white"
      },
      {
       "weight": 0.051,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.05,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.047,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.042,
       "feature": "NOT bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.011,
     "id": "N8",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.06,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.201,
       "feature": "bill color: black"
      },
      {
       "weight": 0.133,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.084,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.068,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.065,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.054,
       "feature": "NOT wing color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.071,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.139,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.866,
     "shared_ref": "N3"
    },
    {
     "weight": 0.013,
     "shared_ref": "N4"
    },
    {
     "weight": 0.013,
     "shared_ref": "N5"
    },
    {
     "weight": 0.012,
     "shared_ref": "N6"
    },
    {
     "weight": 0.011,
     "shared_ref": "N7"
    },
    {
     "weight": 0.011,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R142
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.27,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.773,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.241,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.252,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.441,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.31,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.23,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.105,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.069,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.063,
       "feature": "back color: buff"
      },
      {
       "weight": 0.055,
       "feature": "primary color: brown"
      }
     ]
    },
    {
     "weight": 0.212,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.497,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.272,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.202,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.092,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.061,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.055,
       "feature": "back color: buff"
      },
      {
       "weight": 0.048,
       "feature": "primary color: brown"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N5",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.308,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.969,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.029,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.002,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.0,
       "feature": "NOT size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N6",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.135,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.572,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.422,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.003,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.001,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.001,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.0,
       "feature": "NOT belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.048,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.996,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.002,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.001,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.0,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.0,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.0,
       "feature": "NOT underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.457,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.394,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.32,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.173,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.071,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.042,
       "feature": "eye color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.227,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.002,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.252,
     "shared_ref": "N3"
    },
    {
     "weight": 0.212,
     "shared_ref": "N4"
    },
    {
     "weight": 0.104,
     "shared_ref": "N5"
    },
    {
     "weight": 0.095,
     "shared_ref": "N6"
    },
    {
     "weight": 0.088,
     "shared_ref": "N7"
    },
    {
     "weight": 0.082,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R105
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.897,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.722,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.029,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.759,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.753,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.105,
       "feature": "throat color: white"
      },
      {
       "weight": 0.083,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.08,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.08,
       "feature": "breast color: white"
      },
      {
       "weight": 0.078,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.068,
       "feature": "crown color: brown"
      }
     ]
    },
    {
     "weight": 0.051,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.753,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.113,
       "feature": "throat color: white"
      },
      {
       "weight": 0.089,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.086,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.085,
       "feature": "breast color: white"
      },
      {
       "weight": 0.084,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.072,
       "feature": "crown color: brown"
      }
     ]
    },
    {
     "weight": 0.032,
     "id": "N5",
     "operator": "A",
     "name": "arithmetic mean",
     "andness": 0.534,
     "verbalization": "nice to have",
     "children": [
      {
       "weight": 0.417,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.278,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.156,
       "feature": "back color: buff"
      },
      {
       "weight": 0.15,
       "feature": "back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.021,
     "id": "N6",
     "operator": "SC",
     "name": "medium soft conjunction",
     "andness": 0.652,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.393,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.31,
       "feature": "back color: buff"
      },
      {
       "weight": 0.298,
       "feature": "back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.017,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.052,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.783,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.037,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.036,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.035,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.029,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.028,
       "feature": "belly color: black"
      }
     ]
    },
    {
     "weight": 0.016,
     "id": "N8",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.225,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.746,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.071,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.064,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.063,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.056,
       "feature": "breast pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.278,
   "id": "N9",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.044,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.759,
     "shared_ref": "N3"
    },
    {
     "weight": 0.051,
     "shared_ref": "N4"
    },
    {
     "weight": 0.032,
     "shared_ref": "N5"
    },
    {
     "weight": 0.021,
     "shared_ref": "N6"
    },
    {
     "weight": 0.017,
     "shared_ref": "N7"
    },
    {
     "weight": 0.016,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R043
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.942,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.531,
   "id": "N2",
   "operator": "HHD",
   "name": "high hyper-disjunction",
   "andness": -0.33,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.092,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.583,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.042,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.037,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.034,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.034,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.032,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.032,
       "feature": "NOT underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.329,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.112,
       "feature": "NOT bill color: grey"
      },
      {
       "weight": 0.11,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.104,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.104,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.094,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.094,
       "feature": "NOT forehead color: yellow"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.509,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.1,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.062,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.051,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.05,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.049,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.042,
       "feature": "NOT under tail color: grey"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.574,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.085,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.056,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.035,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.035,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.034,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.029,
       "feature": "NOT bill color: grey"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.454,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.254,
       "feature": "back color: black"
      },
      {
       "weight": 0.177,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.063,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.061,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.049,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.047,
       "feature": "NOT crown color: white"
      }
     ]
    },
    {
     "weight": 0.083,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.304,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.261,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.218,
       "feature": "throat color: white"
      },
      {
       "weight": 0.183,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.152,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.071,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.06,
       "feature": "bill length: shorter than head"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.469,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.431,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.092,
     "shared_ref": "N3"
    },
    {
     "weight": 0.091,
     "shared_ref": "N4"
    },
    {
     "weight": 0.089,
     "shared_ref": "N5"
    },
    {
     "weight": 0.087,
     "shared_ref": "N6"
    },
    {
     "weight": 0.086,
     "shared_ref": "N7"
    },
    {
     "weight": 0.083,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R029
```json
{
 "id": "N1",
 "operator": "DP",
 "name": "product t-conorm",
 "andness": -0.235,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.911,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.049,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.889,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.958,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.28,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.147,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.12,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.106,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.089,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.073,
       "feature": "forehead color: grey"
      }
     ]
    },
    {
     "weight": 0.032,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.065,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.593,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.169,
       "feature": "back color: brown"
      },
      {
       "weight": 0.131,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.058,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.048,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.016,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.073,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.384,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.33,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.103,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.079,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.062,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.025,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.008,
     "id": "N6",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.764,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 1.0,
       "feature": "breast color: grey"
      }
     ]
    },
    {
     "weight": 0.008,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.227,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.321,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.19,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.063,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.059,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.045,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.045,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.006,
     "id": "N8",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.754,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.836,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.085,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.076,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.003,
       "feature": "NOT bill shape: dagger"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.089,
   "id": "N9",
   "operator": "SC+",
   "name": "high soft conjunction",
   "andness": 0.727,
   "verbalization": "nice to have most",
   "children": [
    {
     "weight": 0.889,
     "shared_ref": "N3"
    },
    {
     "weight": 0.032,
     "shared_ref": "N4"
    },
    {
     "weight": 0.016,
     "shared_ref": "N5"
    },
    {
     "weight": 0.008,
     "shared_ref": "N6"
    },
    {
     "weight": 0.008,
     "shared_ref": "N7"
    },
    {
     "weight": 0.006,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R076
```json
{
 "id": "N1",
 "operator": "HD-",
 "name": "low hard disjunction",
 "andness": 0.25,
 "verbalization": "enough to have any",
 "children": [
  {
   "weight": 0.878,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.248,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.543,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.21,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.274,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.154,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.124,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.113,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.106,
       "feature": "back color: buff"
      },
      {
       "weight": 0.083,
       "feature": "bill shape: cone"
      }
     ]
    },
    {
     "weight": 0.129,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.303,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.32,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.257,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.173,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.173,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.076,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.078,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.362,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.282,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.262,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.255,
       "feature": "crown color: white"
      },
      {
       "weight": 0.201,
       "feature": "belly color: grey"
      }
     ]
    },
    {
     "weight": 0.062,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.838,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.041,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.04,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.037,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.037,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.037,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.036,
       "feature": "throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.06,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.557,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.148,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.136,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.134,
       "feature": "crown color: white"
      },
      {
       "weight": 0.106,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.1,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.095,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.06,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.869,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.049,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.036,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.033,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.033,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.033,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.033,
       "feature": "underparts color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.122,
   "id": "N9",
   "operator": "SC+",
   "name": "high soft conjunction",
   "andness": 0.689,
   "verbalization": "nice to have most",
   "children": [
    {
     "weight": 0.543,
     "shared_ref": "N3"
    },
    {
     "weight": 0.129,
     "shared_ref": "N4"
    },
    {
     "weight": 0.078,
     "shared_ref": "N5"
    },
    {
     "weight": 0.062,
     "shared_ref": "N6"
    },
    {
     "weight": 0.06,
     "shared_ref": "N7"
    },
    {
     "weight": 0.06,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R117
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.399,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.549,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.178,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.114,
     "id": "N3",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.096,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.996,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.004,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.0,
       "feature": "NOT crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.09,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.518,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.455,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.021,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.002,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.001,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.001,
       "feature": "NOT leg color: black"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N5",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.02,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.7,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.3,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.049,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.959,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.039,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.001,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.001,
       "feature": "NOT forehead color: grey"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.318,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.075,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.07,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.07,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.067,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.067,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.065,
       "feature": "NOT bill shape: dagger"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.497,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.275,
       "feature": "back color: grey"
      },
      {
       "weight": 0.228,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.151,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.067,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.065,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.055,
       "feature": "forehead color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.451,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.41,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.114,
     "shared_ref": "N3"
    },
    {
     "weight": 0.111,
     "shared_ref": "N4"
    },
    {
     "weight": 0.105,
     "shared_ref": "N5"
    },
    {
     "weight": 0.105,
     "shared_ref": "N6"
    },
    {
     "weight": 0.089,
     "shared_ref": "N7"
    },
    {
     "weight": 0.084,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R173
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 0.996,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.546,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.156,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.142,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.188,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.58,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.323,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.097,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.133,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.256,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.246,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.241,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.109,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.087,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.083,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.075,
       "feature": "breast color: buff"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.414,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.195,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.191,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.096,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.076,
       "feature": "back color: buff"
      },
      {
       "weight": 0.054,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.048,
       "feature": "nape color: buff"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.434,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.116,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.104,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.091,
       "feature": "back color: buff"
      },
      {
       "weight": 0.083,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.071,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.065,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.125,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.606,
       "feature": "back color: brown"
      },
      {
       "weight": 0.241,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.153,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.061,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.303,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.139,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.133,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.124,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.063,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.063,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.057,
       "feature": "NOT forehead color: blue"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.454,
   "id": "N9",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.258,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.142,
     "shared_ref": "N3"
    },
    {
     "weight": 0.133,
     "shared_ref": "N4"
    },
    {
     "weight": 0.107,
     "shared_ref": "N5"
    },
    {
     "weight": 0.105,
     "shared_ref": "N6"
    },
    {
     "weight": 0.088,
     "shared_ref": "N7"
    },
    {
     "weight": 0.061,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R095
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.869,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.759,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.918,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.138,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.468,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.17,
       "feature": "wing color: white"
      },
      {
       "weight": 0.141,
       "feature": "breast color: black"
      },
      {
       "weight": 0.082,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.071,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.053,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.053,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.657,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.056,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.056,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.05,
       "feature": "NOT bill length: about the same as head"
      },
      {
       "weight": 0.046,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.045,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.113,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.653,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.05,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.048,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.046,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.045,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.04,
       "feature": "NOT bill length: about the same as head"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.65,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.077,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.075,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.06,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.058,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.055,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.052,
       "feature": "NOT belly color: black"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.551,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.185,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.144,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.127,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.117,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.099,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.092,
       "feature": "NOT forehead color: white"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.372,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.276,
       "feature": "wing color: white"
      },
      {
       "weight": 0.115,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.108,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.099,
       "feature": "breast color: white"
      },
      {
       "weight": 0.087,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.08,
       "feature": "back color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.241,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.207,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.138,
     "shared_ref": "N3"
    },
    {
     "weight": 0.114,
     "shared_ref": "N4"
    },
    {
     "weight": 0.113,
     "shared_ref": "N5"
    },
    {
     "weight": 0.11,
     "shared_ref": "N6"
    },
    {
     "weight": 0.109,
     "shared_ref": "N7"
    },
    {
     "weight": 0.1,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R191
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.131,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.641,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.906,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.137,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.481,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.095,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.073,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.071,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.068,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.067,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.064,
       "feature": "NOT belly color: grey"
      }
     ]
    },
    {
     "weight": 0.135,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.406,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.148,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.147,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.136,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.127,
       "feature": "NOT leg color: grey"
      },
      {
       "weight": 0.12,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.109,
       "feature": "NOT crown color: grey"
      }
     ]
    },
    {
     "weight": 0.13,
     "id": "N5",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.219,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.373,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.351,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.276,
       "feature": "NOT tail pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.129,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.681,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.079,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.061,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.058,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.053,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.05,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.049,
       "feature": "NOT belly color: brown"
      }
     ]
    },
    {
     "weight": 0.128,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.693,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.055,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.052,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.046,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.043,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.041,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.041,
       "feature": "NOT bill shape: dagger"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.288,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.242,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.137,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.064,
       "feature": "back color: black"
      },
      {
       "weight": 0.063,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.059,
       "feature": "nape color: black"
      },
      {
       "weight": 0.052,
       "feature": "breast color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.359,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.417,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.137,
     "shared_ref": "N3"
    },
    {
     "weight": 0.135,
     "shared_ref": "N4"
    },
    {
     "weight": 0.13,
     "shared_ref": "N5"
    },
    {
     "weight": 0.129,
     "shared_ref": "N6"
    },
    {
     "weight": 0.128,
     "shared_ref": "N7"
    },
    {
     "weight": 0.097,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R057
```json
{
 "id": "N1",
 "operator": "D",
 "name": "pure disjunction",
 "andness": -0.014,
 "verbalization": "decided by highest",
 "children": [
  {
   "weight": 0.86,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.093,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.334,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.243,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.229,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.175,
       "feature": "crown color: black"
      },
      {
       "weight": 0.109,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.1,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.073,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.068,
       "feature": "upper tail color: buff"
      }
     ]
    },
    {
     "weight": 0.333,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.262,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.193,
       "feature": "crown color: black"
      },
      {
       "weight": 0.12,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.112,
       "feature": "back color: buff"
      },
      {
       "weight": 0.11,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.08,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.072,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.244,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.301,
       "feature": "back color: buff"
      },
      {
       "weight": 0.26,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.176,
       "feature": "wing color: black"
      },
      {
       "weight": 0.072,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.064,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.046,
       "feature": "bill shape: cone"
      }
     ]
    },
    {
     "weight": 0.061,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.622,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.139,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.118,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.113,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.112,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.055,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.042,
       "feature": "belly color: grey"
      }
     ]
    },
    {
     "weight": 0.044,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.647,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.167,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.131,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.105,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.083,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.078,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.053,
       "feature": "NOT primary color: brown"
      }
     ]
    },
    {
     "weight": 0.042,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.587,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.205,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.118,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.078,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.075,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.056,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.053,
       "feature": "primary color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.14,
   "id": "N9",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.129,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.334,
     "shared_ref": "N3"
    },
    {
     "weight": 0.333,
     "shared_ref": "N4"
    },
    {
     "weight": 0.109,
     "shared_ref": "N5"
    },
    {
     "weight": 0.061,
     "shared_ref": "N6"
    },
    {
     "weight": 0.044,
     "shared_ref": "N7"
    },
    {
     "weight": 0.042,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R194
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.258,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.614,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.28,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.125,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.216,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.281,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.158,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.143,
       "feature": "back color: grey"
      },
      {
       "weight": 0.104,
       "feature": "breast color: black"
      },
      {
       "weight": 0.08,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.074,
       "feature": "wing color: buff"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.124,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.22,
       "feature": "throat color: black"
      },
      {
       "weight": 0.108,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.063,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.055,
       "feature": "back color: grey"
      },
      {
       "weight": 0.052,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.046,
       "feature": "breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.208,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.126,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.097,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.087,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.083,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.08,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.07,
       "feature": "under tail color: brown"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.182,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.147,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.139,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.116,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.116,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.102,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.092,
       "feature": "NOT upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.521,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.079,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.074,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.074,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.069,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.069,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.067,
       "feature": "NOT under tail color: grey"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.339,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.118,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.092,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.088,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.078,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.073,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.073,
       "feature": "NOT shape: duck-like"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.386,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.171,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.125,
     "shared_ref": "N3"
    },
    {
     "weight": 0.109,
     "shared_ref": "N4"
    },
    {
     "weight": 0.104,
     "shared_ref": "N5"
    },
    {
     "weight": 0.087,
     "shared_ref": "N6"
    },
    {
     "weight": 0.085,
     "shared_ref": "N7"
    },
    {
     "weight": 0.082,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R137
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.407,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.975,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.79,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.357,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.151,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.317,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.204,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.19,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.109,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.098,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.081,
       "feature": "wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.349,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.227,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.209,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.185,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.173,
       "feature": "crown color: black"
      },
      {
       "weight": 0.102,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.096,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.064,
       "feature": "upperparts color: black"
      }
     ]
    },
    {
     "weight": 0.066,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.625,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.158,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.157,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.123,
       "feature": "crown color: black"
      },
      {
       "weight": 0.09,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.085,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.078,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.059,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.506,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.361,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.206,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.18,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.104,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.092,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.057,
       "feature": "wing color: buff"
      }
     ]
    },
    {
     "weight": 0.059,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.203,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.606,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.155,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.138,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.061,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.039,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.038,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.434,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.571,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.282,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.055,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.054,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.039,
       "feature": "wing shape: rounded-wings"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.025,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.765,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.357,
     "shared_ref": "N3"
    },
    {
     "weight": 0.349,
     "shared_ref": "N4"
    },
    {
     "weight": 0.066,
     "shared_ref": "N5"
    },
    {
     "weight": 0.059,
     "shared_ref": "N6"
    },
    {
     "weight": 0.059,
     "shared_ref": "N7"
    },
    {
     "weight": 0.038,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R153
```json
{
 "id": "N1",
 "operator": "D",
 "name": "pure disjunction",
 "andness": -0.021,
 "verbalization": "decided by highest",
 "children": [
  {
   "weight": 0.593,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.907,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.797,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.089,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.175,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.174,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.129,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.115,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.107,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.049,
       "feature": "crown color: brown"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.221,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.265,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.197,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.096,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.075,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.066,
       "feature": "belly color: white"
      },
      {
       "weight": 0.058,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.072,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.263,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.202,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.112,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.086,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.076,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.072,
       "feature": "breast color: white"
      },
      {
       "weight": 0.067,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.009,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.45,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.313,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.267,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.167,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.107,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.094,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.027,
       "feature": "NOT underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.007,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.596,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.241,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.198,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.113,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.077,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.044,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.033,
       "feature": "NOT back color: buff"
      }
     ]
    },
    {
     "weight": 0.004,
     "id": "N8",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.948,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.173,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.125,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.115,
       "feature": "breast color: black"
      },
      {
       "weight": 0.096,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.066,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.062,
       "feature": "size: very small (3 - 5 in)"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.407,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.122,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.797,
     "shared_ref": "N3"
    },
    {
     "weight": 0.089,
     "shared_ref": "N4"
    },
    {
     "weight": 0.072,
     "shared_ref": "N5"
    },
    {
     "weight": 0.009,
     "shared_ref": "N6"
    },
    {
     "weight": 0.007,
     "shared_ref": "N7"
    },
    {
     "weight": 0.004,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R120
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.155,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.564,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.091,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.295,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.32,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.562,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.397,
       "feature": "back color: brown"
      },
      {
       "weight": 0.041,
       "feature": "wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.247,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.326,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.572,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.226,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.117,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.039,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.034,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.012,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.172,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.257,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.484,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.392,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.099,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.025,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.07,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.051,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.987,
       "feature": "back color: brown"
      },
      {
       "weight": 0.007,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.007,
       "feature": "tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N7",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.058,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.558,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.432,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.005,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.005,
       "feature": "tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N8",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.008,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.935,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.065,
       "feature": "shape: perching-like"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.436,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.941,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.295,
     "shared_ref": "N3"
    },
    {
     "weight": 0.247,
     "shared_ref": "N4"
    },
    {
     "weight": 0.172,
     "shared_ref": "N5"
    },
    {
     "weight": 0.07,
     "shared_ref": "N6"
    },
    {
     "weight": 0.064,
     "shared_ref": "N7"
    },
    {
     "weight": 0.064,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R086
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.185,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.644,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.18,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.11,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.485,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.275,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.273,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.234,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.218,
       "feature": "NOT nape color: yellow"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N4",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.033,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.841,
       "feature": "back color: brown"
      },
      {
       "weight": 0.035,
       "feature": "back color: white"
      },
      {
       "weight": 0.032,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.021,
       "feature": "NOT primary color: white"
      },
      {
       "weight": 0.02,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.013,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.034,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.591,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.344,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.023,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.023,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.018,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.554,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.345,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.344,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.311,
       "feature": "NOT wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.721,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.069,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.061,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.042,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.042,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.038,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.038,
       "feature": "NOT nape color: buff"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.317,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 1.0,
       "feature": "NOT breast color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.356,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.524,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.11,
     "shared_ref": "N3"
    },
    {
     "weight": 0.102,
     "shared_ref": "N4"
    },
    {
     "weight": 0.102,
     "shared_ref": "N5"
    },
    {
     "weight": 0.1,
     "shared_ref": "N6"
    },
    {
     "weight": 0.1,
     "shared_ref": "N7"
    },
    {
     "weight": 0.096,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R187
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.311,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.54,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.972,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.173,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.543,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.154,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.078,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.074,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.069,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.068,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.063,
       "feature": "back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.486,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.185,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.093,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.089,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.082,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.075,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.064,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N5",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.021,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.293,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.219,
       "feature": "breast color: white"
      },
      {
       "weight": 0.121,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.101,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.084,
       "feature": "nape color: white"
      },
      {
       "weight": 0.074,
       "feature": "wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.066,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.567,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.138,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.075,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.071,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.071,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.069,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.066,
       "feature": "NOT belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.063,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.557,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.168,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.136,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.104,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.086,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.084,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.078,
       "feature": "NOT belly color: brown"
      }
     ]
    },
    {
     "weight": 0.061,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.522,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.13,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.116,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.087,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.058,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.055,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.053,
       "feature": "NOT underparts color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.46,
   "id": "N9",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.005,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.173,
     "shared_ref": "N3"
    },
    {
     "weight": 0.11,
     "shared_ref": "N4"
    },
    {
     "weight": 0.092,
     "shared_ref": "N5"
    },
    {
     "weight": 0.066,
     "shared_ref": "N6"
    },
    {
     "weight": 0.063,
     "shared_ref": "N7"
    },
    {
     "weight": 0.061,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R195
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 1.03,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.651,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": -0.013,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.242,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.506,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.241,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.193,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.15,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.146,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.067,
       "feature": "back color: buff"
      },
      {
       "weight": 0.047,
       "feature": "throat color: white"
      }
     ]
    },
    {
     "weight": 0.228,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.372,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.275,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.215,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.208,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.116,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.096,
       "feature": "back color: buff"
      },
      {
       "weight": 0.047,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.12,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.386,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.397,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.134,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.106,
       "feature": "breast color: white"
      },
      {
       "weight": 0.077,
       "feature": "throat color: white"
      },
      {
       "weight": 0.075,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.05,
       "feature": "belly color: white"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.183,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.23,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.23,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.147,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.14,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.139,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.114,
       "feature": "NOT crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.062,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.264,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.267,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.137,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.133,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.123,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.092,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.091,
       "feature": "NOT breast color: buff"
      }
     ]
    },
    {
     "weight": 0.06,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.044,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.912,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.019,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.013,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.011,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.01,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.008,
       "feature": "NOT wing pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.349,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.065,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.242,
     "shared_ref": "N3"
    },
    {
     "weight": 0.228,
     "shared_ref": "N4"
    },
    {
     "weight": 0.12,
     "shared_ref": "N5"
    },
    {
     "weight": 0.064,
     "shared_ref": "N6"
    },
    {
     "weight": 0.062,
     "shared_ref": "N7"
    },
    {
     "weight": 0.06,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R001
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.897,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.507,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.026,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.097,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.598,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.104,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.067,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.066,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.066,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.063,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.062,
       "feature": "NOT crown color: blue"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.445,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.193,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.137,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.114,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.098,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.089,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.088,
       "feature": "NOT wing color: white"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.593,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.142,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.07,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.061,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.034,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.033,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.032,
       "feature": "NOT belly color: black"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.626,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.07,
       "feature": "NOT breast color: white"
      },
      {
       "weight": 0.059,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.057,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.052,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.049,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.048,
       "feature": "NOT upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.094,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.617,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.113,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.084,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.048,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.043,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.042,
       "feature": "NOT bill length: about the same as head"
      },
      {
       "weight": 0.039,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.364,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.263,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.23,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.218,
       "feature": "crown color: black"
      },
      {
       "weight": 0.111,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.09,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.088,
       "feature": "back color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.493,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.908,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.097,
     "shared_ref": "N3"
    },
    {
     "weight": 0.096,
     "shared_ref": "N4"
    },
    {
     "weight": 0.094,
     "shared_ref": "N5"
    },
    {
     "weight": 0.094,
     "shared_ref": "N6"
    },
    {
     "weight": 0.094,
     "shared_ref": "N7"
    },
    {
     "weight": 0.088,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R075
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.433,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.982,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.835,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.673,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.095,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.163,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.102,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.092,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.087,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.086,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.084,
       "feature": "upperparts color: buff"
      }
     ]
    },
    {
     "weight": 0.074,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.482,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.504,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.126,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.111,
       "feature": "crown color: black"
      },
      {
       "weight": 0.1,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.059,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.05,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.059,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.459,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.374,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.29,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.124,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.123,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.055,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.035,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.057,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.572,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.359,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.298,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.241,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.102,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.038,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.623,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.212,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.163,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.143,
       "feature": "back color: buff"
      },
      {
       "weight": 0.092,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.077,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.076,
       "feature": "bill shape: cone"
      }
     ]
    },
    {
     "weight": 0.032,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.599,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.311,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.208,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.176,
       "feature": "back color: buff"
      },
      {
       "weight": 0.117,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.11,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.079,
       "feature": "upperparts color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.018,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.752,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.673,
     "shared_ref": "N3"
    },
    {
     "weight": 0.074,
     "shared_ref": "N4"
    },
    {
     "weight": 0.059,
     "shared_ref": "N5"
    },
    {
     "weight": 0.057,
     "shared_ref": "N6"
    },
    {
     "weight": 0.038,
     "shared_ref": "N7"
    },
    {
     "weight": 0.032,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R161
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.721,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.817,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.182,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.417,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.168,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.246,
       "feature": "nape color: black"
      },
      {
       "weight": 0.118,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.077,
       "feature": "belly color: black"
      },
      {
       "weight": 0.065,
       "feature": "back color: black"
      },
      {
       "weight": 0.061,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.06,
       "feature": "breast color: black"
      }
     ]
    },
    {
     "weight": 0.274,
     "id": "N4",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.157,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.991,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.005,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.002,
       "feature": "crown color: white"
      },
      {
       "weight": 0.002,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.002,
       "feature": "wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N5",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.053,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.987,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.003,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.003,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.003,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.002,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.002,
       "feature": "back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N6",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.075,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.554,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.419,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.009,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.004,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.004,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.002,
       "feature": "NOT breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.026,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.212,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.113,
       "feature": "back color: black"
      },
      {
       "weight": 0.105,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.101,
       "feature": "throat color: black"
      },
      {
       "weight": 0.076,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.073,
       "feature": "primary color: black"
      },
      {
       "weight": 0.068,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.021,
     "id": "N8",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.003,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.382,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.24,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.162,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.116,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.099,
       "feature": "NOT upperparts color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.183,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.897,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.417,
     "shared_ref": "N3"
    },
    {
     "weight": 0.274,
     "shared_ref": "N4"
    },
    {
     "weight": 0.116,
     "shared_ref": "N5"
    },
    {
     "weight": 0.087,
     "shared_ref": "N6"
    },
    {
     "weight": 0.026,
     "shared_ref": "N7"
    },
    {
     "weight": 0.021,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R023
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.495,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.821,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.989,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.297,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.192,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.443,
       "feature": "primary color: black"
      },
      {
       "weight": 0.216,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.206,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.135,
       "feature": "bill shape: cone"
      }
     ]
    },
    {
     "weight": 0.112,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.615,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.253,
       "feature": "primary color: black"
      },
      {
       "weight": 0.118,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.077,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.076,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.065,
       "feature": "wing color: black"
      },
      {
       "weight": 0.063,
       "feature": "bill color: black"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.554,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.255,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.217,
       "feature": "wing color: black"
      },
      {
       "weight": 0.207,
       "feature": "breast color: white"
      },
      {
       "weight": 0.132,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.11,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.079,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.35,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.37,
       "feature": "bill color: black"
      },
      {
       "weight": 0.336,
       "feature": "back color: brown"
      },
      {
       "weight": 0.187,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.107,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.453,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.505,
       "feature": "upper tail color: brown"
      },
      {
       "weight": 0.211,
       "feature": "throat color: white"
      },
      {
       "weight": 0.154,
       "feature": "belly color: white"
      },
      {
       "weight": 0.129,
       "feature": "upperparts color: brown"
      }
     ]
    },
    {
     "weight": 0.057,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.419,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.059,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.049,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.045,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.041,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.038,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.036,
       "feature": "NOT nape color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.179,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.27,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.297,
     "shared_ref": "N3"
    },
    {
     "weight": 0.112,
     "shared_ref": "N4"
    },
    {
     "weight": 0.098,
     "shared_ref": "N5"
    },
    {
     "weight": 0.095,
     "shared_ref": "N6"
    },
    {
     "weight": 0.08,
     "shared_ref": "N7"
    },
    {
     "weight": 0.057,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R094
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 0.978,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.502,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.215,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.156,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.356,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.378,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.218,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.129,
       "feature": "back color: black"
      },
      {
       "weight": 0.096,
       "feature": "primary color: black"
      },
      {
       "weight": 0.062,
       "feature": "bill color: black"
      },
      {
       "weight": 0.06,
       "feature": "wing color: black"
      }
     ]
    },
    {
     "weight": 0.151,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.369,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.445,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.108,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.088,
       "feature": "nape color: black"
      },
      {
       "weight": 0.064,
       "feature": "back color: black"
      },
      {
       "weight": 0.05,
       "feature": "leg color: black"
      },
      {
       "weight": 0.048,
       "feature": "primary color: black"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.101,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.577,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.243,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.115,
       "feature": "nape color: black"
      },
      {
       "weight": 0.065,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.459,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.229,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.211,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.201,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.103,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.095,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.088,
       "feature": "NOT wing pattern: spotted"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.196,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.276,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.247,
       "feature": "crown color: black"
      },
      {
       "weight": 0.185,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.154,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.137,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.331,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.44,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.301,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.259,
       "feature": "NOT underparts color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.498,
   "id": "N9",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.242,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.156,
     "shared_ref": "N3"
    },
    {
     "weight": 0.151,
     "shared_ref": "N4"
    },
    {
     "weight": 0.104,
     "shared_ref": "N5"
    },
    {
     "weight": 0.082,
     "shared_ref": "N6"
    },
    {
     "weight": 0.08,
     "shared_ref": "N7"
    },
    {
     "weight": 0.076,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R180
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.115,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.642,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.099,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.52,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.932,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.196,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.178,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.125,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.123,
       "feature": "primary color: black"
      },
      {
       "weight": 0.119,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.08,
       "feature": "upper tail color: buff"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N4",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.271,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.521,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.474,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.005,
       "feature": "NOT nape color: yellow"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N5",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.789,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.24,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.218,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.15,
       "feature": "primary color: black"
      },
      {
       "weight": 0.098,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.078,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.076,
       "feature": "nape color: buff"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.162,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.412,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.375,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.213,
       "feature": "underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.061,
     "id": "N7",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.135,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.549,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.432,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.019,
       "feature": "primary color: buff"
      }
     ]
    },
    {
     "weight": 0.059,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.094,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.923,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.041,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.036,
       "feature": "NOT underparts color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.358,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.297,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.52,
     "shared_ref": "N3"
    },
    {
     "weight": 0.097,
     "shared_ref": "N4"
    },
    {
     "weight": 0.093,
     "shared_ref": "N5"
    },
    {
     "weight": 0.092,
     "shared_ref": "N6"
    },
    {
     "weight": 0.061,
     "shared_ref": "N7"
    },
    {
     "weight": 0.059,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R059
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 1.008,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.576,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.981,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.208,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.357,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.247,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.231,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.197,
       "feature": "primary color: white"
      },
      {
       "weight": 0.12,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.067,
       "feature": "breast color: white"
      },
      {
       "weight": 0.035,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.18,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.122,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.588,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.305,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.107,
       "feature": "wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.158,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.26,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.428,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.341,
       "feature": "primary color: white"
      },
      {
       "weight": 0.08,
       "feature": "throat color: white"
      },
      {
       "weight": 0.04,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.028,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.028,
       "feature": "back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.34,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.177,
       "feature": "breast color: white"
      },
      {
       "weight": 0.121,
       "feature": "throat color: white"
      },
      {
       "weight": 0.11,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.091,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.089,
       "feature": "belly color: white"
      },
      {
       "weight": 0.084,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.066,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.312,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.393,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.334,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.273,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.064,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.142,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.285,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.272,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.264,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.178,
       "feature": "NOT wing pattern: spotted"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.424,
   "id": "N9",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.195,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.208,
     "shared_ref": "N3"
    },
    {
     "weight": 0.18,
     "shared_ref": "N4"
    },
    {
     "weight": 0.158,
     "shared_ref": "N5"
    },
    {
     "weight": 0.107,
     "shared_ref": "N6"
    },
    {
     "weight": 0.066,
     "shared_ref": "N7"
    },
    {
     "weight": 0.064,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R036
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.072,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.777,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.261,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.98,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.254,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.237,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.166,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.15,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.122,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.103,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.086,
       "feature": "leg color: black"
      }
     ]
    },
    {
     "weight": 0.006,
     "id": "N4",
     "operator": "SD-",
     "name": "low soft disjunction",
     "andness": 0.393,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.263,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.184,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.166,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.135,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.096,
       "feature": "leg color: black"
      },
      {
       "weight": 0.089,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.005,
     "id": "N5",
     "operator": "SD-",
     "name": "low soft disjunction",
     "andness": 0.445,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.983,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.017,
       "feature": "NOT underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.001,
     "id": "N6",
     "operator": "SD+",
     "name": "high soft disjunction",
     "andness": 0.258,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.356,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.083,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.082,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.081,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.079,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.072,
       "feature": "forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.001,
     "id": "N7",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.681,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.274,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.119,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.118,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.112,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.111,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.1,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.001,
     "id": "N8",
     "operator": "SD+",
     "name": "high soft disjunction",
     "andness": 0.264,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.181,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.15,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.109,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.107,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.107,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.1,
       "feature": "belly color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.223,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.753,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.98,
     "shared_ref": "N3"
    },
    {
     "weight": 0.006,
     "shared_ref": "N4"
    },
    {
     "weight": 0.005,
     "shared_ref": "N5"
    },
    {
     "weight": 0.001,
     "shared_ref": "N6"
    },
    {
     "weight": 0.001,
     "shared_ref": "N7"
    },
    {
     "weight": 0.001,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R143
```json
{
 "id": "N1",
 "operator": "HC-",
 "name": "low hard conjunction",
 "andness": 0.802,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.886,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.862,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.177,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.255,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.373,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.346,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.281,
       "feature": "wing pattern: solid"
      }
     ]
    },
    {
     "weight": 0.133,
     "id": "N4",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.016,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.502,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.498,
       "feature": "NOT leg color: black"
      }
     ]
    },
    {
     "weight": 0.123,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.448,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.075,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.07,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.065,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.062,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.061,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.058,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.119,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.484,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.05,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.047,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.047,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.046,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.044,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.043,
       "feature": "NOT under tail color: brown"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.18,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.452,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.341,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.208,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.302,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.225,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.22,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.155,
       "feature": "back color: buff"
      },
      {
       "weight": 0.143,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.129,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.129,
       "feature": "back color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.114,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.86,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.177,
     "shared_ref": "N3"
    },
    {
     "weight": 0.133,
     "shared_ref": "N4"
    },
    {
     "weight": 0.123,
     "shared_ref": "N5"
    },
    {
     "weight": 0.119,
     "shared_ref": "N6"
    },
    {
     "weight": 0.104,
     "shared_ref": "N7"
    },
    {
     "weight": 0.09,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R158
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.192,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.763,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.27,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.615,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.399,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.193,
       "feature": "back color: white"
      },
      {
       "weight": 0.182,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.148,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.119,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.114,
       "feature": "crown color: black"
      },
      {
       "weight": 0.073,
       "feature": "under tail color: white"
      }
     ]
    },
    {
     "weight": 0.184,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.208,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.216,
       "feature": "nape color: white"
      },
      {
       "weight": 0.197,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.189,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.124,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.119,
       "feature": "crown color: black"
      },
      {
       "weight": 0.059,
       "feature": "back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.054,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.257,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.495,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.26,
       "feature": "throat color: white"
      },
      {
       "weight": 0.171,
       "feature": "primary color: white"
      },
      {
       "weight": 0.074,
       "feature": "belly color: white"
      }
     ]
    },
    {
     "weight": 0.05,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.31,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.511,
       "feature": "wing color: white"
      },
      {
       "weight": 0.319,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.16,
       "feature": "belly color: white"
      },
      {
       "weight": 0.01,
       "feature": "NOT breast color: black"
      }
     ]
    },
    {
     "weight": 0.019,
     "id": "N7",
     "operator": "SC-",
     "name": "low soft conjunction",
     "andness": 0.576,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.331,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.257,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.19,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.171,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.007,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.007,
       "feature": "NOT shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.012,
     "id": "N8",
     "operator": "A",
     "name": "arithmetic mean",
     "andness": 0.483,
     "verbalization": "nice to have",
     "children": [
      {
       "weight": 0.285,
       "feature": "back color: white"
      },
      {
       "weight": 0.279,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.251,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.108,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.076,
       "feature": "primary color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.237,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.812,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.615,
     "shared_ref": "N3"
    },
    {
     "weight": 0.184,
     "shared_ref": "N4"
    },
    {
     "weight": 0.054,
     "shared_ref": "N5"
    },
    {
     "weight": 0.05,
     "shared_ref": "N6"
    },
    {
     "weight": 0.019,
     "shared_ref": "N7"
    },
    {
     "weight": 0.012,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R164
```json
{
 "id": "N1",
 "operator": "DP",
 "name": "product t-conorm",
 "andness": -0.241,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.718,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.26,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.605,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.198,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.101,
       "feature": "belly color: black"
      },
      {
       "weight": 0.091,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.075,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.074,
       "feature": "crown color: black"
      },
      {
       "weight": 0.067,
       "feature": "nape color: black"
      },
      {
       "weight": 0.062,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.138,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.225,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.103,
       "feature": "belly color: black"
      },
      {
       "weight": 0.093,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.076,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.076,
       "feature": "crown color: black"
      },
      {
       "weight": 0.068,
       "feature": "nape color: black"
      },
      {
       "weight": 0.063,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.039,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.325,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.279,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.203,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.149,
       "feature": "nape color: white"
      },
      {
       "weight": 0.109,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.101,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.094,
       "feature": "breast color: grey"
      }
     ]
    },
    {
     "weight": 0.034,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.314,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.284,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.216,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.115,
       "feature": "nape color: white"
      },
      {
       "weight": 0.085,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.083,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.078,
       "feature": "size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.029,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.294,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.437,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.241,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.127,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.117,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.077,
       "feature": "head pattern: plain"
      }
     ]
    },
    {
     "weight": 0.019,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.667,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.092,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.085,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.083,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.082,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.079,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.071,
       "feature": "upper tail color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.282,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.232,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.605,
     "shared_ref": "N3"
    },
    {
     "weight": 0.138,
     "shared_ref": "N4"
    },
    {
     "weight": 0.039,
     "shared_ref": "N5"
    },
    {
     "weight": 0.034,
     "shared_ref": "N6"
    },
    {
     "weight": 0.029,
     "shared_ref": "N7"
    },
    {
     "weight": 0.019,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R022
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.948,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.621,
   "id": "N2",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.507,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.103,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.565,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.072,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.061,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.06,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.056,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.053,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.053,
       "feature": "NOT wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.453,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.095,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.079,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.071,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.067,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.063,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.063,
       "feature": "NOT shape: duck-like"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.421,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.089,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.073,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.061,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.054,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.049,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.048,
       "feature": "NOT wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.619,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.07,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.063,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.053,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.052,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.042,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.037,
       "feature": "NOT bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.449,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.052,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.04,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.038,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.037,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.036,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.036,
       "feature": "NOT throat color: buff"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.083,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.135,
       "feature": "NOT upper tail color: black"
      },
      {
       "weight": 0.112,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.101,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.099,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.09,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.089,
       "feature": "back color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.379,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.465,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.103,
     "shared_ref": "N3"
    },
    {
     "weight": 0.102,
     "shared_ref": "N4"
    },
    {
     "weight": 0.102,
     "shared_ref": "N5"
    },
    {
     "weight": 0.1,
     "shared_ref": "N6"
    },
    {
     "weight": 0.095,
     "shared_ref": "N7"
    },
    {
     "weight": 0.086,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R172
```json
{
 "id": "N1",
 "operator": "SC+",
 "name": "high soft conjunction",
 "andness": 0.749,
 "verbalization": "nice to have most",
 "children": [
  {
   "weight": 0.739,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.751,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.133,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.405,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.14,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.101,
       "feature": "crown color: black"
      },
      {
       "weight": 0.096,
       "feature": "back color: white"
      },
      {
       "weight": 0.095,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.085,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.079,
       "feature": "upperparts color: white"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.379,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.145,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.123,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.113,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.092,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.081,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.08,
       "feature": "NOT crown color: brown"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.486,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.043,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.039,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.035,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.033,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.033,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.032,
       "feature": "NOT belly color: brown"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.444,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.062,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.059,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.058,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.05,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.045,
       "feature": "NOT bill color: grey"
      },
      {
       "weight": 0.043,
       "feature": "upper tail color: buff"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.364,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.069,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.058,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.055,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.043,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.041,
       "feature": "NOT bill color: grey"
      },
      {
       "weight": 0.041,
       "feature": "back color: yellow"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.051,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.142,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.114,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.113,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.098,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.095,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.092,
       "feature": "upper tail color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.261,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.041,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.133,
     "shared_ref": "N3"
    },
    {
     "weight": 0.114,
     "shared_ref": "N4"
    },
    {
     "weight": 0.111,
     "shared_ref": "N5"
    },
    {
     "weight": 0.11,
     "shared_ref": "N6"
    },
    {
     "weight": 0.109,
     "shared_ref": "N7"
    },
    {
     "weight": 0.092,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R005
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.048,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.706,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.943,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.129,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.715,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.043,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.041,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.036,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.035,
       "feature": "belly color: black"
      },
      {
       "weight": 0.034,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.034,
       "feature": "NOT breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.129,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.102,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.901,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.012,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.01,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.01,
       "feature": "belly color: black"
      },
      {
       "weight": 0.01,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.009,
       "feature": "underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.126,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.743,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.046,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.042,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.038,
       "feature": "NOT leg color: grey"
      },
      {
       "weight": 0.038,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.038,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.038,
       "feature": "bill color: buff"
      }
     ]
    },
    {
     "weight": 0.125,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.21,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.576,
       "feature": "back color: white"
      },
      {
       "weight": 0.369,
       "feature": "wing color: white"
      },
      {
       "weight": 0.005,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.005,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.005,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.004,
       "feature": "primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.106,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.131,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.34,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.244,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.162,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.147,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.107,
       "feature": "NOT underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.075,
     "id": "N8",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.176,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.555,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.445,
       "feature": "forehead color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.294,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.402,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.129,
     "shared_ref": "N3"
    },
    {
     "weight": 0.129,
     "shared_ref": "N4"
    },
    {
     "weight": 0.126,
     "shared_ref": "N5"
    },
    {
     "weight": 0.125,
     "shared_ref": "N6"
    },
    {
     "weight": 0.106,
     "shared_ref": "N7"
    },
    {
     "weight": 0.075,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R147
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.839,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.683,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.874,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.145,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.464,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.154,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.11,
       "feature": "nape color: black"
      },
      {
       "weight": 0.091,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.086,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.067,
       "feature": "back color: white"
      },
      {
       "weight": 0.064,
       "feature": "wing color: white"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.482,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.048,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.048,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.046,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.044,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.042,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.038,
       "feature": "NOT back color: yellow"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.486,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.051,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.05,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.048,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.047,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.045,
       "feature": "NOT bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.319,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.078,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.076,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.075,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.074,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.068,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.067,
       "feature": "NOT belly color: buff"
      }
     ]
    },
    {
     "weight": 0.102,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.436,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.051,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.048,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.047,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.047,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.046,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.046,
       "feature": "NOT tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.525,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.038,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.037,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.037,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.036,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.036,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.036,
       "feature": "NOT forehead color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.317,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.088,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.145,
     "shared_ref": "N3"
    },
    {
     "weight": 0.103,
     "shared_ref": "N4"
    },
    {
     "weight": 0.103,
     "shared_ref": "N5"
    },
    {
     "weight": 0.103,
     "shared_ref": "N6"
    },
    {
     "weight": 0.102,
     "shared_ref": "N7"
    },
    {
     "weight": 0.101,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R050
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.105,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.861,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.751,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.38,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.121,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.311,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.217,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.071,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.07,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.054,
       "feature": "crown color: black"
      },
      {
       "weight": 0.036,
       "feature": "bill color: black"
      }
     ]
    },
    {
     "weight": 0.334,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.295,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.399,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.091,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.089,
       "feature": "size: medium (9 - 16 in)"
      },
      {
       "weight": 0.088,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.059,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.051,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.228,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.474,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.21,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.164,
       "feature": "crown color: black"
      },
      {
       "weight": 0.142,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.121,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.111,
       "feature": "bill color: black"
      },
      {
       "weight": 0.11,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.015,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.626,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.341,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.112,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.109,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.079,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.067,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.063,
       "feature": "NOT back color: white"
      }
     ]
    },
    {
     "weight": 0.009,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.723,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.091,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.065,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.065,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.057,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.056,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.054,
       "feature": "NOT leg color: black"
      }
     ]
    },
    {
     "weight": 0.006,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.774,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.614,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.158,
       "feature": "eye color: black"
      },
      {
       "weight": 0.108,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.081,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.038,
       "feature": "NOT bill color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.139,
   "id": "N9",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.247,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.38,
     "shared_ref": "N3"
    },
    {
     "weight": 0.334,
     "shared_ref": "N4"
    },
    {
     "weight": 0.228,
     "shared_ref": "N5"
    },
    {
     "weight": 0.015,
     "shared_ref": "N6"
    },
    {
     "weight": 0.009,
     "shared_ref": "N7"
    },
    {
     "weight": 0.006,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R038
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.093,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.554,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.089,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.228,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.39,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.378,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.146,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.085,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.076,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.059,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.049,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.203,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.335,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.379,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.147,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.086,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.076,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.063,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.051,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N5",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.213,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.558,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.302,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.095,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.019,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.007,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.006,
       "feature": "NOT underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.072,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.435,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.349,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.232,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.228,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.191,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.071,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.4,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.281,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.264,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.24,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.215,
       "feature": "NOT bill shape: dagger"
      }
     ]
    },
    {
     "weight": 0.059,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.478,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.081,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.076,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.076,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.075,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.065,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.062,
       "feature": "NOT forehead color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.446,
   "id": "N9",
   "operator": "D",
   "name": "pure disjunction",
   "andness": -0.002,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.228,
     "shared_ref": "N3"
    },
    {
     "weight": 0.203,
     "shared_ref": "N4"
    },
    {
     "weight": 0.086,
     "shared_ref": "N5"
    },
    {
     "weight": 0.072,
     "shared_ref": "N6"
    },
    {
     "weight": 0.071,
     "shared_ref": "N7"
    },
    {
     "weight": 0.059,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R107
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.97,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.715,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.077,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.216,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.797,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.341,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.171,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.125,
       "feature": "breast color: white"
      },
      {
       "weight": 0.091,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.081,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.071,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.196,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.394,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.197,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.182,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.094,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.022,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.014,
       "feature": "NOT bill color: black"
      }
     ]
    },
    {
     "weight": 0.165,
     "id": "N5",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.738,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.428,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.306,
       "feature": "wing color: white"
      },
      {
       "weight": 0.267,
       "feature": "belly color: brown"
      }
     ]
    },
    {
     "weight": 0.164,
     "id": "N6",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.788,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.245,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.193,
       "feature": "breast color: white"
      },
      {
       "weight": 0.155,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.141,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.101,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.06,
       "feature": "NOT breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.067,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.036,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.24,
       "feature": "belly color: white"
      },
      {
       "weight": 0.129,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.099,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.093,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.042,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.04,
       "feature": "NOT bill color: black"
      }
     ]
    },
    {
     "weight": 0.042,
     "id": "N8",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.983,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.176,
       "feature": "throat color: white"
      },
      {
       "weight": 0.16,
       "feature": "belly color: white"
      },
      {
       "weight": 0.158,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.128,
       "feature": "back color: brown"
      },
      {
       "weight": 0.101,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.058,
       "feature": "NOT back pattern: striped"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.285,
   "id": "N9",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.167,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.216,
     "shared_ref": "N3"
    },
    {
     "weight": 0.196,
     "shared_ref": "N4"
    },
    {
     "weight": 0.165,
     "shared_ref": "N5"
    },
    {
     "weight": 0.164,
     "shared_ref": "N6"
    },
    {
     "weight": 0.067,
     "shared_ref": "N7"
    },
    {
     "weight": 0.042,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R189
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.225,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.517,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.116,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.183,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.412,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.373,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.176,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.123,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.068,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.056,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.055,
       "feature": "breast color: brown"
      }
     ]
    },
    {
     "weight": 0.152,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.275,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.342,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.238,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.133,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.109,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.079,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.025,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.125,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.223,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.763,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.095,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.041,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.038,
       "feature": "bill color: black"
      },
      {
       "weight": 0.036,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.026,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.074,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.291,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.21,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.143,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.079,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.079,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.076,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.074,
       "feature": "NOT crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.459,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.268,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.14,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.098,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.096,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.09,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.086,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.07,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.163,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.296,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.281,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.254,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.168,
       "feature": "NOT leg color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.483,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.212,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.183,
     "shared_ref": "N3"
    },
    {
     "weight": 0.152,
     "shared_ref": "N4"
    },
    {
     "weight": 0.125,
     "shared_ref": "N5"
    },
    {
     "weight": 0.074,
     "shared_ref": "N6"
    },
    {
     "weight": 0.073,
     "shared_ref": "N7"
    },
    {
     "weight": 0.07,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R079
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.353,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.802,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.204,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.126,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.333,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.271,
       "feature": "nape color: black"
      },
      {
       "weight": 0.269,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.221,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.141,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.065,
       "feature": "throat color: white"
      },
      {
       "weight": 0.028,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.121,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.313,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.426,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.212,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.165,
       "feature": "belly color: white"
      },
      {
       "weight": 0.104,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.074,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.009,
       "feature": "NOT under tail color: black"
      }
     ]
    },
    {
     "weight": 0.119,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.188,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.319,
       "feature": "nape color: black"
      },
      {
       "weight": 0.26,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.214,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.077,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.064,
       "feature": "belly color: white"
      },
      {
       "weight": 0.041,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.108,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.174,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.172,
       "feature": "throat color: white"
      },
      {
       "weight": 0.142,
       "feature": "wing color: black"
      },
      {
       "weight": 0.119,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.089,
       "feature": "bill color: black"
      },
      {
       "weight": 0.065,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.199,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.428,
       "feature": "primary color: white"
      },
      {
       "weight": 0.172,
       "feature": "breast color: white"
      },
      {
       "weight": 0.124,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.081,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.077,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.022,
       "feature": "NOT back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N8",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.995,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.413,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.35,
       "feature": "primary color: white"
      },
      {
       "weight": 0.01,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.01,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.009,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.008,
       "feature": "NOT primary color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.198,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.872,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.126,
     "shared_ref": "N3"
    },
    {
     "weight": 0.121,
     "shared_ref": "N4"
    },
    {
     "weight": 0.119,
     "shared_ref": "N5"
    },
    {
     "weight": 0.114,
     "shared_ref": "N6"
    },
    {
     "weight": 0.104,
     "shared_ref": "N7"
    },
    {
     "weight": 0.1,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R132
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.928,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.786,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.064,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.538,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.764,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.188,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.179,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.09,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.084,
       "feature": "breast color: white"
      },
      {
       "weight": 0.08,
       "feature": "belly color: white"
      },
      {
       "weight": 0.068,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.219,
     "id": "N4",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.824,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.321,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.167,
       "feature": "wing color: black"
      },
      {
       "weight": 0.143,
       "feature": "breast color: white"
      },
      {
       "weight": 0.099,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.097,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.047,
       "feature": "NOT upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.055,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.237,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.107,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.078,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.074,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.063,
       "feature": "NOT primary color: white"
      },
      {
       "weight": 0.058,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.056,
       "feature": "NOT wing color: grey"
      }
     ]
    },
    {
     "weight": 0.042,
     "id": "N6",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.911,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.226,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.123,
       "feature": "wing color: black"
      },
      {
       "weight": 0.113,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.086,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.081,
       "feature": "bill color: black"
      },
      {
       "weight": 0.073,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.029,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.611,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.046,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.041,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.04,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.039,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.039,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.039,
       "feature": "NOT underparts color: brown"
      }
     ]
    },
    {
     "weight": 0.029,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.42,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.085,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.05,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.045,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.043,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.035,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.032,
       "feature": "NOT wing pattern: spotted"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.214,
   "id": "N9",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.111,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.538,
     "shared_ref": "N3"
    },
    {
     "weight": 0.219,
     "shared_ref": "N4"
    },
    {
     "weight": 0.055,
     "shared_ref": "N5"
    },
    {
     "weight": 0.042,
     "shared_ref": "N6"
    },
    {
     "weight": 0.029,
     "shared_ref": "N7"
    },
    {
     "weight": 0.029,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R068
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.971,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.767,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.048,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.511,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.942,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.24,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.167,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.136,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.099,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.09,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.081,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.063,
     "id": "N4",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.002,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.269,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.095,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.068,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.064,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.064,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.055,
       "feature": "NOT primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.051,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.26,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.164,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.097,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.079,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.076,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.063,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.058,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.048,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.441,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.17,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.091,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.067,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.059,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.057,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.052,
       "feature": "NOT upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.043,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.521,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.283,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.089,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.073,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.065,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.062,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.05,
       "feature": "NOT throat color: grey"
      }
     ]
    },
    {
     "weight": 0.043,
     "id": "N8",
     "operator": "SC-",
     "name": "low soft conjunction",
     "andness": 0.599,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.765,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.235,
       "feature": "head pattern: eyebrow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.233,
   "id": "N9",
   "operator": "D",
   "name": "pure disjunction",
   "andness": -0.015,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.511,
     "shared_ref": "N3"
    },
    {
     "weight": 0.063,
     "shared_ref": "N4"
    },
    {
     "weight": 0.051,
     "shared_ref": "N5"
    },
    {
     "weight": 0.048,
     "shared_ref": "N6"
    },
    {
     "weight": 0.043,
     "shared_ref": "N7"
    },
    {
     "weight": 0.043,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R015
```json
{
 "id": "N1",
 "operator": "DP",
 "name": "product t-conorm",
 "andness": -0.303,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.672,
   "id": "N2",
   "operator": "HHD",
   "name": "high hyper-disjunction",
   "andness": -0.407,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.946,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.08,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.236,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.171,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.144,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.112,
       "feature": "throat color: white"
      },
      {
       "weight": 0.111,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.063,
       "feature": "breast color: white"
      }
     ]
    },
    {
     "weight": 0.018,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.525,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.399,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.225,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.147,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.101,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.045,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.037,
       "feature": "NOT upperparts color: black"
      }
     ]
    },
    {
     "weight": 0.016,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.204,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.331,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.157,
       "feature": "throat color: white"
      },
      {
       "weight": 0.156,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.088,
       "feature": "breast color: white"
      },
      {
       "weight": 0.073,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.053,
       "feature": "underparts color: white"
      }
     ]
    },
    {
     "weight": 0.007,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.528,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.28,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.237,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.176,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.032,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.027,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.027,
       "feature": "NOT wing color: black"
      }
     ]
    },
    {
     "weight": 0.004,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.445,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.224,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.17,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.144,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.134,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.111,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.102,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.001,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.554,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.18,
       "feature": "belly color: white"
      },
      {
       "weight": 0.151,
       "feature": "back color: brown"
      },
      {
       "weight": 0.124,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.12,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.075,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.073,
       "feature": "shape: perching-like"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.328,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.845,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.946,
     "shared_ref": "N3"
    },
    {
     "weight": 0.018,
     "shared_ref": "N4"
    },
    {
     "weight": 0.016,
     "shared_ref": "N5"
    },
    {
     "weight": 0.007,
     "shared_ref": "N6"
    },
    {
     "weight": 0.004,
     "shared_ref": "N7"
    },
    {
     "weight": 0.001,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R179
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.686,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.802,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.256,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.215,
     "id": "N3",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.877,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.426,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.219,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.099,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.096,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.084,
       "feature": "back color: grey"
      },
      {
       "weight": 0.076,
       "feature": "belly color: white"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.789,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.245,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.158,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.13,
       "feature": "breast color: white"
      },
      {
       "weight": 0.127,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.121,
       "feature": "belly color: white"
      },
      {
       "weight": 0.116,
       "feature": "underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N5",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.758,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.329,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.183,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.171,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.163,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.155,
       "feature": "underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.365,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.056,
       "feature": "crown color: white"
      },
      {
       "weight": 0.053,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.051,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.043,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.038,
       "feature": "wing color: white"
      },
      {
       "weight": 0.036,
       "feature": "NOT size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.452,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.048,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.046,
       "feature": "crown color: white"
      },
      {
       "weight": 0.036,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.036,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.035,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.033,
       "feature": "NOT nape color: brown"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.195,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.092,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.075,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.071,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.071,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.07,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.07,
       "feature": "back color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.198,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.227,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.215,
     "shared_ref": "N3"
    },
    {
     "weight": 0.109,
     "shared_ref": "N4"
    },
    {
     "weight": 0.108,
     "shared_ref": "N5"
    },
    {
     "weight": 0.092,
     "shared_ref": "N6"
    },
    {
     "weight": 0.092,
     "shared_ref": "N7"
    },
    {
     "weight": 0.089,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R141
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.098,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.722,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.998,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.132,
     "id": "N3",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.24,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.294,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.256,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.234,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.217,
       "feature": "NOT head pattern: plain"
      }
     ]
    },
    {
     "weight": 0.122,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.619,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.074,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.054,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.049,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.048,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.047,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "NOT belly color: grey"
      }
     ]
    },
    {
     "weight": 0.12,
     "id": "N5",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.024,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.433,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.273,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.241,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.021,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.017,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.015,
       "feature": "NOT wing shape: pointed-wings"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.724,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.083,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.046,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.037,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.034,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.031,
       "feature": "NOT underparts color: grey"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.211,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.453,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.427,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.121,
       "feature": "NOT breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.668,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.054,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.04,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.04,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.038,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.037,
       "feature": "NOT belly color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.278,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.35,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.132,
     "shared_ref": "N3"
    },
    {
     "weight": 0.122,
     "shared_ref": "N4"
    },
    {
     "weight": 0.12,
     "shared_ref": "N5"
    },
    {
     "weight": 0.117,
     "shared_ref": "N6"
    },
    {
     "weight": 0.115,
     "shared_ref": "N7"
    },
    {
     "weight": 0.105,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R167
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.666,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.74,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.051,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.151,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.443,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.221,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.164,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.095,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.079,
       "feature": "belly color: white"
      },
      {
       "weight": 0.078,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.063,
       "feature": "primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.147,
     "id": "N4",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.201,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.478,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.32,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.116,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.051,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.034,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.129,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.129,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.055,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.054,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.049,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.049,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.048,
       "feature": "wing pattern: spotted"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.111,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.527,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.23,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.227,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.005,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.004,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.004,
       "feature": "NOT crown color: white"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.354,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.566,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.379,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.005,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.003,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.003,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.002,
       "feature": "forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N8",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.077,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.591,
       "feature": "belly color: white"
      },
      {
       "weight": 0.322,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.086,
       "feature": "shape: perching-like"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.26,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.999,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.151,
     "shared_ref": "N3"
    },
    {
     "weight": 0.147,
     "shared_ref": "N4"
    },
    {
     "weight": 0.129,
     "shared_ref": "N5"
    },
    {
     "weight": 0.117,
     "shared_ref": "N6"
    },
    {
     "weight": 0.101,
     "shared_ref": "N7"
    },
    {
     "weight": 0.084,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R128
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.044,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.837,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.048,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.883,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.053,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.167,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.151,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.126,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.076,
       "feature": "wing color: white"
      },
      {
       "weight": 0.06,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.053,
       "feature": "wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.015,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.748,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.072,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.052,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.048,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.045,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.042,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.041,
       "feature": "bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.015,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.774,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.199,
       "feature": "back color: white"
      },
      {
       "weight": 0.109,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.085,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.062,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.05,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.046,
       "feature": "primary color: buff"
      }
     ]
    },
    {
     "weight": 0.015,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.746,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.107,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.085,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.07,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.068,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.068,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.067,
       "feature": "throat color: grey"
      }
     ]
    },
    {
     "weight": 0.014,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.779,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.099,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.061,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.06,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.053,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.046,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.04,
       "feature": "primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.014,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.797,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.157,
       "feature": "back color: white"
      },
      {
       "weight": 0.101,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.082,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.061,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.06,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.058,
       "feature": "crown color: blue"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.163,
   "id": "N9",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.22,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.883,
     "shared_ref": "N3"
    },
    {
     "weight": 0.015,
     "shared_ref": "N4"
    },
    {
     "weight": 0.015,
     "shared_ref": "N5"
    },
    {
     "weight": 0.015,
     "shared_ref": "N6"
    },
    {
     "weight": 0.014,
     "shared_ref": "N7"
    },
    {
     "weight": 0.014,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R182
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.425,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.586,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.133,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.128,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.231,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.166,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.14,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.123,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.118,
       "feature": "back color: white"
      },
      {
       "weight": 0.089,
       "feature": "breast color: black"
      },
      {
       "weight": 0.051,
       "feature": "back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N4",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.036,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.541,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.187,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.162,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.11,
       "feature": "NOT breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N5",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.057,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.532,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.381,
       "feature": "back color: white"
      },
      {
       "weight": 0.069,
       "feature": "crown color: white"
      },
      {
       "weight": 0.004,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.004,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.003,
       "feature": "NOT size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.036,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.179,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.157,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.114,
       "feature": "breast color: black"
      },
      {
       "weight": 0.065,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.062,
       "feature": "nape color: white"
      },
      {
       "weight": 0.057,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.336,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 1.0,
       "feature": "NOT bill color: grey"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N8",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.001,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.248,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.204,
       "feature": "belly color: black"
      },
      {
       "weight": 0.118,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.097,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.087,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.075,
       "feature": "wing color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.414,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.248,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.128,
     "shared_ref": "N3"
    },
    {
     "weight": 0.096,
     "shared_ref": "N4"
    },
    {
     "weight": 0.095,
     "shared_ref": "N5"
    },
    {
     "weight": 0.091,
     "shared_ref": "N6"
    },
    {
     "weight": 0.086,
     "shared_ref": "N7"
    },
    {
     "weight": 0.081,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R096
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 0.997,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.662,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.109,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.188,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.372,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.235,
       "feature": "primary color: white"
      },
      {
       "weight": 0.158,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.131,
       "feature": "back color: grey"
      },
      {
       "weight": 0.107,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.075,
       "feature": "primary color: black"
      },
      {
       "weight": 0.07,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.168,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.219,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.309,
       "feature": "throat color: black"
      },
      {
       "weight": 0.218,
       "feature": "primary color: white"
      },
      {
       "weight": 0.146,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.121,
       "feature": "back color: grey"
      },
      {
       "weight": 0.108,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.099,
       "feature": "primary color: grey"
      }
     ]
    },
    {
     "weight": 0.109,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.264,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.537,
       "feature": "throat color: black"
      },
      {
       "weight": 0.187,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.113,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.094,
       "feature": "breast color: white"
      },
      {
       "weight": 0.043,
       "feature": "belly color: white"
      },
      {
       "weight": 0.026,
       "feature": "bill color: black"
      }
     ]
    },
    {
     "weight": 0.103,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.215,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.404,
       "feature": "primary color: black"
      },
      {
       "weight": 0.17,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.163,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.147,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.115,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.06,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.474,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.238,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.209,
       "feature": "NOT upper tail color: black"
      },
      {
       "weight": 0.131,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.12,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.082,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.08,
       "feature": "NOT breast color: brown"
      }
     ]
    },
    {
     "weight": 0.057,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.524,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.275,
       "feature": "NOT belly color: black"
      },
      {
       "weight": 0.12,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.087,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.085,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.078,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.078,
       "feature": "NOT nape color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.338,
   "id": "N9",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.035,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.188,
     "shared_ref": "N3"
    },
    {
     "weight": 0.168,
     "shared_ref": "N4"
    },
    {
     "weight": 0.109,
     "shared_ref": "N5"
    },
    {
     "weight": 0.103,
     "shared_ref": "N6"
    },
    {
     "weight": 0.06,
     "shared_ref": "N7"
    },
    {
     "weight": 0.057,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R090
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.922,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.587,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.901,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.095,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.697,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.188,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.062,
       "feature": "NOT bill length: about the same as head"
      },
      {
       "weight": 0.058,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.054,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.05,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.045,
       "feature": "NOT under tail color: grey"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N4",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.287,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.269,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.196,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.164,
       "feature": "NOT bill color: grey"
      },
      {
       "weight": 0.141,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.107,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.066,
       "feature": "NOT crown color: blue"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.676,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.098,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.086,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.056,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.053,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.05,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.048,
       "feature": "NOT belly color: white"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.607,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.061,
       "feature": "NOT throat color: white"
      },
      {
       "weight": 0.06,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.057,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.055,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.054,
       "feature": "NOT wing pattern: striped"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.621,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.063,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.061,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.06,
       "feature": "NOT upper tail color: buff"
      },
      {
       "weight": 0.055,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.053,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.051,
       "feature": "NOT forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.423,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.13,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.121,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.1,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.094,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.088,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.08,
       "feature": "nape color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.413,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.535,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.095,
     "shared_ref": "N3"
    },
    {
     "weight": 0.093,
     "shared_ref": "N4"
    },
    {
     "weight": 0.093,
     "shared_ref": "N5"
    },
    {
     "weight": 0.092,
     "shared_ref": "N6"
    },
    {
     "weight": 0.089,
     "shared_ref": "N7"
    },
    {
     "weight": 0.089,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R129
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.89,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.643,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.026,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.324,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.325,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.335,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.268,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.108,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.091,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.072,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.205,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.299,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.425,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.359,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.198,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.018,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.144,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.163,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.396,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.317,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.104,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.085,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.055,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.034,
       "feature": "throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.077,
     "id": "N6",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.048,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.929,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.026,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.02,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.009,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.009,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.006,
       "feature": "NOT upperparts color: white"
      }
     ]
    },
    {
     "weight": 0.052,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.338,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.126,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.118,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.116,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.1,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.096,
       "feature": "breast color: grey"
      },
      {
       "weight": 0.088,
       "feature": "NOT bill color: buff"
      }
     ]
    },
    {
     "weight": 0.048,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.241,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.176,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.15,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.129,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.127,
       "feature": "NOT crown color: white"
      },
      {
       "weight": 0.113,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.105,
       "feature": "NOT bill color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.357,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.882,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.324,
     "shared_ref": "N3"
    },
    {
     "weight": 0.205,
     "shared_ref": "N4"
    },
    {
     "weight": 0.144,
     "shared_ref": "N5"
    },
    {
     "weight": 0.077,
     "shared_ref": "N6"
    },
    {
     "weight": 0.052,
     "shared_ref": "N7"
    },
    {
     "weight": 0.048,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R060
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.551,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.988,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.052,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.196,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.134,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.31,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.196,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.187,
       "feature": "bill color: black"
      },
      {
       "weight": 0.164,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.084,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.059,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.147,
     "id": "N4",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.994,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.735,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.208,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.056,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.001,
       "feature": "NOT belly color: black"
      }
     ]
    },
    {
     "weight": 0.147,
     "id": "N5",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.977,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.716,
       "feature": "leg color: black"
      },
      {
       "weight": 0.154,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.131,
       "feature": "NOT belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.138,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.347,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.634,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.366,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.1,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.175,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.4,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.38,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.219,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.053,
     "id": "N8",
     "operator": "A",
     "name": "arithmetic mean",
     "andness": 0.5,
     "verbalization": "nice to have",
     "children": [
      {
       "weight": 0.769,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.218,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.003,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.003,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.002,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.002,
       "feature": "NOT back pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.012,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.783,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.196,
     "shared_ref": "N3"
    },
    {
     "weight": 0.147,
     "shared_ref": "N4"
    },
    {
     "weight": 0.147,
     "shared_ref": "N5"
    },
    {
     "weight": 0.138,
     "shared_ref": "N6"
    },
    {
     "weight": 0.1,
     "shared_ref": "N7"
    },
    {
     "weight": 0.053,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R166
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.336,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.893,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.201,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.217,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.362,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.415,
       "feature": "wing color: white"
      },
      {
       "weight": 0.148,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.117,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.071,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.056,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.053,
       "feature": "NOT breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.185,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.576,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.228,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.224,
       "feature": "throat color: white"
      },
      {
       "weight": 0.197,
       "feature": "belly color: white"
      },
      {
       "weight": 0.171,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.104,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.065,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.177,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.539,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.353,
       "feature": "wing color: white"
      },
      {
       "weight": 0.133,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.131,
       "feature": "breast color: white"
      },
      {
       "weight": 0.13,
       "feature": "throat color: white"
      },
      {
       "weight": 0.126,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.115,
       "feature": "belly color: white"
      }
     ]
    },
    {
     "weight": 0.134,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.379,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.34,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.268,
       "feature": "NOT size: small (5 - 9 in)"
      },
      {
       "weight": 0.057,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.047,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.04,
       "feature": "NOT belly color: grey"
      }
     ]
    },
    {
     "weight": 0.036,
     "id": "N7",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.232,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.466,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.277,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.216,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.003,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.003,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.003,
       "feature": "NOT upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.035,
     "id": "N8",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.202,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.851,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.08,
       "feature": "breast color: white"
      },
      {
       "weight": 0.046,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.007,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.004,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.004,
       "feature": "NOT upper tail color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.107,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.905,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.217,
     "shared_ref": "N3"
    },
    {
     "weight": 0.185,
     "shared_ref": "N4"
    },
    {
     "weight": 0.177,
     "shared_ref": "N5"
    },
    {
     "weight": 0.134,
     "shared_ref": "N6"
    },
    {
     "weight": 0.036,
     "shared_ref": "N7"
    },
    {
     "weight": 0.035,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R131
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.809,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.936,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.19,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.677,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.231,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.164,
       "feature": "primary color: white"
      },
      {
       "weight": 0.12,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.1,
       "feature": "nape color: white"
      },
      {
       "weight": 0.077,
       "feature": "throat color: white"
      },
      {
       "weight": 0.07,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.047,
     "id": "N4",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.006,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.246,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.157,
       "feature": "throat color: white"
      },
      {
       "weight": 0.143,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.102,
       "feature": "breast color: white"
      },
      {
       "weight": 0.099,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.087,
       "feature": "belly color: white"
      }
     ]
    },
    {
     "weight": 0.046,
     "id": "N5",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.02,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.23,
       "feature": "nape color: black"
      },
      {
       "weight": 0.171,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.098,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.095,
       "feature": "bill color: black"
      },
      {
       "weight": 0.055,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.05,
       "feature": "NOT belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.043,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.246,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.265,
       "feature": "nape color: black"
      },
      {
       "weight": 0.113,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.11,
       "feature": "bill color: black"
      },
      {
       "weight": 0.083,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.063,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.057,
       "feature": "NOT primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.031,
     "id": "N7",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.825,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.327,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.261,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.161,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.127,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.114,
       "feature": "primary color: white"
      },
      {
       "weight": 0.005,
       "feature": "NOT forehead color: grey"
      }
     ]
    },
    {
     "weight": 0.031,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.399,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.118,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.064,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.05,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.043,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.036,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.034,
       "feature": "NOT breast color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.064,
   "id": "N9",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.044,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.677,
     "shared_ref": "N3"
    },
    {
     "weight": 0.047,
     "shared_ref": "N4"
    },
    {
     "weight": 0.046,
     "shared_ref": "N5"
    },
    {
     "weight": 0.043,
     "shared_ref": "N6"
    },
    {
     "weight": 0.031,
     "shared_ref": "N7"
    },
    {
     "weight": 0.031,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R081
```json
{
 "id": "N1",
 "operator": "HD-",
 "name": "low hard disjunction",
 "andness": 0.24,
 "verbalization": "enough to have any",
 "children": [
  {
   "weight": 0.937,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.129,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.505,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.233,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.221,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.218,
       "feature": "throat color: black"
      },
      {
       "weight": 0.167,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.113,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.1,
       "feature": "back color: grey"
      },
      {
       "weight": 0.063,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.312,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.254,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.194,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.191,
       "feature": "throat color: black"
      },
      {
       "weight": 0.151,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.146,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.099,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.088,
       "feature": "back color: grey"
      }
     ]
    },
    {
     "weight": 0.028,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.852,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.156,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.147,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.052,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.049,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.045,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.044,
       "feature": "NOT tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.027,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.688,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.116,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.112,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.097,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.096,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.085,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.08,
       "feature": "head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.026,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.778,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.123,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.11,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.087,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.086,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.085,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.053,
       "feature": "tail pattern: striped"
      }
     ]
    },
    {
     "weight": 0.025,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.746,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.53,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.074,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.062,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.039,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.032,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.022,
       "feature": "bill length: shorter than head"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.063,
   "id": "N9",
   "operator": "SC+",
   "name": "high soft conjunction",
   "andness": 0.735,
   "verbalization": "nice to have most",
   "children": [
    {
     "weight": 0.505,
     "shared_ref": "N3"
    },
    {
     "weight": 0.312,
     "shared_ref": "N4"
    },
    {
     "weight": 0.028,
     "shared_ref": "N5"
    },
    {
     "weight": 0.027,
     "shared_ref": "N6"
    },
    {
     "weight": 0.026,
     "shared_ref": "N7"
    },
    {
     "weight": 0.025,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R033
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.518,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.718,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.013,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.24,
     "id": "N3",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.94,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.351,
       "feature": "throat color: black"
      },
      {
       "weight": 0.324,
       "feature": "nape color: black"
      },
      {
       "weight": 0.2,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.125,
       "feature": "size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.132,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.455,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.28,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.257,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.145,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.123,
       "feature": "crown color: black"
      },
      {
       "weight": 0.085,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.084,
       "feature": "belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.528,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.253,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.143,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.12,
       "feature": "crown color: black"
      },
      {
       "weight": 0.083,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.082,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.073,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.248,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.41,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.19,
       "feature": "throat color: black"
      },
      {
       "weight": 0.176,
       "feature": "nape color: black"
      },
      {
       "weight": 0.091,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.078,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.056,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.159,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.083,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.083,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.08,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.079,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.077,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.077,
       "feature": "NOT primary color: brown"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.551,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.036,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.034,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.032,
       "feature": "NOT back color: black"
      },
      {
       "weight": 0.031,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.029,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.029,
       "feature": "NOT upper tail color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.282,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.093,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.24,
     "shared_ref": "N3"
    },
    {
     "weight": 0.132,
     "shared_ref": "N4"
    },
    {
     "weight": 0.115,
     "shared_ref": "N5"
    },
    {
     "weight": 0.091,
     "shared_ref": "N6"
    },
    {
     "weight": 0.076,
     "shared_ref": "N7"
    },
    {
     "weight": 0.073,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R048
```json
{
 "id": "N1",
 "operator": "C",
 "name": "pure conjunction",
 "andness": 1.004,
 "verbalization": "decided by lowest",
 "children": [
  {
   "weight": 0.633,
   "id": "N2",
   "operator": "HD",
   "name": "medium hard disjunction",
   "andness": 0.12,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.2,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.536,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.393,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.251,
       "feature": "crown color: black"
      },
      {
       "weight": 0.141,
       "feature": "bill color: black"
      },
      {
       "weight": 0.073,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.045,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.045,
       "feature": "primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.139,
     "id": "N4",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.233,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.783,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.076,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.071,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.069,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.0,
       "feature": "NOT nape color: brown"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N5",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.017,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.273,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.264,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.149,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.055,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.05,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "NOT wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N6",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.06,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.264,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.158,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.156,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.109,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.093,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.085,
       "feature": "NOT upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.075,
     "id": "N7",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.031,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.342,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.294,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.188,
       "feature": "crown color: black"
      },
      {
       "weight": 0.054,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.041,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.034,
       "feature": "belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.063,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.412,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.112,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.08,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.077,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.076,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.075,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.069,
       "feature": "size: medium (9 - 16 in)"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.367,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.432,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.2,
     "shared_ref": "N3"
    },
    {
     "weight": 0.139,
     "shared_ref": "N4"
    },
    {
     "weight": 0.116,
     "shared_ref": "N5"
    },
    {
     "weight": 0.105,
     "shared_ref": "N6"
    },
    {
     "weight": 0.075,
     "shared_ref": "N7"
    },
    {
     "weight": 0.063,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R016
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.643,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.68,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.184,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.134,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.365,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.132,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.128,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.112,
       "feature": "breast color: black"
      },
      {
       "weight": 0.073,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.07,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.056,
       "feature": "forehead color: grey"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N4",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.084,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.598,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.343,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.008,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.008,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.008,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.008,
       "feature": "NOT breast color: grey"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.337,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.573,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.222,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.205,
       "feature": "NOT throat color: grey"
      }
     ]
    },
    {
     "weight": 0.08,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.18,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.198,
       "feature": "breast color: black"
      },
      {
       "weight": 0.129,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.089,
       "feature": "back color: black"
      },
      {
       "weight": 0.085,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.069,
       "feature": "wing color: white"
      },
      {
       "weight": 0.063,
       "feature": "primary color: black"
      }
     ]
    },
    {
     "weight": 0.079,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.284,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.191,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.176,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.15,
       "feature": "NOT back color: brown"
      },
      {
       "weight": 0.142,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.129,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.119,
       "feature": "NOT under tail color: buff"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.465,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.122,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.102,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.098,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.094,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.089,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.089,
       "feature": "NOT bill color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.32,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.189,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.134,
     "shared_ref": "N3"
    },
    {
     "weight": 0.088,
     "shared_ref": "N4"
    },
    {
     "weight": 0.082,
     "shared_ref": "N5"
    },
    {
     "weight": 0.08,
     "shared_ref": "N6"
    },
    {
     "weight": 0.079,
     "shared_ref": "N7"
    },
    {
     "weight": 0.076,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R099
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.099,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.821,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.923,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.898,
     "id": "N3",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.977,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.37,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.283,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.083,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.071,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.067,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.065,
       "feature": "nape color: grey"
      }
     ]
    },
    {
     "weight": 0.036,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.266,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.357,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.276,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.112,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.065,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.062,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.036,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.014,
     "id": "N5",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.84,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.347,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.265,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.223,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.095,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.07,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.01,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.378,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.304,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.229,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.132,
       "feature": "eye color: black"
      },
      {
       "weight": 0.132,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.077,
       "feature": "NOT upperparts color: yellow"
      },
      {
       "weight": 0.07,
       "feature": "NOT tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.009,
     "id": "N7",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.911,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.752,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.225,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.022,
       "feature": "NOT leg color: grey"
      }
     ]
    },
    {
     "weight": 0.005,
     "id": "N8",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.915,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.411,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.304,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.27,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.015,
       "feature": "NOT back color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.179,
   "id": "N9",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.764,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.898,
     "shared_ref": "N3"
    },
    {
     "weight": 0.036,
     "shared_ref": "N4"
    },
    {
     "weight": 0.014,
     "shared_ref": "N5"
    },
    {
     "weight": 0.01,
     "shared_ref": "N6"
    },
    {
     "weight": 0.009,
     "shared_ref": "N7"
    },
    {
     "weight": 0.005,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R045
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.26,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.632,
   "id": "N2",
   "operator": "HHD",
   "name": "high hyper-disjunction",
   "andness": -0.391,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.981,
     "id": "N3",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.879,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.195,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.121,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.064,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.05,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.048,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.047,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.003,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.048,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.123,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.119,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.109,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.09,
       "feature": "breast color: white"
      },
      {
       "weight": 0.089,
       "feature": "wing color: black"
      },
      {
       "weight": 0.067,
       "feature": "under tail color: grey"
      }
     ]
    },
    {
     "weight": 0.003,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.529,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.131,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.086,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.068,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.066,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.059,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.058,
       "feature": "NOT leg color: grey"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.413,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.101,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.082,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.053,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.037,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.033,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.032,
       "feature": "belly color: buff"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.19,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.314,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.227,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.17,
       "feature": "eye color: black"
      },
      {
       "weight": 0.159,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.132,
       "feature": "NOT breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.488,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.144,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.067,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.036,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.035,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.029,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.029,
       "feature": "belly color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.368,
   "id": "N9",
   "operator": "HHD",
   "name": "high hyper-disjunction",
   "andness": -0.339,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.981,
     "shared_ref": "N3"
    },
    {
     "weight": 0.003,
     "shared_ref": "N4"
    },
    {
     "weight": 0.003,
     "shared_ref": "N5"
    },
    {
     "weight": 0.002,
     "shared_ref": "N6"
    },
    {
     "weight": 0.002,
     "shared_ref": "N7"
    },
    {
     "weight": 0.002,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R058
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.878,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.676,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.908,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.095,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.311,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.127,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.127,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.086,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.082,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.078,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.075,
       "feature": "upperparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.643,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.09,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.09,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.081,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.064,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.062,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.061,
       "feature": "NOT crown color: blue"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.628,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.078,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.057,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.046,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.043,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.042,
       "feature": "NOT upper tail color: black"
      },
      {
       "weight": 0.042,
       "feature": "NOT breast color: white"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.575,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.079,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.072,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.071,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.054,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.052,
       "feature": "NOT upper tail color: black"
      },
      {
       "weight": 0.052,
       "feature": "NOT forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.575,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.074,
       "feature": "NOT nape color: black"
      },
      {
       "weight": 0.067,
       "feature": "NOT under tail color: black"
      },
      {
       "weight": 0.065,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.065,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.062,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.061,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.089,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.494,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.136,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.136,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.13,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.129,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.126,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.124,
       "feature": "NOT belly color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.324,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.089,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.095,
     "shared_ref": "N3"
    },
    {
     "weight": 0.092,
     "shared_ref": "N4"
    },
    {
     "weight": 0.092,
     "shared_ref": "N5"
    },
    {
     "weight": 0.091,
     "shared_ref": "N6"
    },
    {
     "weight": 0.091,
     "shared_ref": "N7"
    },
    {
     "weight": 0.089,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R004
```json
{
 "id": "N1",
 "operator": "LHD",
 "name": "low hyper-disjunction",
 "andness": -0.097,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.641,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.121,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.982,
     "id": "N3",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.03,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.221,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.21,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.156,
       "feature": "breast color: yellow"
      },
      {
       "weight": 0.135,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.078,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.074,
       "feature": "back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.003,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.314,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.55,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.104,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.086,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.071,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.057,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.053,
       "feature": "NOT crown color: black"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.494,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.382,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.042,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.04,
       "feature": "NOT upperparts color: grey"
      },
      {
       "weight": 0.037,
       "feature": "NOT crown color: black"
      },
      {
       "weight": 0.035,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.021,
       "feature": "NOT leg color: grey"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.326,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.178,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.115,
       "feature": "breast color: black"
      },
      {
       "weight": 0.109,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.106,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.092,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.087,
       "feature": "NOT primary color: brown"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.619,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.089,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.056,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.052,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.048,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.044,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.044,
       "feature": "head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.002,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.652,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.087,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.074,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.062,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.053,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.047,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.036,
       "feature": "NOT wing pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.359,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.144,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.982,
     "shared_ref": "N3"
    },
    {
     "weight": 0.003,
     "shared_ref": "N4"
    },
    {
     "weight": 0.002,
     "shared_ref": "N5"
    },
    {
     "weight": 0.002,
     "shared_ref": "N6"
    },
    {
     "weight": 0.002,
     "shared_ref": "N7"
    },
    {
     "weight": 0.002,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R114
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.952,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.679,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.076,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.176,
     "id": "N3",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.245,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.888,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.054,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.038,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.008,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.007,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.005,
       "feature": "breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.165,
     "id": "N4",
     "operator": "HD-",
     "name": "low hard disjunction",
     "andness": 0.249,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.929,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.052,
       "feature": "crown color: white"
      },
      {
       "weight": 0.011,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.008,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.087,
     "id": "N5",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.314,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.152,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.117,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.116,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.085,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.071,
       "feature": "wing color: buff"
      }
     ]
    },
    {
     "weight": 0.074,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.08,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.239,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.154,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.114,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.112,
       "feature": "back color: buff"
      },
      {
       "weight": 0.093,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.082,
       "feature": "NOT primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N7",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.996,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.309,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.147,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.12,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.104,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.1,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.094,
       "feature": "NOT tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.069,
     "id": "N8",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.099,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.144,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.113,
       "feature": "back color: buff"
      },
      {
       "weight": 0.111,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.11,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.097,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.079,
       "feature": "forehead color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.321,
   "id": "N9",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.082,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.176,
     "shared_ref": "N3"
    },
    {
     "weight": 0.165,
     "shared_ref": "N4"
    },
    {
     "weight": 0.087,
     "shared_ref": "N5"
    },
    {
     "weight": 0.074,
     "shared_ref": "N6"
    },
    {
     "weight": 0.073,
     "shared_ref": "N7"
    },
    {
     "weight": 0.069,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R064
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.31,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.575,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.001,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.143,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.38,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.193,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.168,
       "feature": "wing color: white"
      },
      {
       "weight": 0.119,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.085,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.074,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.072,
       "feature": "wing pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.122,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.254,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.553,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.254,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.091,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.075,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.028,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.118,
     "id": "N5",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.129,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.525,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.329,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.146,
       "feature": "throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.112,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.367,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.182,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.143,
       "feature": "wing color: white"
      },
      {
       "weight": 0.13,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.102,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.072,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.063,
       "feature": "belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.287,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.414,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.24,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.166,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.076,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.056,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.026,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.373,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.332,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.309,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.132,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.12,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.106,
       "feature": "NOT breast pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.425,
   "id": "N9",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.001,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.143,
     "shared_ref": "N3"
    },
    {
     "weight": 0.122,
     "shared_ref": "N4"
    },
    {
     "weight": 0.118,
     "shared_ref": "N5"
    },
    {
     "weight": 0.112,
     "shared_ref": "N6"
    },
    {
     "weight": 0.108,
     "shared_ref": "N7"
    },
    {
     "weight": 0.091,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R171
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.648,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.726,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.259,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.118,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.319,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.284,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.174,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.144,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.07,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.068,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.068,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.113,
     "id": "N4",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.088,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.369,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.268,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.254,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.077,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.032,
       "feature": "throat color: yellow"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.311,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.28,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.216,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.141,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.085,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.069,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.067,
       "feature": "upper tail color: grey"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N6",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.02,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.963,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.02,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.017,
       "feature": "crown color: white"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.119,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.837,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.101,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.029,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.012,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.007,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.007,
       "feature": "NOT belly color: black"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.28,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.544,
       "feature": "crown color: white"
      },
      {
       "weight": 0.456,
       "feature": "NOT underparts color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.274,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.082,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.118,
     "shared_ref": "N3"
    },
    {
     "weight": 0.113,
     "shared_ref": "N4"
    },
    {
     "weight": 0.108,
     "shared_ref": "N5"
    },
    {
     "weight": 0.108,
     "shared_ref": "N6"
    },
    {
     "weight": 0.101,
     "shared_ref": "N7"
    },
    {
     "weight": 0.088,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R072
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.894,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.641,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.034,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.15,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.328,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.256,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.162,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.153,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.07,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.056,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.049,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.54,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.127,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.116,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.1,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.087,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.084,
       "feature": "NOT bill color: grey"
      },
      {
       "weight": 0.076,
       "feature": "NOT nape color: grey"
      }
     ]
    },
    {
     "weight": 0.099,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.474,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.147,
       "feature": "NOT primary color: black"
      },
      {
       "weight": 0.128,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.106,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.105,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.095,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.091,
       "feature": "NOT nape color: buff"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.355,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.22,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.203,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.197,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.195,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.185,
       "feature": "NOT size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.502,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.113,
       "feature": "NOT belly color: grey"
      },
      {
       "weight": 0.109,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.097,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.097,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.096,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.092,
       "feature": "NOT bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.076,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.367,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.107,
       "feature": "NOT primary color: white"
      },
      {
       "weight": 0.1,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.089,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.079,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.073,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.072,
       "feature": "NOT bill shape: hooked seabird"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.359,
   "id": "N9",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.901,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.15,
     "shared_ref": "N3"
    },
    {
     "weight": 0.105,
     "shared_ref": "N4"
    },
    {
     "weight": 0.099,
     "shared_ref": "N5"
    },
    {
     "weight": 0.098,
     "shared_ref": "N6"
    },
    {
     "weight": 0.093,
     "shared_ref": "N7"
    },
    {
     "weight": 0.076,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R121
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.733,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.828,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.229,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.146,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.375,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.563,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.437,
       "feature": "wing pattern: solid"
      }
     ]
    },
    {
     "weight": 0.129,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.537,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.119,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.11,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.097,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.09,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.078,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.053,
       "feature": "tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.495,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.13,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.12,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.106,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.098,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.058,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.057,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.48,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.061,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.056,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.055,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.054,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.028,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.026,
       "feature": "NOT bill color: black"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N7",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.298,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.051,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.048,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.047,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.045,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.045,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.044,
       "feature": "NOT size: medium (9 - 16 in)"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.089,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.065,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.053,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.053,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.053,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.052,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.052,
       "feature": "NOT forehead color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.172,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.427,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.146,
     "shared_ref": "N3"
    },
    {
     "weight": 0.129,
     "shared_ref": "N4"
    },
    {
     "weight": 0.116,
     "shared_ref": "N5"
    },
    {
     "weight": 0.093,
     "shared_ref": "N6"
    },
    {
     "weight": 0.092,
     "shared_ref": "N7"
    },
    {
     "weight": 0.088,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R126
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.192,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.586,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.007,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.243,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.197,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.277,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.276,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.215,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.114,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.078,
       "feature": "back color: grey"
      },
      {
       "weight": 0.04,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.153,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.311,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.263,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.205,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.108,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.083,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.075,
       "feature": "back color: grey"
      },
      {
       "weight": 0.056,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.132,
     "id": "N5",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.952,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.678,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.107,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.085,
       "feature": "crown color: grey"
      },
      {
       "weight": 0.082,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.028,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.021,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.075,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.296,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.429,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.288,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.187,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.053,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.015,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.007,
       "feature": "NOT primary color: grey"
      }
     ]
    },
    {
     "weight": 0.061,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.542,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.171,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.149,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.139,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.133,
       "feature": "NOT throat color: grey"
      },
      {
       "weight": 0.114,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.11,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.06,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.478,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.197,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.186,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.182,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.174,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.161,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.1,
       "feature": "NOT underparts color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.414,
   "id": "N9",
   "operator": "D",
   "name": "pure disjunction",
   "andness": -0.03,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.243,
     "shared_ref": "N3"
    },
    {
     "weight": 0.153,
     "shared_ref": "N4"
    },
    {
     "weight": 0.132,
     "shared_ref": "N5"
    },
    {
     "weight": 0.075,
     "shared_ref": "N6"
    },
    {
     "weight": 0.061,
     "shared_ref": "N7"
    },
    {
     "weight": 0.06,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R017
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.482,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.794,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.918,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.452,
     "id": "N3",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 0.982,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.255,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.246,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.154,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.102,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.042,
       "feature": "wing shape: rounded-wings"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.122,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.445,
       "feature": "upperparts color: yellow"
      },
      {
       "weight": 0.205,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.155,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.113,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.082,
       "feature": "bill length: shorter than head"
      }
     ]
    },
    {
     "weight": 0.077,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.459,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.23,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.223,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.139,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.125,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.099,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.062,
       "feature": "primary color: yellow"
      }
     ]
    },
    {
     "weight": 0.067,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.291,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.451,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.279,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.27,
       "feature": "breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.058,
     "id": "N7",
     "operator": "HC+",
     "name": "high hard conjunction",
     "andness": 0.924,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.355,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.163,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.102,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.097,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.082,
       "feature": "crown color: black"
      },
      {
       "weight": 0.067,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.048,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.256,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.069,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.054,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.043,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.042,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.041,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.041,
       "feature": "NOT belly color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.206,
   "id": "N9",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.85,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.452,
     "shared_ref": "N3"
    },
    {
     "weight": 0.096,
     "shared_ref": "N4"
    },
    {
     "weight": 0.077,
     "shared_ref": "N5"
    },
    {
     "weight": 0.067,
     "shared_ref": "N6"
    },
    {
     "weight": 0.058,
     "shared_ref": "N7"
    },
    {
     "weight": 0.048,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R049
```json
{
 "id": "N1",
 "operator": "CC",
 "name": "drastic conjunction",
 "andness": 1.977,
 "verbalization": "must all be completely satisfied",
 "children": [
  {
   "weight": 0.859,
   "id": "N2",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.243,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.196,
     "id": "N3",
     "operator": "SD",
     "name": "medium soft disjunction",
     "andness": 0.372,
     "verbalization": "nice to have some",
     "children": [
      {
       "weight": 0.411,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.247,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.205,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.132,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.005,
       "feature": "NOT bill color: grey"
      }
     ]
    },
    {
     "weight": 0.166,
     "id": "N4",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.75,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.595,
       "feature": "bill color: buff"
      },
      {
       "weight": 0.296,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.081,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.028,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.135,
     "id": "N5",
     "operator": "SC",
     "name": "medium soft conjunction",
     "andness": 0.623,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.381,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.272,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.204,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.073,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.056,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.014,
       "feature": "belly color: brown"
      }
     ]
    },
    {
     "weight": 0.131,
     "id": "N6",
     "operator": "C",
     "name": "pure conjunction",
     "andness": 1.008,
     "verbalization": "decided by lowest",
     "children": [
      {
       "weight": 0.329,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.276,
       "feature": "crown color: black"
      },
      {
       "weight": 0.22,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.117,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.042,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.008,
       "feature": "NOT breast color: buff"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.134,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.654,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.114,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.085,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.065,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.045,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.037,
       "feature": "NOT under tail color: brown"
      }
     ]
    },
    {
     "weight": 0.074,
     "id": "N8",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.709,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.722,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.182,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.097,
       "feature": "breast color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.141,
   "id": "N9",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.242,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.196,
     "shared_ref": "N3"
    },
    {
     "weight": 0.166,
     "shared_ref": "N4"
    },
    {
     "weight": 0.135,
     "shared_ref": "N5"
    },
    {
     "weight": 0.131,
     "shared_ref": "N6"
    },
    {
     "weight": 0.081,
     "shared_ref": "N7"
    },
    {
     "weight": 0.074,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R002
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.657,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.829,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.877,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.311,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.144,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.297,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.233,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.219,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.135,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.116,
       "feature": "wing color: black"
      }
     ]
    },
    {
     "weight": 0.072,
     "id": "N4",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.008,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.16,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.081,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.073,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.072,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.065,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.062,
       "feature": "NOT under tail color: white"
      }
     ]
    },
    {
     "weight": 0.07,
     "id": "N5",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.204,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.163,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.046,
       "feature": "bill shape: cone"
      },
      {
       "weight": 0.043,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.041,
       "feature": "upperparts color: buff"
      },
      {
       "weight": 0.035,
       "feature": "crown color: white"
      },
      {
       "weight": 0.034,
       "feature": "forehead color: white"
      }
     ]
    },
    {
     "weight": 0.068,
     "id": "N6",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.271,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.085,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.079,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.048,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.041,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.041,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.035,
       "feature": "bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.068,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.366,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.343,
       "feature": "forehead color: yellow"
      },
      {
       "weight": 0.246,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.191,
       "feature": "head pattern: plain"
      },
      {
       "weight": 0.124,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.097,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.067,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.206,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.156,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.049,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.047,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.046,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.043,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.043,
       "feature": "nape color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.171,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.152,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.311,
     "shared_ref": "N3"
    },
    {
     "weight": 0.072,
     "shared_ref": "N4"
    },
    {
     "weight": 0.07,
     "shared_ref": "N5"
    },
    {
     "weight": 0.068,
     "shared_ref": "N6"
    },
    {
     "weight": 0.068,
     "shared_ref": "N7"
    },
    {
     "weight": 0.067,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R119
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.87,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.637,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.907,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.107,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.409,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.277,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.175,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.135,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.096,
       "feature": "throat color: white"
      },
      {
       "weight": 0.092,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.087,
       "feature": "primary color: brown"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.547,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.175,
       "feature": "NOT tail pattern: striped"
      },
      {
       "weight": 0.119,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.07,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.066,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.065,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.064,
       "feature": "NOT upper tail color: white"
      }
     ]
    },
    {
     "weight": 0.083,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.6,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.076,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.073,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.057,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.057,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.051,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.051,
       "feature": "NOT wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.082,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.563,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.08,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.077,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.076,
       "feature": "NOT bill color: black"
      },
      {
       "weight": 0.076,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.074,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.069,
       "feature": "NOT bill shape: hooked seabird"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.532,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.082,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.072,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.072,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.069,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.067,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.063,
       "feature": "NOT crown color: white"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.599,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.063,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.062,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.061,
       "feature": "NOT breast color: yellow"
      },
      {
       "weight": 0.061,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.059,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.053,
       "feature": "NOT under tail color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.363,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.526,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.107,
     "shared_ref": "N3"
    },
    {
     "weight": 0.084,
     "shared_ref": "N4"
    },
    {
     "weight": 0.083,
     "shared_ref": "N5"
    },
    {
     "weight": 0.082,
     "shared_ref": "N6"
    },
    {
     "weight": 0.081,
     "shared_ref": "N7"
    },
    {
     "weight": 0.081,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R097
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.528,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.521,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 0.976,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.134,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.285,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.212,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.169,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.158,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.129,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.124,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.119,
       "feature": "tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.118,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.398,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.431,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.224,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.135,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.082,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.071,
       "feature": "upperparts color: brown"
      },
      {
       "weight": 0.058,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.532,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.286,
       "feature": "primary color: white"
      },
      {
       "weight": 0.168,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.105,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.084,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.076,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.067,
       "feature": "back color: brown"
      }
     ]
    },
    {
     "weight": 0.116,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.569,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.203,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.136,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.119,
       "feature": "breast color: white"
      },
      {
       "weight": 0.118,
       "feature": "throat color: white"
      },
      {
       "weight": 0.118,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.113,
       "feature": "belly color: white"
      }
     ]
    },
    {
     "weight": 0.104,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.505,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.301,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.266,
       "feature": "primary color: white"
      },
      {
       "weight": 0.073,
       "feature": "head pattern: eyebrow"
      },
      {
       "weight": 0.071,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.063,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.062,
       "feature": "back color: brown"
      }
     ]
    },
    {
     "weight": 0.062,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.313,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.095,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.049,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.048,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.047,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.047,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.045,
       "feature": "NOT wing color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.479,
   "id": "N9",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.09,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.134,
     "shared_ref": "N3"
    },
    {
     "weight": 0.118,
     "shared_ref": "N4"
    },
    {
     "weight": 0.117,
     "shared_ref": "N5"
    },
    {
     "weight": 0.116,
     "shared_ref": "N6"
    },
    {
     "weight": 0.104,
     "shared_ref": "N7"
    },
    {
     "weight": 0.062,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R056
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.853,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.594,
   "id": "N2",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.37,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.111,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.303,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.361,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.258,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.196,
       "feature": "wing color: white"
      },
      {
       "weight": 0.125,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.036,
       "feature": "bill color: black"
      },
      {
       "weight": 0.023,
       "feature": "size: small (5 - 9 in)"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N4",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.16,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.371,
       "feature": "wing color: white"
      },
      {
       "weight": 0.259,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.118,
       "feature": "leg color: black"
      },
      {
       "weight": 0.113,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.065,
       "feature": "wing color: black"
      },
      {
       "weight": 0.027,
       "feature": "bill shape: all-purpose"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N5",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.107,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.669,
       "feature": "nape color: buff"
      },
      {
       "weight": 0.18,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.094,
       "feature": "bill color: black"
      },
      {
       "weight": 0.05,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.004,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.003,
       "feature": "NOT upperparts color: black"
      }
     ]
    },
    {
     "weight": 0.072,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.146,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.352,
       "feature": "throat color: black"
      },
      {
       "weight": 0.236,
       "feature": "back color: grey"
      },
      {
       "weight": 0.21,
       "feature": "leg color: black"
      },
      {
       "weight": 0.095,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.075,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.031,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.072,
     "id": "N7",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.039,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.996,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.004,
       "feature": "NOT wing color: grey"
      }
     ]
    },
    {
     "weight": 0.07,
     "id": "N8",
     "operator": "D",
     "name": "pure disjunction",
     "andness": -0.01,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.468,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.275,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.125,
       "feature": "underparts color: grey"
      },
      {
       "weight": 0.066,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.046,
       "feature": "primary color: grey"
      },
      {
       "weight": 0.003,
       "feature": "NOT tail pattern: striped"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.406,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.054,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.111,
     "shared_ref": "N3"
    },
    {
     "weight": 0.097,
     "shared_ref": "N4"
    },
    {
     "weight": 0.095,
     "shared_ref": "N5"
    },
    {
     "weight": 0.072,
     "shared_ref": "N6"
    },
    {
     "weight": 0.072,
     "shared_ref": "N7"
    },
    {
     "weight": 0.07,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R184
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.521,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.995,
   "id": "N2",
   "operator": "HC-",
   "name": "low hard conjunction",
   "andness": 0.75,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.275,
     "id": "N3",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.168,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.313,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.229,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.143,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.092,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.053,
       "feature": "leg color: black"
      },
      {
       "weight": 0.046,
       "feature": "upperparts color: grey"
      }
     ]
    },
    {
     "weight": 0.219,
     "id": "N4",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.256,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.443,
       "feature": "tail pattern: multi-colored"
      },
      {
       "weight": 0.196,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.089,
       "feature": "upperparts color: grey"
      },
      {
       "weight": 0.075,
       "feature": "bill color: black"
      },
      {
       "weight": 0.059,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.059,
       "feature": "back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.202,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.321,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.38,
       "feature": "upper tail color: grey"
      },
      {
       "weight": 0.29,
       "feature": "back color: buff"
      },
      {
       "weight": 0.203,
       "feature": "leg color: black"
      },
      {
       "weight": 0.072,
       "feature": "belly pattern: solid"
      },
      {
       "weight": 0.055,
       "feature": "shape: perching-like"
      }
     ]
    },
    {
     "weight": 0.172,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.31,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.446,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.341,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.092,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.071,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.049,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.039,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.11,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.589,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.245,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.132,
       "feature": "wing color: grey"
      },
      {
       "weight": 0.033,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.001,
       "feature": "under tail color: white"
      }
     ]
    },
    {
     "weight": 0.031,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.515,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.498,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.355,
       "feature": "under tail color: grey"
      },
      {
       "weight": 0.061,
       "feature": "tail shape: notched tail"
      },
      {
       "weight": 0.051,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.035,
       "feature": "eye color: black"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.005,
   "id": "N9",
   "operator": "HD-",
   "name": "low hard disjunction",
   "andness": 0.25,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.275,
     "shared_ref": "N3"
    },
    {
     "weight": 0.219,
     "shared_ref": "N4"
    },
    {
     "weight": 0.202,
     "shared_ref": "N5"
    },
    {
     "weight": 0.172,
     "shared_ref": "N6"
    },
    {
     "weight": 0.039,
     "shared_ref": "N7"
    },
    {
     "weight": 0.031,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R123
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.926,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.72,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.009,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.413,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.756,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.304,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.187,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.112,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.081,
       "feature": "size: small (5 - 9 in)"
      },
      {
       "weight": 0.074,
       "feature": "crown color: black"
      },
      {
       "weight": 0.065,
       "feature": "upperparts color: black"
      }
     ]
    },
    {
     "weight": 0.242,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.759,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.271,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.167,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.1,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.084,
       "feature": "nape color: white"
      },
      {
       "weight": 0.058,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.054,
       "feature": "forehead color: black"
      }
     ]
    },
    {
     "weight": 0.045,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.313,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.192,
       "feature": "wing color: white"
      },
      {
       "weight": 0.186,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.15,
       "feature": "throat color: white"
      },
      {
       "weight": 0.122,
       "feature": "bill color: black"
      },
      {
       "weight": 0.06,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.043,
       "feature": "belly color: brown"
      }
     ]
    },
    {
     "weight": 0.044,
     "id": "N6",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.274,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.163,
       "feature": "primary color: black"
      },
      {
       "weight": 0.149,
       "feature": "wing color: black"
      },
      {
       "weight": 0.118,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.079,
       "feature": "NOT under tail color: white"
      },
      {
       "weight": 0.073,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.05,
       "feature": "size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.042,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.486,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.057,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.041,
       "feature": "belly color: grey"
      },
      {
       "weight": 0.04,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.038,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.038,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.037,
       "feature": "belly color: yellow"
      }
     ]
    },
    {
     "weight": 0.042,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.723,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.037,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.037,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.035,
       "feature": "back color: yellow"
      },
      {
       "weight": 0.034,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.033,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.033,
       "feature": "forehead color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.28,
   "id": "N9",
   "operator": "D",
   "name": "pure disjunction",
   "andness": 0.03,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.413,
     "shared_ref": "N3"
    },
    {
     "weight": 0.242,
     "shared_ref": "N4"
    },
    {
     "weight": 0.045,
     "shared_ref": "N5"
    },
    {
     "weight": 0.044,
     "shared_ref": "N6"
    },
    {
     "weight": 0.042,
     "shared_ref": "N7"
    },
    {
     "weight": 0.042,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R008
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.267,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.658,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.871,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.153,
     "id": "N3",
     "operator": "HD+",
     "name": "high hard disjunction",
     "andness": 0.087,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.643,
       "feature": "throat color: white"
      },
      {
       "weight": 0.213,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.143,
       "feature": "under tail color: black"
      }
     ]
    },
    {
     "weight": 0.141,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.724,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.031,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.026,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.025,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.025,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.025,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.025,
       "feature": "NOT forehead color: grey"
      }
     ]
    },
    {
     "weight": 0.132,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.735,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.038,
       "feature": "NOT wing pattern: spotted"
      },
      {
       "weight": 0.027,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.023,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.023,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.023,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.022,
       "feature": "NOT upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.121,
     "id": "N6",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.12,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.287,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.205,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.187,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.169,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.152,
       "feature": "NOT primary color: white"
      }
     ]
    },
    {
     "weight": 0.108,
     "id": "N7",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.138,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.293,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.248,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.242,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.217,
       "feature": "NOT primary color: white"
      }
     ]
    },
    {
     "weight": 0.098,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.389,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.195,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.161,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.108,
       "feature": "nape color: white"
      },
      {
       "weight": 0.104,
       "feature": "belly color: black"
      },
      {
       "weight": 0.079,
       "feature": "throat color: white"
      },
      {
       "weight": 0.07,
       "feature": "leg color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.342,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.427,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.153,
     "shared_ref": "N3"
    },
    {
     "weight": 0.141,
     "shared_ref": "N4"
    },
    {
     "weight": 0.132,
     "shared_ref": "N5"
    },
    {
     "weight": 0.121,
     "shared_ref": "N6"
    },
    {
     "weight": 0.108,
     "shared_ref": "N7"
    },
    {
     "weight": 0.098,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R042
```json
{
 "id": "N1",
 "operator": "HHD",
 "name": "high hyper-disjunction",
 "andness": -0.637,
 "verbalization": "above the highest",
 "children": [
  {
   "weight": 0.815,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.198,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.146,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.431,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.361,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.13,
       "feature": "back color: white"
      },
      {
       "weight": 0.101,
       "feature": "wing color: white"
      },
      {
       "weight": 0.091,
       "feature": "upper tail color: black"
      },
      {
       "weight": 0.085,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.068,
       "feature": "upperparts color: black"
      }
     ]
    },
    {
     "weight": 0.138,
     "id": "N4",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.482,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.203,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.12,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.083,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.08,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.068,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.067,
       "feature": "wing shape: pointed-wings"
      }
     ]
    },
    {
     "weight": 0.113,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.496,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.221,
       "feature": "back pattern: striped"
      },
      {
       "weight": 0.09,
       "feature": "upper tail color: white"
      },
      {
       "weight": 0.087,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.074,
       "feature": "wing pattern: striped"
      },
      {
       "weight": 0.073,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.047,
       "feature": "back color: white"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.511,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.089,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.061,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.06,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.058,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.055,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.054,
       "feature": "NOT forehead color: black"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.562,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.072,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.066,
       "feature": "NOT shape: perching-like"
      },
      {
       "weight": 0.05,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.043,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.039,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.038,
       "feature": "NOT crown color: black"
      }
     ]
    },
    {
     "weight": 0.092,
     "id": "N8",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.244,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.118,
       "feature": "breast pattern: striped"
      },
      {
       "weight": 0.112,
       "feature": "NOT breast color: black"
      },
      {
       "weight": 0.1,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.093,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.088,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.085,
       "feature": "NOT belly color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.185,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.458,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.146,
     "shared_ref": "N3"
    },
    {
     "weight": 0.138,
     "shared_ref": "N4"
    },
    {
     "weight": 0.113,
     "shared_ref": "N5"
    },
    {
     "weight": 0.101,
     "shared_ref": "N6"
    },
    {
     "weight": 0.097,
     "shared_ref": "N7"
    },
    {
     "weight": 0.092,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R078
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.904,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.711,
   "id": "N2",
   "operator": "HC+",
   "name": "high hard conjunction",
   "andness": 0.945,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.11,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.343,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.202,
       "feature": "belly color: black"
      },
      {
       "weight": 0.12,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.103,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.069,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.068,
       "feature": "back color: white"
      },
      {
       "weight": 0.054,
       "feature": "wing color: white"
      }
     ]
    },
    {
     "weight": 0.09,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.501,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.257,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.152,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.144,
       "feature": "NOT tail pattern: solid"
      },
      {
       "weight": 0.128,
       "feature": "NOT bill shape: all-purpose"
      },
      {
       "weight": 0.122,
       "feature": "NOT underparts color: buff"
      },
      {
       "weight": 0.116,
       "feature": "NOT leg color: black"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.488,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.143,
       "feature": "NOT belly pattern: solid"
      },
      {
       "weight": 0.109,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.105,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.094,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.077,
       "feature": "NOT back color: grey"
      },
      {
       "weight": 0.077,
       "feature": "NOT tail pattern: solid"
      }
     ]
    },
    {
     "weight": 0.086,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.496,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.2,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.16,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.14,
       "feature": "NOT throat color: black"
      },
      {
       "weight": 0.137,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.13,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.122,
       "feature": "NOT back pattern: striped"
      }
     ]
    },
    {
     "weight": 0.085,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.331,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.116,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.087,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.084,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.079,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.078,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.075,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.078,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.344,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.173,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.134,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.116,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.114,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.08,
       "feature": "primary color: white"
      },
      {
       "weight": 0.076,
       "feature": "under tail color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.289,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.275,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.11,
     "shared_ref": "N3"
    },
    {
     "weight": 0.09,
     "shared_ref": "N4"
    },
    {
     "weight": 0.086,
     "shared_ref": "N5"
    },
    {
     "weight": 0.086,
     "shared_ref": "N6"
    },
    {
     "weight": 0.085,
     "shared_ref": "N7"
    },
    {
     "weight": 0.078,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R163
```json
{
 "id": "N1",
 "operator": "CP",
 "name": "product t-norm",
 "andness": 1.198,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.636,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.879,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.148,
     "id": "N3",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.07,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.524,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.239,
       "feature": "leg color: grey"
      },
      {
       "weight": 0.138,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.093,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.005,
       "feature": "NOT tail pattern: multi-colored"
      }
     ]
    },
    {
     "weight": 0.14,
     "id": "N4",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.239,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.728,
       "feature": "wing pattern: multi-colored"
      },
      {
       "weight": 0.235,
       "feature": "tail pattern: solid"
      },
      {
       "weight": 0.02,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.006,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.005,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.005,
       "feature": "NOT forehead color: blue"
      }
     ]
    },
    {
     "weight": 0.128,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.332,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.245,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.21,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.192,
       "feature": "NOT wing color: yellow"
      },
      {
       "weight": 0.187,
       "feature": "NOT nape color: yellow"
      },
      {
       "weight": 0.167,
       "feature": "NOT crown color: white"
      }
     ]
    },
    {
     "weight": 0.123,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.62,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.048,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.043,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.042,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.042,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.041,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.041,
       "feature": "NOT forehead color: white"
      }
     ]
    },
    {
     "weight": 0.115,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.711,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.029,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.027,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.026,
       "feature": "NOT belly color: buff"
      },
      {
       "weight": 0.025,
       "feature": "NOT upperparts color: buff"
      },
      {
       "weight": 0.025,
       "feature": "NOT crown color: brown"
      },
      {
       "weight": 0.025,
       "feature": "NOT underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.075,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.205,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.292,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.115,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.107,
       "feature": "upperparts color: white"
      },
      {
       "weight": 0.058,
       "feature": "primary color: white"
      },
      {
       "weight": 0.05,
       "feature": "wing color: white"
      },
      {
       "weight": 0.045,
       "feature": "leg color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.364,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.382,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.148,
     "shared_ref": "N3"
    },
    {
     "weight": 0.14,
     "shared_ref": "N4"
    },
    {
     "weight": 0.128,
     "shared_ref": "N5"
    },
    {
     "weight": 0.123,
     "shared_ref": "N6"
    },
    {
     "weight": 0.115,
     "shared_ref": "N7"
    },
    {
     "weight": 0.075,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R135
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.926,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.772,
   "id": "N2",
   "operator": "LHC",
   "name": "low hyper-conjunction",
   "andness": 1.111,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.132,
     "id": "N3",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.524,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.174,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.166,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.154,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.13,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.129,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.126,
       "feature": "NOT underparts color: buff"
      }
     ]
    },
    {
     "weight": 0.111,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.621,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.073,
       "feature": "NOT leg color: black"
      },
      {
       "weight": 0.072,
       "feature": "NOT wing color: grey"
      },
      {
       "weight": 0.069,
       "feature": "NOT wing shape: rounded-wings"
      },
      {
       "weight": 0.057,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.055,
       "feature": "NOT wing pattern: solid"
      },
      {
       "weight": 0.053,
       "feature": "NOT nape color: brown"
      }
     ]
    },
    {
     "weight": 0.11,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.628,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.086,
       "feature": "NOT wing color: brown"
      },
      {
       "weight": 0.083,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.077,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.076,
       "feature": "NOT upperparts color: brown"
      },
      {
       "weight": 0.076,
       "feature": "NOT primary color: brown"
      },
      {
       "weight": 0.073,
       "feature": "NOT crown color: brown"
      }
     ]
    },
    {
     "weight": 0.107,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.624,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.04,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.037,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.036,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.036,
       "feature": "NOT size: medium (9 - 16 in)"
      },
      {
       "weight": 0.036,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.034,
       "feature": "NOT belly color: buff"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N7",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.281,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.257,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.133,
       "feature": "wing pattern: spotted"
      },
      {
       "weight": 0.065,
       "feature": "nape color: black"
      },
      {
       "weight": 0.058,
       "feature": "back color: white"
      },
      {
       "weight": 0.057,
       "feature": "nape color: white"
      },
      {
       "weight": 0.053,
       "feature": "leg color: grey"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.611,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.091,
       "feature": "NOT underparts color: black"
      },
      {
       "weight": 0.072,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.046,
       "feature": "NOT forehead color: brown"
      },
      {
       "weight": 0.045,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.032,
       "feature": "NOT nape color: brown"
      },
      {
       "weight": 0.03,
       "feature": "NOT back color: brown"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.228,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.413,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.132,
     "shared_ref": "N3"
    },
    {
     "weight": 0.111,
     "shared_ref": "N4"
    },
    {
     "weight": 0.11,
     "shared_ref": "N5"
    },
    {
     "weight": 0.107,
     "shared_ref": "N6"
    },
    {
     "weight": 0.105,
     "shared_ref": "N7"
    },
    {
     "weight": 0.097,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R083
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.888,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.634,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.076,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.739,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.165,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.132,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.119,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.087,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.084,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.059,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.062,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.154,
       "feature": "under tail color: black"
      },
      {
       "weight": 0.139,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.102,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.098,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.069,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.067,
       "feature": "crown color: brown"
      }
     ]
    },
    {
     "weight": 0.03,
     "id": "N5",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.65,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.307,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.025,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.017,
       "feature": "back color: brown"
      }
     ]
    },
    {
     "weight": 0.018,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.422,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.072,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.07,
       "feature": "wing color: yellow"
      },
      {
       "weight": 0.068,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.067,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.067,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.065,
       "feature": "under tail color: grey"
      }
     ]
    },
    {
     "weight": 0.018,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.61,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.057,
       "feature": "NOT bill length: shorter than head"
      },
      {
       "weight": 0.054,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.051,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.034,
       "feature": "throat color: grey"
      },
      {
       "weight": 0.034,
       "feature": "NOT breast color: buff"
      },
      {
       "weight": 0.032,
       "feature": "crown color: white"
      }
     ]
    },
    {
     "weight": 0.017,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.453,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.406,
       "feature": "wing pattern: solid"
      },
      {
       "weight": 0.041,
       "feature": "forehead color: grey"
      },
      {
       "weight": 0.037,
       "feature": "nape color: yellow"
      },
      {
       "weight": 0.036,
       "feature": "crown color: yellow"
      },
      {
       "weight": 0.035,
       "feature": "nape color: grey"
      },
      {
       "weight": 0.035,
       "feature": "wing color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.366,
   "id": "N9",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.048,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.739,
     "shared_ref": "N3"
    },
    {
     "weight": 0.062,
     "shared_ref": "N4"
    },
    {
     "weight": 0.03,
     "shared_ref": "N5"
    },
    {
     "weight": 0.018,
     "shared_ref": "N6"
    },
    {
     "weight": 0.018,
     "shared_ref": "N7"
    },
    {
     "weight": 0.017,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R025
```json
{
 "id": "N1",
 "operator": "HC+",
 "name": "high hard conjunction",
 "andness": 0.911,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.745,
   "id": "N2",
   "operator": "C",
   "name": "pure conjunction",
   "andness": 1.004,
   "verbalization": "decided by lowest",
   "children": [
    {
     "weight": 0.115,
     "id": "N3",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.218,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.426,
       "feature": "underparts color: black"
      },
      {
       "weight": 0.35,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.183,
       "feature": "wing color: brown"
      },
      {
       "weight": 0.041,
       "feature": "eye color: black"
      }
     ]
    },
    {
     "weight": 0.095,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.585,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.369,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.317,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.203,
       "feature": "NOT primary color: yellow"
      },
      {
       "weight": 0.112,
       "feature": "NOT crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.479,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.101,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.095,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.087,
       "feature": "NOT underparts color: grey"
      },
      {
       "weight": 0.081,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.076,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.063,
       "feature": "NOT underparts color: white"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.628,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.347,
       "feature": "NOT breast pattern: solid"
      },
      {
       "weight": 0.256,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.204,
       "feature": "NOT leg color: grey"
      },
      {
       "weight": 0.128,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.066,
       "feature": "NOT size: very small (3 - 5 in)"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.601,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.381,
       "feature": "NOT head pattern: plain"
      },
      {
       "weight": 0.352,
       "feature": "NOT forehead color: yellow"
      },
      {
       "weight": 0.137,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.129,
       "feature": "NOT belly color: grey"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.55,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.222,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.214,
       "feature": "NOT forehead color: white"
      },
      {
       "weight": 0.184,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.172,
       "feature": "NOT throat color: yellow"
      },
      {
       "weight": 0.142,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.066,
       "feature": "NOT back color: yellow"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.255,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.346,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.115,
     "shared_ref": "N3"
    },
    {
     "weight": 0.095,
     "shared_ref": "N4"
    },
    {
     "weight": 0.093,
     "shared_ref": "N5"
    },
    {
     "weight": 0.091,
     "shared_ref": "N6"
    },
    {
     "weight": 0.091,
     "shared_ref": "N7"
    },
    {
     "weight": 0.088,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R199
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.898,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.729,
   "id": "N2",
   "operator": "D",
   "name": "pure disjunction",
   "andness": -0.015,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.542,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.146,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.104,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.1,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.096,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.077,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.056,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.167,
     "id": "N4",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.866,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.145,
       "feature": "back pattern: solid"
      },
      {
       "weight": 0.103,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.099,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.096,
       "feature": "breast pattern: solid"
      },
      {
       "weight": 0.076,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.056,
       "feature": "belly pattern: solid"
      }
     ]
    },
    {
     "weight": 0.052,
     "id": "N5",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.285,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.139,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.097,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.072,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.069,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.059,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.054,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.05,
     "id": "N6",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.171,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.131,
       "feature": "NOT wing color: buff"
      },
      {
       "weight": 0.123,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.109,
       "feature": "NOT underparts color: white"
      },
      {
       "weight": 0.083,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.07,
       "feature": "back color: white"
      },
      {
       "weight": 0.068,
       "feature": "NOT belly color: white"
      }
     ]
    },
    {
     "weight": 0.045,
     "id": "N7",
     "operator": "LHC",
     "name": "low hyper-conjunction",
     "andness": 1.127,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.275,
       "feature": "bill length: shorter than head"
      },
      {
       "weight": 0.192,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.167,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.109,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.102,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.089,
       "feature": "wing color: grey"
      }
     ]
    },
    {
     "weight": 0.03,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.236,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.162,
       "feature": "NOT back pattern: striped"
      },
      {
       "weight": 0.154,
       "feature": "shape: perching-like"
      },
      {
       "weight": 0.132,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.129,
       "feature": "NOT under tail color: grey"
      },
      {
       "weight": 0.113,
       "feature": "NOT tail shape: notched tail"
      },
      {
       "weight": 0.087,
       "feature": "NOT wing pattern: multi-colored"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.271,
   "id": "N9",
   "operator": "D",
   "name": "pure disjunction",
   "andness": -0.007,
   "verbalization": "decided by highest",
   "children": [
    {
     "weight": 0.542,
     "shared_ref": "N3"
    },
    {
     "weight": 0.167,
     "shared_ref": "N4"
    },
    {
     "weight": 0.052,
     "shared_ref": "N5"
    },
    {
     "weight": 0.05,
     "shared_ref": "N6"
    },
    {
     "weight": 0.045,
     "shared_ref": "N7"
    },
    {
     "weight": 0.03,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R160
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.608,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.653,
   "id": "N2",
   "operator": "DP",
   "name": "product t-conorm",
   "andness": -0.207,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.134,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.434,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.178,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.139,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.093,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.075,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.055,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.054,
       "feature": "under tail color: brown"
      }
     ]
    },
    {
     "weight": 0.126,
     "id": "N4",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.167,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.495,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.41,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.086,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.006,
       "feature": "NOT upper tail color: grey"
      },
      {
       "weight": 0.003,
       "feature": "NOT primary color: black"
      }
     ]
    },
    {
     "weight": 0.105,
     "id": "N5",
     "operator": "HD",
     "name": "medium hard disjunction",
     "andness": 0.167,
     "verbalization": "enough to have any",
     "children": [
      {
       "weight": 0.496,
       "feature": "upper tail color: buff"
      },
      {
       "weight": 0.41,
       "feature": "under tail color: buff"
      },
      {
       "weight": 0.086,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.004,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.003,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.002,
       "feature": "NOT head pattern: eyebrow"
      }
     ]
    },
    {
     "weight": 0.101,
     "id": "N6",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.011,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.891,
       "feature": "back color: buff"
      },
      {
       "weight": 0.03,
       "feature": "NOT wing shape: pointed-wings"
      },
      {
       "weight": 0.029,
       "feature": "NOT size: very small (3 - 5 in)"
      },
      {
       "weight": 0.013,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.013,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.013,
       "feature": "NOT crown color: yellow"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.358,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.212,
       "feature": "bill color: grey"
      },
      {
       "weight": 0.166,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.132,
       "feature": "back color: buff"
      },
      {
       "weight": 0.09,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.064,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.064,
       "feature": "back pattern: solid"
      }
     ]
    },
    {
     "weight": 0.081,
     "id": "N8",
     "operator": "LHD",
     "name": "low hyper-disjunction",
     "andness": -0.032,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.278,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.256,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.177,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.162,
       "feature": "NOT back color: white"
      },
      {
       "weight": 0.127,
       "feature": "NOT underparts color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.347,
   "id": "N9",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.266,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.134,
     "shared_ref": "N3"
    },
    {
     "weight": 0.126,
     "shared_ref": "N4"
    },
    {
     "weight": 0.105,
     "shared_ref": "N5"
    },
    {
     "weight": 0.101,
     "shared_ref": "N6"
    },
    {
     "weight": 0.097,
     "shared_ref": "N7"
    },
    {
     "weight": 0.081,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R104
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.86,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.846,
   "id": "N2",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.046,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.39,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.114,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.1,
       "feature": "upperparts color: black"
      },
      {
       "weight": 0.086,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.069,
       "feature": "breast color: white"
      },
      {
       "weight": 0.067,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.066,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.324,
     "id": "N4",
     "operator": "SC+",
     "name": "high soft conjunction",
     "andness": 0.75,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.155,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.15,
       "feature": "back color: black"
      },
      {
       "weight": 0.117,
       "feature": "bill length: about the same as head"
      },
      {
       "weight": 0.093,
       "feature": "breast color: white"
      },
      {
       "weight": 0.09,
       "feature": "underparts color: white"
      },
      {
       "weight": 0.09,
       "feature": "breast pattern: solid"
      }
     ]
    },
    {
     "weight": 0.035,
     "id": "N5",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.574,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.08,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.067,
       "feature": "NOT leg color: buff"
      },
      {
       "weight": 0.049,
       "feature": "NOT forehead color: black"
      },
      {
       "weight": 0.048,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.046,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.045,
       "feature": "NOT tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.035,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.655,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.069,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.052,
       "feature": "forehead color: blue"
      },
      {
       "weight": 0.047,
       "feature": "NOT wing color: black"
      },
      {
       "weight": 0.043,
       "feature": "NOT bill shape: hooked seabird"
      },
      {
       "weight": 0.043,
       "feature": "crown color: blue"
      },
      {
       "weight": 0.042,
       "feature": "shape: duck-like"
      }
     ]
    },
    {
     "weight": 0.034,
     "id": "N7",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.533,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.096,
       "feature": "NOT wing pattern: striped"
      },
      {
       "weight": 0.084,
       "feature": "NOT back pattern: solid"
      },
      {
       "weight": 0.082,
       "feature": "NOT back color: buff"
      },
      {
       "weight": 0.061,
       "feature": "NOT primary color: buff"
      },
      {
       "weight": 0.056,
       "feature": "NOT bill shape: dagger"
      },
      {
       "weight": 0.054,
       "feature": "NOT tail shape: notched tail"
      }
     ]
    },
    {
     "weight": 0.033,
     "id": "N8",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.452,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.175,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.053,
       "feature": "NOT forehead color: grey"
      },
      {
       "weight": 0.046,
       "feature": "forehead color: white"
      },
      {
       "weight": 0.045,
       "feature": "NOT back color: yellow"
      },
      {
       "weight": 0.042,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.04,
       "feature": "NOT crown color: white"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.154,
   "id": "N9",
   "operator": "HD+",
   "name": "high hard disjunction",
   "andness": 0.099,
   "verbalization": "enough to have any",
   "children": [
    {
     "weight": 0.39,
     "shared_ref": "N3"
    },
    {
     "weight": 0.324,
     "shared_ref": "N4"
    },
    {
     "weight": 0.035,
     "shared_ref": "N5"
    },
    {
     "weight": 0.035,
     "shared_ref": "N6"
    },
    {
     "weight": 0.034,
     "shared_ref": "N7"
    },
    {
     "weight": 0.033,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R089
```json
{
 "id": "N1",
 "operator": "LHC",
 "name": "low hyper-conjunction",
 "andness": 1.129,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.596,
   "id": "N2",
   "operator": "CP",
   "name": "product t-norm",
   "andness": 1.27,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.136,
     "id": "N3",
     "operator": "DP",
     "name": "product t-conorm",
     "andness": -0.307,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.807,
       "feature": "NOT breast color: grey"
      },
      {
       "weight": 0.193,
       "feature": "NOT wing color: yellow"
      }
     ]
    },
    {
     "weight": 0.131,
     "id": "N4",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.4,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.531,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.055,
       "feature": "NOT underparts color: brown"
      },
      {
       "weight": 0.053,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.049,
       "feature": "NOT forehead color: blue"
      },
      {
       "weight": 0.048,
       "feature": "NOT back pattern: multi-colored"
      },
      {
       "weight": 0.047,
       "feature": "NOT crown color: white"
      }
     ]
    },
    {
     "weight": 0.121,
     "id": "N5",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.65,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.088,
       "feature": "NOT under tail color: buff"
      },
      {
       "weight": 0.078,
       "feature": "NOT throat color: buff"
      },
      {
       "weight": 0.07,
       "feature": "NOT crown color: blue"
      },
      {
       "weight": 0.069,
       "feature": "NOT breast pattern: striped"
      },
      {
       "weight": 0.068,
       "feature": "NOT crown color: yellow"
      },
      {
       "weight": 0.064,
       "feature": "NOT nape color: buff"
      }
     ]
    },
    {
     "weight": 0.117,
     "id": "N6",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.731,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.066,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.059,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.058,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT underparts color: yellow"
      },
      {
       "weight": 0.05,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.046,
       "feature": "NOT belly color: brown"
      }
     ]
    },
    {
     "weight": 0.114,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.669,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.342,
       "feature": "bill shape: dagger"
      },
      {
       "weight": 0.045,
       "feature": "NOT head pattern: eyebrow"
      },
      {
       "weight": 0.045,
       "feature": "NOT nape color: white"
      },
      {
       "weight": 0.044,
       "feature": "NOT belly color: yellow"
      },
      {
       "weight": 0.038,
       "feature": "NOT breast pattern: multi-colored"
      },
      {
       "weight": 0.037,
       "feature": "NOT belly color: black"
      }
     ]
    },
    {
     "weight": 0.084,
     "id": "N8",
     "operator": "CP",
     "name": "product t-norm",
     "andness": 1.208,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.346,
       "feature": "under tail color: white"
      },
      {
       "weight": 0.146,
       "feature": "primary color: buff"
      },
      {
       "weight": 0.12,
       "feature": "under tail color: brown"
      },
      {
       "weight": 0.055,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.05,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.046,
       "feature": "belly color: buff"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.404,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.561,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.136,
     "shared_ref": "N3"
    },
    {
     "weight": 0.131,
     "shared_ref": "N4"
    },
    {
     "weight": 0.121,
     "shared_ref": "N5"
    },
    {
     "weight": 0.117,
     "shared_ref": "N6"
    },
    {
     "weight": 0.114,
     "shared_ref": "N7"
    },
    {
     "weight": 0.084,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R138
```json
{
 "id": "N1",
 "operator": "HHC",
 "name": "high hyper-conjunction",
 "andness": 1.442,
 "verbalization": "below the lowest",
 "children": [
  {
   "weight": 0.93,
   "id": "N2",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.085,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.605,
     "id": "N3",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.75,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.125,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.125,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.088,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.087,
       "feature": "back color: buff"
      },
      {
       "weight": 0.065,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.063,
       "feature": "leg color: buff"
      }
     ]
    },
    {
     "weight": 0.199,
     "id": "N4",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.812,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.285,
       "feature": "back color: buff"
      },
      {
       "weight": 0.215,
       "feature": "underparts color: brown"
      },
      {
       "weight": 0.207,
       "feature": "leg color: buff"
      },
      {
       "weight": 0.168,
       "feature": "breast color: buff"
      },
      {
       "weight": 0.126,
       "feature": "under tail color: brown"
      }
     ]
    },
    {
     "weight": 0.073,
     "id": "N5",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.837,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.359,
       "feature": "tail pattern: striped"
      },
      {
       "weight": 0.253,
       "feature": "belly color: brown"
      },
      {
       "weight": 0.145,
       "feature": "crown color: brown"
      },
      {
       "weight": 0.11,
       "feature": "underparts color: buff"
      },
      {
       "weight": 0.051,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.043,
       "feature": "back color: brown"
      }
     ]
    },
    {
     "weight": 0.062,
     "id": "N6",
     "operator": "HC",
     "name": "medium hard conjunction",
     "andness": 0.879,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.468,
       "feature": "breast color: brown"
      },
      {
       "weight": 0.224,
       "feature": "belly color: buff"
      },
      {
       "weight": 0.111,
       "feature": "forehead color: brown"
      },
      {
       "weight": 0.079,
       "feature": "throat color: buff"
      },
      {
       "weight": 0.067,
       "feature": "nape color: brown"
      },
      {
       "weight": 0.052,
       "feature": "upper tail color: brown"
      }
     ]
    },
    {
     "weight": 0.019,
     "id": "N7",
     "operator": "HC-",
     "name": "low hard conjunction",
     "andness": 0.76,
     "verbalization": "must have all",
     "children": [
      {
       "weight": 0.181,
       "feature": "bill shape: all-purpose"
      },
      {
       "weight": 0.141,
       "feature": "wing color: buff"
      },
      {
       "weight": 0.1,
       "feature": "primary color: brown"
      },
      {
       "weight": 0.095,
       "feature": "size: very small (3 - 5 in)"
      },
      {
       "weight": 0.086,
       "feature": "wing shape: rounded-wings"
      },
      {
       "weight": 0.083,
       "feature": "wing color: brown"
      }
     ]
    },
    {
     "weight": 0.01,
     "id": "N8",
     "operator": "SC",
     "name": "medium soft conjunction",
     "andness": 0.643,
     "verbalization": "nice to have most",
     "children": [
      {
       "weight": 0.08,
       "feature": "back pattern: multi-colored"
      },
      {
       "weight": 0.079,
       "feature": "shape: duck-like"
      },
      {
       "weight": 0.066,
       "feature": "bill shape: hooked seabird"
      },
      {
       "weight": 0.064,
       "feature": "throat color: yellow"
      },
      {
       "weight": 0.061,
       "feature": "breast pattern: multi-colored"
      },
      {
       "weight": 0.06,
       "feature": "bill shape: dagger"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.07,
   "id": "N9",
   "operator": "LHD",
   "name": "low hyper-disjunction",
   "andness": -0.177,
   "verbalization": "above the highest",
   "children": [
    {
     "weight": 0.605,
     "shared_ref": "N3"
    },
    {
     "weight": 0.199,
     "shared_ref": "N4"
    },
    {
     "weight": 0.073,
     "shared_ref": "N5"
    },
    {
     "weight": 0.062,
     "shared_ref": "N6"
    },
    {
     "weight": 0.019,
     "shared_ref": "N7"
    },
    {
     "weight": 0.01,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```

### R018
```json
{
 "id": "N1",
 "operator": "HC",
 "name": "medium hard conjunction",
 "andness": 0.884,
 "verbalization": "must have all",
 "children": [
  {
   "weight": 0.56,
   "id": "N2",
   "operator": "HC",
   "name": "medium hard conjunction",
   "andness": 0.837,
   "verbalization": "must have all",
   "children": [
    {
     "weight": 0.122,
     "id": "N3",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.386,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.382,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.119,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.106,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.095,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.07,
       "feature": "underparts color: yellow"
      },
      {
       "weight": 0.057,
       "feature": "breast color: yellow"
      }
     ]
    },
    {
     "weight": 0.097,
     "id": "N4",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.017,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.247,
       "feature": "NOT bill shape: cone"
      },
      {
       "weight": 0.21,
       "feature": "NOT wing pattern: multi-colored"
      },
      {
       "weight": 0.188,
       "feature": "NOT wing color: white"
      },
      {
       "weight": 0.148,
       "feature": "NOT primary color: grey"
      },
      {
       "weight": 0.118,
       "feature": "NOT upperparts color: white"
      },
      {
       "weight": 0.089,
       "feature": "NOT breast pattern: striped"
      }
     ]
    },
    {
     "weight": 0.096,
     "id": "N5",
     "operator": "D",
     "name": "pure disjunction",
     "andness": 0.001,
     "verbalization": "decided by highest",
     "children": [
      {
       "weight": 0.235,
       "feature": "NOT tail pattern: multi-colored"
      },
      {
       "weight": 0.21,
       "feature": "NOT crown color: grey"
      },
      {
       "weight": 0.203,
       "feature": "NOT nape color: grey"
      },
      {
       "weight": 0.142,
       "feature": "NOT belly color: white"
      },
      {
       "weight": 0.107,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.103,
       "feature": "NOT wing color: buff"
      }
     ]
    },
    {
     "weight": 0.093,
     "id": "N6",
     "operator": "HHC",
     "name": "high hyper-conjunction",
     "andness": 1.372,
     "verbalization": "below the lowest",
     "children": [
      {
       "weight": 0.33,
       "feature": "wing shape: pointed-wings"
      },
      {
       "weight": 0.103,
       "feature": "forehead color: black"
      },
      {
       "weight": 0.092,
       "feature": "belly color: yellow"
      },
      {
       "weight": 0.088,
       "feature": "bill color: black"
      },
      {
       "weight": 0.082,
       "feature": "primary color: yellow"
      },
      {
       "weight": 0.061,
       "feature": "underparts color: yellow"
      }
     ]
    },
    {
     "weight": 0.091,
     "id": "N7",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.483,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.085,
       "feature": "NOT upperparts color: black"
      },
      {
       "weight": 0.058,
       "feature": "NOT shape: duck-like"
      },
      {
       "weight": 0.056,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.054,
       "feature": "NOT upper tail color: white"
      },
      {
       "weight": 0.052,
       "feature": "NOT breast color: brown"
      },
      {
       "weight": 0.052,
       "feature": "NOT belly color: brown"
      }
     ]
    },
    {
     "weight": 0.088,
     "id": "N8",
     "operator": "HHD",
     "name": "high hyper-disjunction",
     "andness": -0.489,
     "verbalization": "above the highest",
     "children": [
      {
       "weight": 0.055,
       "feature": "NOT bill color: buff"
      },
      {
       "weight": 0.051,
       "feature": "NOT under tail color: brown"
      },
      {
       "weight": 0.047,
       "feature": "NOT nape color: buff"
      },
      {
       "weight": 0.047,
       "feature": "NOT upper tail color: brown"
      },
      {
       "weight": 0.044,
       "feature": "NOT belly color: brown"
      },
      {
       "weight": 0.044,
       "feature": "NOT throat color: grey"
      }
     ]
    }
   ]
  },
  {
   "weight": 0.44,
   "id": "N9",
   "operator": "HHC",
   "name": "high hyper-conjunction",
   "andness": 1.51,
   "verbalization": "below the lowest",
   "children": [
    {
     "weight": 0.122,
     "shared_ref": "N3"
    },
    {
     "weight": 0.097,
     "shared_ref": "N4"
    },
    {
     "weight": 0.096,
     "shared_ref": "N5"
    },
    {
     "weight": 0.093,
     "shared_ref": "N6"
    },
    {
     "weight": 0.091,
     "shared_ref": "N7"
    },
    {
     "weight": 0.088,
     "shared_ref": "N8"
    }
   ]
  }
 ]
}
```
