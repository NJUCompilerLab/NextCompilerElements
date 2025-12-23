# IR Spec v0.1

## Overview

Intermediate Representation Specification for a pedagogical deep learning compiler.

**Design Principles**:
- Unified Representation: Everything is an Operation
- SSA Form: Each value is uniquely defined
- Recursive Structure: Supports nested subgraphs, control flow, multi-level IR
- Progressive: Static shape first, reserved for dynamic extension

---

## Core Structure

### Three Basic Components

```
Operation {
  name: string           # SSA name, e.g., "%conv1", "@main"
  op_type: string        # Operator type, e.g., "Conv2D", "Function", "Module"
  inputs: [string]       # SSA reference, e.g., ["%x", "%weight"]
  attrs: {string: any}   # Attribute values (schema defined in operator specification)
  output_types: [Type]   # Output types
  body?: [Region]        # Optional, nested regions
}

Region {
  blocks: [Block]        # Block list
}

Block {
  label: string          # Block label, e.g., "@entry", "@loop_body"
  args: [Value]          # Block arguments (replaces phi nodes)
  ops: [Operation]       # Operation sequence
  terminator: Operation  # Terminator instruction (br/cond_br/return)
}
```

### Type Definition

```
Type {
  dtype: string          # "f32", "f16", "i32", "i8", "bool"
  shape: [int]           # Static shape, e.g., [1, 64, 28, 28]
}

Value {
  name: string           # SSA name
  type: Type
}
```

**Reserved Extension** (Dynamic Shape):
```
shape: [Dim]
Dim = int | string       # int=static, string=symbolic
```

---

## Hierarchy

```
Module (Operation, op_type="Module")
└─ body: [Region]
    └─ blocks: [Block]
        └─ ops: [Function, ...]

Function (Operation, op_type="Function")
└─ body: [Region]
    └─ blocks: [Block]
        └─ ops: [Conv2D, ReLU, ...]
             └─ body?: [Region]  ← Recursive (for control flow/loops)
```

---

## JSON Serialization Format

### Full Form (Layered Display)

**Layer 1: Module**
```json
{
  "name": "@module",
  "op_type": "Module",
  "body": [<Region>]
}
```

**Layer 2: Region (Module's body)**
```json
{
  "blocks": [<Block>]
}
```

**Layer 3: Block (contains Function)**
```json
{
  "label": "@entry",
  "args": [],
  "ops": [<Function>, ...],
  "terminator": {"op_type": "Return"}
}
```

**Layer 4: Function (an Operation with body)**
```json
{
  "name": "@main",
  "op_type": "Function",
  "body": [<Region>]
}
```

**Layer 5: Function's Region → Block → Ops**
```json
{
  "blocks": [{
    "label": "@entry",
    "args": [
      {"name": "%input", "type": {"dtype": "f32", "shape": [1, 3, 224, 224]}}
    ],
    "ops": [
      {"name": "%conv1", "op_type": "Conv2D", "inputs": ["%input", "%weight"],
       "attrs": {"stride": [1,1]}, "output_types": [{"dtype": "f32", "shape": [1,64,224,224]}]},
      {"name": "%relu1", "op_type": "ReLU", "inputs": ["%conv1"],
       "output_types": [{"dtype": "f32", "shape": [1,64,224,224]}]}
    ],
    "terminator": {"op_type": "Return", "inputs": ["%relu1"]}
  }]
}
```

**Complete Composition Example** (A simple Conv + ReLU model):
```json
{
  "name": "@module",
  "op_type": "Module",
  "body": [{
    "blocks": [{
      "label": "@entry",
      "ops": [{
        "name": "@main",
        "op_type": "Function",
        "body": [{
          "blocks": [{
            "label": "@entry",
            "args": [{"name": "%input", "type": {"dtype": "f32", "shape": [1,3,224,224]}}],
            "ops": [
              {"name": "%conv1", "op_type": "Conv2D", "inputs": ["%input", "%weight"],
               "attrs": {"stride": [1,1], "padding": [1,1,1,1]},
               "output_types": [{"dtype": "f32", "shape": [1,64,224,224]}]},
              {"name": "%relu1", "op_type": "ReLU", "inputs": ["%conv1"],
               "output_types": [{"dtype": "f32", "shape": [1,64,224,224]}]}
            ],
            "terminator": {"op_type": "Return", "inputs": ["%relu1"]}
          }]
        }]
      }],
      "terminator": {"op_type": "Return"}
    }]
  }]
}
```

### Simplified Form (Omit nesting for single Block)

```json
{
  "version": "0.1",
  "functions": [{
    "name": "@main",
    "inputs": [{"name": "%input", "dtype": "f32", "shape": [1, 3, 224, 224]}],
    "outputs": ["%relu1"],
    "ops": [
      {"name": "%conv1", "op_type": "Conv2D", "inputs": ["%input", "%weight"],
       "attrs": {"stride": [1,1]}, "output_types": [{"dtype": "f32", "shape": [1,64,224,224]}]},
      {"name": "%relu1", "op_type": "ReLU", "inputs": ["%conv1"],
       "output_types": [{"dtype": "f32", "shape": [1,64,224,224]}]}
    ]
  }]
}
```

Automatically expands to full form during parsing.

---

## Operator Specification (op_spec.yaml)

### Structure

```yaml
OpName:
  category: string           # nn / elementwise / activation / transform
  attrs:
    attr_name:
      type: string           # int / int[] / float / string / bool
      required: bool
      default: any           # Optional, default value
  dtype_rule: string         # Type inference rule
  shape_rule: ...            # Shape inference rule (Phase 2)
  constraints: [string]      # Constraints
```

### Example

```yaml
Conv2D:
  category: nn
  attrs:
    stride:   {type: "int[]", required: true}
    padding:  {type: "int[]", default: [0, 0, 0, 0]}
    dilation: {type: "int[]", default: [1, 1]}
    groups:   {type: "int", default: 1}
  dtype_rule: same_as_input
  constraints:
    - C_in % groups == 0
    - C_out % groups == 0

ReLU:
  category: activation
  attrs: {}
  dtype_rule: same_as_input

Add:
  category: elementwise
  attrs: {}
  dtype_rule: same_as_inputs  # Requires both inputs to have the same type

MatMul:
  category: linalg
  attrs: {}
  dtype_rule: same_as_inputs
  constraints:
    - lhs.shape[-1] == rhs.shape[-2]
```

---

## Terminator Types

| Terminator | Usage | Format |
|------------|------|------|
| `Return` | Function/Region Return | `{"op_type": "Return", "inputs": ["%result"]}` |
| `Br` | Unconditional Branch | `{"op_type": "Br", "dest": "@block", "args": [...]}` |
| `CondBr` | Conditional Branch | `{"op_type": "CondBr", "cond": "%c", "then": "@b1", "else": "@b2"}` |

---

## Naming Convention

| Type | Format | Example |
|------|------|------|
| Value | `%name` | `%conv1`, `%input` |
| Function/Global | `@name` | `@main`, `@attention` |
| Block Label | `@name` | `@entry`, `@loop_body` |
| Operator Type | PascalCase | `Conv2D`, `MatMul`, `ReLU` |
| Attribute Name | snake_case | `stride`, `padding`, `kernel_size` |

---

## Design Decisions Summary

| Component | Decision |
|------|------|
| Core Structure | Operation + Region + Block |
| SSA | Simplified SSA, block arguments replace phi |
| Type | dtype + shape decoupled |
| Shape | Static priority, reserved dynamic extension (symbolic) |
| Attrs | Schema defined in op_spec, IR only stores values |
| Serialization | JSON, supports both simplified and full forms |
| Unified Representation | Module/Function/Op are all Operations |
