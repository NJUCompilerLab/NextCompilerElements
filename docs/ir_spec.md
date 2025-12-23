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
  op_type: string              # Operator type, e.g., "Conv2D", "Function", "Module"
  inputs: [ValuePtr]           # Input values (SSA references)
  results: [ValuePtr]          # Output values (define SSA names and types)
  attrs: {string: any}         # Attribute values (schema defined in operator specification)
  regions: [Region]            # Optional nested regions for control flow

  # Derived properties (computed, not stored):
  name() -> string             # Symbol ops: "@" + attrs["sym_name"]
                               # Regular ops: results[0]->name()
  outputTypes() -> [Type]      # Derived from results[i]->type()
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

### Use-Def Chain

Value tracks all its uses for SSA transformations:

```
Value {
  name: string               # SSA name
  type: Type
  definingOp: Operation?     # Op that defines this value (null for block args)
  uses: [Operation]          # All ops that use this value

  # Methods:
  replaceAllUsesWith(newValue)  # Replace all uses with another value
  hasUses() -> bool             # Check if value has any uses
  numUses() -> int              # Number of uses
}
```

This enables:
- Dead code elimination (remove ops with no uses)
- Constant propagation (replace uses with constant)
- Common subexpression elimination (share identical computations)

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

### Text Format (MLIR-style)

For human readability, IR also supports a text format:

```
module @mlp_module {
  func @main(%input: f32[1, 784], %weight: f32[784, 256], %bias: f32[256]) -> f32[1, 256] {
  entry:
    %v0 = MatMul(%input, %weight) : f32[1, 256]
    %v1 = Add(%v0, %bias) : f32[1, 256]
    %v2 = ReLU(%v1) : f32[1, 256]
    return %v2
  }
}
```

**Syntax:**
| Element | Format | Example |
|---------|--------|---------|
| Module | `module @name { ... }` | `module @gpt2 { ... }` |
| Function | `func @name(args) -> ret { blocks }` | `func @main(%x: f32[10]) -> f32[10] { ... }` |
| Block | `label:` or `label(args):` | `entry:`, `loop(%i: i32):` |
| Operation | `%name = Op<attrs>(inputs) : type` | `%y = MatMul(%a, %b) : f32[2, 3]` |
| Attrs | `<key=value, ...>` | `<axis=-1, eps=1e-5>` |
| Type | `dtype[dim, ...]` | `f32[1, 64, 224, 224]` |
| Scalar | `dtype` (no brackets) | `f32`, `bool` |
| Return | `return %value` | `return %output` |
| Branch | `br @label` | `br @loop` |
| Cond Branch | `cond_br %c, @then, @else` | `cond_br %flag, @true, @false` |

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
  shape_rule: string         # Shape inference (broadcast/matmul implemented)
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

| Terminator | Attrs | Inputs | JSON Format |
|------------|-------|--------|-------------|
| `Return` | - | return values | `{"op_type": "Return", "inputs": ["%result"]}` |
| `Br` | `dest: string` | branch args | `{"op_type": "Br", "attrs": {"dest": "@block"}, "inputs": ["%arg"]}` |
| `CondBr` | `then: string, else: string` | condition | `{"op_type": "CondBr", "attrs": {"then": "@b1", "else": "@b2"}, "inputs": ["%cond"]}` |

**Note:** Branch destinations are stored in `attrs`, not as direct fields. Branch arguments (for block args) are passed via `inputs`.

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
