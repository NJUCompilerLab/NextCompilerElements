/// Test: Build a simple MLP (Linear + ReLU) IR and dump to JSON
///
/// Model: y = ReLU(x @ W + b)
/// - Input:  x  [1, 784]
/// - Weight: W  [784, 256]
/// - Bias:   b  [256]
/// - Output: y  [1, 256]

#include "ir/Module.h"
#include "ir/IRBuilder.h"
#include "ir/IRPrinter.h"
#include "ir/IRTextPrinter.h"
#include <iostream>
#include <cassert>

using namespace nc::ir;

void testBasicMLP() {
    std::cout << "=== Test 1: Basic MLP (Linear + ReLU) ===\n\n";

    // Create module
    Module module("@mlp_module");

    // Add main function
    Operation* mainFunc = module.addFunction("main");
    Block* funcBlock = mainFunc->getRegion(0)->entryBlock();

    // Add function arguments (inputs)
    auto input = funcBlock->addArg("%input", Type(DType::F32, Shape({1, 784})));
    auto weight = funcBlock->addArg("%weight", Type(DType::F32, Shape({784, 256})));
    auto bias = funcBlock->addArg("%bias", Type(DType::F32, Shape({256})));

    // Build operations using IRBuilder
    IRBuilder builder(funcBlock);

    // %matmul = MatMul(%input, %weight) : [1, 256]
    auto matmul = builder.createMatMul(input, weight,
                                        Type(DType::F32, Shape({1, 256})));

    // %add = Add(%matmul, %bias) : [1, 256]
    auto add = builder.createAdd(matmul, bias,
                                  Type(DType::F32, Shape({1, 256})));

    // %relu = ReLU(%add) : [1, 256]
    auto relu = builder.createReLU(add);

    // Return %relu
    builder.createReturn({relu});

    // Print to JSON
    std::string jsonOutput = IRPrinter::print(module);
    std::cout << "=== JSON Format ===\n";
    std::cout << jsonOutput << std::endl;

    // Print to Text (MLIR-style)
    std::string textOutput = IRTextPrinter::print(module);
    std::cout << "\n=== Text Format ===\n";
    std::cout << textOutput << std::endl;

    // Basic verification
    assert(module.name() == "@mlp_module");
    assert(mainFunc->opType() == "Function");
    assert(funcBlock->numOps() == 3); // MatMul, Add, ReLU
    assert(funcBlock->terminator() != nullptr);
    assert(funcBlock->terminator()->opType() == "Return");

    std::cout << "Test 1 passed!\n\n";
}

void testOpsWithAttributes() {
    std::cout << "=== Test 2: Ops with Attributes ===\n\n";

    Module module("@attention_module");
    Operation* func = module.addFunction("attention");
    Block* block = func->getRegion(0)->entryBlock();

    // Add inputs
    auto input = block->addArg("%input", Type(DType::F32, Shape({1, 512, 768})));
    auto gamma = block->addArg("%gamma", Type(DType::F32, Shape({768})));
    auto beta = block->addArg("%beta", Type(DType::F32, Shape({768})));

    IRBuilder builder(block);

    // LayerNorm with eps attribute
    auto ln = builder.createLayerNorm(input, gamma, beta, -1, 1e-5);

    // Softmax with axis attribute
    auto softmax = builder.createSoftmax(ln, -1);

    // Reshape with shape attribute
    auto reshaped = builder.createReshape(softmax, {1, 512, 12, 64},
                                           Type(DType::F32, Shape({1, 512, 12, 64})));

    // Permute with perm attribute
    auto permuted = builder.createPermute(reshaped, {0, 2, 1, 3},
                                           Type(DType::F32, Shape({1, 12, 512, 64})));

    builder.createReturn({permuted});

    // Print
    std::string textOutput = IRTextPrinter::print(module);
    std::cout << textOutput << std::endl;

    // Verify attributes are present
    const auto& ops = block->ops();
    assert(ops.size() == 4);
    assert(ops[0]->hasAttr("axis"));
    assert(ops[0]->hasAttr("eps"));
    assert(ops[0]->getAttr<int64_t>("axis") == -1);
    assert(ops[1]->hasAttr("axis"));
    assert(ops[2]->hasAttr("shape"));
    assert(ops[3]->hasAttr("perm"));

    std::cout << "Test 2 passed!\n\n";
}

void testEdgeCases() {
    std::cout << "=== Test 3: Edge Cases ===\n\n";

    // Test 3a: Empty function (no ops, no args)
    {
        Module module("@empty_module");
        Operation* func = module.addFunction("empty_func");
        Block* block = func->getRegion(0)->entryBlock();

        // Just add a return with no values
        IRBuilder builder(block);
        builder.createReturn({});

        std::string text = IRTextPrinter::print(module);
        std::cout << "Empty function:\n" << text << std::endl;

        assert(block->numOps() == 0);
        assert(block->terminator() != nullptr);
    }

    // Test 3b: Scalar type (empty shape)
    {
        Module module("@scalar_module");
        Operation* func = module.addFunction("scalar_func");
        Block* block = func->getRegion(0)->entryBlock();

        auto scalar = block->addArg("%scalar", Type(DType::F32, Shape({})));

        IRBuilder builder(block);
        builder.createReturn({scalar});

        std::string text = IRTextPrinter::print(module);
        std::cout << "Scalar type:\n" << text << std::endl;

        // Verify scalar type has empty shape
        assert(scalar->type().shape().rank() == 0);
    }

    // Test 3c: Constant operation
    {
        Module module("@const_module");
        Operation* func = module.addFunction("const_func");
        Block* block = func->getRegion(0)->entryBlock();

        IRBuilder builder(block);
        auto c = builder.createConstant(Type(DType::F32, Shape({2, 2})), 1.0);

        builder.createReturn({c});

        std::string text = IRTextPrinter::print(module);
        std::cout << "Constant op:\n" << text << std::endl;

        assert(block->numOps() == 1);
        assert(block->ops()[0]->opType() == "Constant");
    }

    std::cout << "Test 3 passed!\n\n";
}

void testUseDefChain() {
    std::cout << "=== Test 4: Use-Def Chain ===\n\n";

    Module module("@usedef_module");
    Operation* func = module.addFunction("test");
    Block* block = func->getRegion(0)->entryBlock();

    auto x = block->addArg("%x", Type(DType::F32, Shape({1, 10})));
    auto y = block->addArg("%y", Type(DType::F32, Shape({1, 10})));

    IRBuilder builder(block);

    // Create: %add = Add(%x, %y)
    auto add = builder.createAdd(x, y, Type(DType::F32, Shape({1, 10})));

    // Create: %relu = ReLU(%add)
    auto relu = builder.createReLU(add);

    // Verify use-def chain for %x
    assert(x->numUses() == 1);  // Used by Add
    assert(x->uses()[0]->opType() == "Add");

    // Verify use-def chain for %add
    assert(add->numUses() == 1);  // Used by ReLU
    assert(add->uses()[0]->opType() == "ReLU");

    // Test replaceAllUsesWith: replace %x with %y
    x->replaceAllUsesWith(y);

    // Now Add should use (%y, %y) instead of (%x, %y)
    const auto& ops = block->ops();
    assert(ops[0]->inputs()[0] == y);
    assert(ops[0]->inputs()[1] == y);
    assert(x->numUses() == 0);  // %x has no uses now
    assert(y->numUses() == 1);  // %y used by Add (once per Op, not per input slot)

    std::cout << "Use-Def chain test passed!\n";
    std::cout << "Test 4 passed!\n\n";
}

void testControlFlow() {
    std::cout << "=== Test 5: Control Flow (Multi-Block) ===\n\n";

    Module module("@cfg_module");
    Operation* func = module.addFunction("if_else");
    Region* region = func->getRegion(0);

    // Entry block with conditional branch
    Block* entry = region->entryBlock();
    auto cond = entry->addArg("%cond", Type(DType::Bool, Shape({})));
    auto x = entry->addArg("%x", Type(DType::F32, Shape({10})));

    // Add then/else blocks
    Block* thenBlock = region->addBlock("@then");
    Block* elseBlock = region->addBlock("@else");
    Block* mergeBlock = region->addBlock("@merge");

    // Entry: cond_br %cond, @then, @else
    IRBuilder entryBuilder(entry);
    entryBuilder.createCondBr(cond, thenBlock, elseBlock);

    // Then block: %t = ReLU(%x), br @merge
    IRBuilder thenBuilder(thenBlock);
    auto thenResult = thenBuilder.createReLU(x);
    thenBuilder.createBr(mergeBlock);

    // Else block: %e = GELU(%x), br @merge
    IRBuilder elseBuilder(elseBlock);
    auto elseResult = elseBuilder.createGELU(x);
    elseBuilder.createBr(mergeBlock);

    // Merge block: return (simplified, no phi for now)
    IRBuilder mergeBuilder(mergeBlock);
    mergeBuilder.createReturn({});

    // Print IR
    std::string textOutput = IRTextPrinter::print(module);
    std::cout << textOutput << std::endl;

    // Verify structure
    assert(region->numBlocks() == 4);
    assert(entry->terminator()->opType() == "CondBr");
    assert(thenBlock->terminator()->opType() == "Br");
    assert(elseBlock->terminator()->opType() == "Br");
    assert(mergeBlock->terminator()->opType() == "Return");

    // Verify CondBr attributes
    assert(entry->terminator()->hasAttr("then"));
    assert(entry->terminator()->hasAttr("else"));
    assert(entry->terminator()->getAttr<std::string>("then") == "@then");
    assert(entry->terminator()->getAttr<std::string>("else") == "@else");

    std::cout << "Test 5 passed!\n\n";
}

void testShapeInference() {
    std::cout << "=== Test 6: Shape Inference ===\n\n";

    Module module("@shape_infer_module");
    Operation* func = module.addFunction("test");
    Block* block = func->getRegion(0)->entryBlock();

    // Test 1: Broadcasting - [1, 10] + [10] -> [1, 10]
    auto a = block->addArg("%a", Type(DType::F32, Shape({1, 10})));
    auto b = block->addArg("%b", Type(DType::F32, Shape({10})));

    IRBuilder builder(block);
    auto addResult = builder.createAdd(a, b);  // No explicit resultType

    assert(addResult->type().shape().rank() == 2);
    assert(addResult->type().shape().dims() == std::vector<int64_t>({1, 10}));
    std::cout << "  Broadcasting [1,10] + [10] -> [1,10] ✓\n";

    // Test 2: MatMul - [2, 3, 4] @ [4, 5] -> [2, 3, 5]
    auto c = block->addArg("%c", Type(DType::F32, Shape({2, 3, 4})));
    auto d = block->addArg("%d", Type(DType::F32, Shape({4, 5})));

    auto matmulResult = builder.createMatMul(c, d);  // No explicit resultType

    assert(matmulResult->type().shape().rank() == 3);
    assert(matmulResult->type().shape().dims() == std::vector<int64_t>({2, 3, 5}));
    std::cout << "  MatMul [2,3,4] @ [4,5] -> [2,3,5] ✓\n";

    // Test 3: Mul broadcasting - [3, 1] * [1, 4] -> [3, 4]
    auto e = block->addArg("%e", Type(DType::F32, Shape({3, 1})));
    auto f = block->addArg("%f", Type(DType::F32, Shape({1, 4})));

    auto mulResult = builder.createMul(e, f);

    assert(mulResult->type().shape().dims() == std::vector<int64_t>({3, 4}));
    std::cout << "  Mul [3,1] * [1,4] -> [3,4] ✓\n";

    builder.createReturn({mulResult});

    std::cout << "\nTest 6 passed!\n\n";
}

int main() {
    testBasicMLP();
    testOpsWithAttributes();
    testEdgeCases();
    testUseDefChain();
    testControlFlow();
    testShapeInference();

    std::cout << "=== All tests passed! ===\n";
    return 0;
}

