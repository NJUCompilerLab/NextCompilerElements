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
    Operation* mainFunc = module.addFunction("@main");
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
    Operation* func = module.addFunction("@attention");
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
        Operation* func = module.addFunction("@empty_func");
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
        Operation* func = module.addFunction("@scalar_func");
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
        Operation* func = module.addFunction("@const_func");
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

int main() {
    testBasicMLP();
    testOpsWithAttributes();
    testEdgeCases();

    std::cout << "=== All tests passed! ===\n";
    return 0;
}

