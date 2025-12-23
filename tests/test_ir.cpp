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
#include <iostream>
#include <cassert>

using namespace nc::ir;

int main() {
    std::cout << "=== Test: Build Linear + ReLU IR ===\n\n";

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
    std::cout << jsonOutput << std::endl;

    // Basic verification
    assert(module.name() == "@mlp_module");
    assert(mainFunc->opType() == "Function");
    assert(funcBlock->numOps() == 3); // MatMul, Add, ReLU
    assert(funcBlock->terminator() != nullptr);
    assert(funcBlock->terminator()->opType() == "Return");

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
