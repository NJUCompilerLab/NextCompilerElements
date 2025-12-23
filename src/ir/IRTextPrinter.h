#pragma once

#include "ir/Module.h"
#include "ir/Operation.h"
#include "ir/Block.h"
#include "ir/Region.h"
#include "ir/Value.h"
#include "ir/Type.h"
#include <sstream>
#include <string>

namespace nc {
namespace ir {

/// IRTextPrinter - serializes IR to human-readable text format (MLIR-style)
///
/// Example output:
/// ```
/// module @mlp_module {
///   func @main(%input: f32[1, 784], %weight: f32[784, 256]) -> f32[1, 256] {
///   entry:
///     %v0 = MatMul(%input, %weight) : f32[1, 256]
///     %v1 = ReLU(%v0) : f32[1, 256]
///     return %v1
///   }
/// }
/// ```
class IRTextPrinter {
public:
    /// Print Module to text string
    static std::string print(const Module& module) {
        IRTextPrinter printer;
        printer.printModule(module.op());
        return printer.os_.str();
    }

private:
    std::ostringstream os_;
    int indent_ = 0;

    void printModule(const Operation* op) {
        os_ << "module " << op->name() << " {\n";
        indent_++;

        // Module has one region containing functions
        if (op->numRegions() > 0) {
            const auto& region = op->regions()[0];
            for (const auto& block : region->blocks()) {
                for (const auto& funcOp : block->ops()) {
                    if (funcOp->opType() == "Function") {
                        printFunction(funcOp.get());
                    }
                }
            }
        }

        indent_--;
        os_ << "}\n";
    }

    void printFunction(const Operation* op) {
        emitIndent();
        os_ << "func " << op->name() << "(";

        // Function has one region, entry block has arguments
        std::vector<const Value*> funcArgs;
        bool hasReturn = false;
        Type returnType;

        if (op->numRegions() > 0) {
            const auto& region = op->regions()[0];
            if (!region->blocks().empty()) {
                const auto& entryBlock = region->blocks()[0];
                // Collect function arguments
                for (size_t i = 0; i < entryBlock->args().size(); i++) {
                    if (i > 0) os_ << ", ";
                    const auto& arg = entryBlock->args()[i];
                    os_ << arg->name() << ": " << formatType(arg->type());
                }
                // Find return type from terminator
                if (entryBlock->terminator() &&
                    !entryBlock->terminator()->inputs().empty()) {
                    hasReturn = true;
                    returnType = entryBlock->terminator()->inputs()[0]->type();
                }
            }
        }

        os_ << ")";
        if (hasReturn) {
            os_ << " -> " << formatType(returnType);
        }
        os_ << " {\n";

        // Print blocks
        if (op->numRegions() > 0) {
            const auto& region = op->regions()[0];
            bool first = true;
            for (const auto& block : region->blocks()) {
                printBlock(block.get(), first);
                first = false;
            }
        }

        emitIndent();
        os_ << "}\n";
    }

    void printBlock(const Block* block, bool isEntry = false) {
        // Block label
        emitIndent();
        os_ << block->label().substr(1); // Remove '@' prefix for label
        // Only print block args if not entry block (entry args shown in func signature)
        if (!isEntry && !block->args().empty()) {
            os_ << "(";
            for (size_t i = 0; i < block->args().size(); i++) {
                if (i > 0) os_ << ", ";
                const auto& arg = block->args()[i];
                os_ << arg->name() << ": " << formatType(arg->type());
            }
            os_ << ")";
        }
        os_ << ":\n";

        indent_++;

        // Print operations
        for (const auto& op : block->ops()) {
            printOperation(op.get());
        }

        // Print terminator
        if (block->terminator()) {
            printTerminator(block->terminator());
        }

        indent_--;
    }

    void printOperation(const Operation* op) {
        emitIndent();

        // %name = OpType<attrs>(inputs) : output_type
        os_ << op->name() << " = " << op->opType();

        // Attributes
        if (!op->attrs().empty()) {
            os_ << "<" << formatAttrs(op->attrs()) << ">";
        }

        // Inputs
        os_ << "(";
        for (size_t i = 0; i < op->inputs().size(); i++) {
            if (i > 0) os_ << ", ";
            os_ << op->inputs()[i]->name();
        }
        os_ << ")";

        // Output types (support multiple)
        if (!op->outputTypes().empty()) {
            os_ << " : ";
            if (op->outputTypes().size() == 1) {
                os_ << formatType(op->outputTypes()[0]);
            } else {
                os_ << "(";
                for (size_t i = 0; i < op->outputTypes().size(); i++) {
                    if (i > 0) os_ << ", ";
                    os_ << formatType(op->outputTypes()[i]);
                }
                os_ << ")";
            }
        }

        os_ << "\n";

        // Print nested regions if any (for control flow ops)
        if (op->numRegions() > 0) {
            for (const auto& region : op->regions()) {
                indent_++;
                for (const auto& block : region->blocks()) {
                    printBlock(block.get(), false);
                }
                indent_--;
            }
        }
    }

    void printTerminator(const Operation* op) {
        emitIndent();

        if (op->opType() == "Return") {
            os_ << "return";
            if (!op->inputs().empty()) {
                for (size_t i = 0; i < op->inputs().size(); i++) {
                    os_ << (i == 0 ? " " : ", ") << op->inputs()[i]->name();
                }
            }
        } else if (op->opType() == "Br") {
            // br @dest(args)
            os_ << "br";
            if (op->hasAttr("dest")) {
                os_ << " " << op->getAttr<std::string>("dest");
            }
            if (!op->inputs().empty()) {
                os_ << "(";
                for (size_t i = 0; i < op->inputs().size(); i++) {
                    if (i > 0) os_ << ", ";
                    os_ << op->inputs()[i]->name();
                }
                os_ << ")";
            }
        } else if (op->opType() == "CondBr") {
            // cond_br %cond, @then, @else
            os_ << "cond_br";
            if (!op->inputs().empty()) {
                os_ << " " << op->inputs()[0]->name();
            }
            if (op->hasAttr("then")) {
                os_ << ", " << op->getAttr<std::string>("then");
            }
            if (op->hasAttr("else")) {
                os_ << ", " << op->getAttr<std::string>("else");
            }
        } else {
            os_ << op->opType();
        }
        os_ << "\n";
    }

    std::string formatType(const Type& type) const {
        std::ostringstream ss;
        ss << dtypeToString(type.dtype());
        const auto& dims = type.shape().dims();
        if (!dims.empty()) {
            ss << "[";
            for (size_t i = 0; i < dims.size(); i++) {
                if (i > 0) ss << ", ";
                ss << dims[i];
            }
            ss << "]";
        }
        // Scalar types print as just "f32" without brackets
        return ss.str();
    }

    std::string formatAttrs(const Attributes& attrs) const {
        std::ostringstream ss;
        bool first = true;
        for (const auto& [key, value] : attrs) {
            if (!first) ss << ", ";
            first = false;
            ss << key << "=" << formatAttrValue(value);
        }
        return ss.str();
    }

    std::string formatAttrValue(const AttrValue& value) const {
        return std::visit([](auto&& arg) -> std::string {
            using T = std::decay_t<decltype(arg)>;
            std::ostringstream ss;
            if constexpr (std::is_same_v<T, int64_t>) {
                ss << arg;
            } else if constexpr (std::is_same_v<T, double>) {
                ss << arg;
            } else if constexpr (std::is_same_v<T, bool>) {
                ss << (arg ? "true" : "false");
            } else if constexpr (std::is_same_v<T, std::string>) {
                ss << "\"" << arg << "\"";
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                ss << "[";
                for (size_t i = 0; i < arg.size(); i++) {
                    if (i > 0) ss << ", ";
                    ss << arg[i];
                }
                ss << "]";
            } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                ss << "[";
                for (size_t i = 0; i < arg.size(); i++) {
                    if (i > 0) ss << ", ";
                    ss << arg[i];
                }
                ss << "]";
            }
            return ss.str();
        }, value);
    }

    void emitIndent() {
        for (int i = 0; i < indent_; i++) {
            os_ << "  ";
        }
    }
};

} // namespace ir
} // namespace nc
