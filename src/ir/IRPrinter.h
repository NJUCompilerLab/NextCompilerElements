#pragma once

#include "ir/Module.h"
#include "ir/Operation.h"
#include "ir/Block.h"
#include "ir/Region.h"
#include "ir/Value.h"
#include "ir/Type.h"
#include <nlohmann/json.hpp>
#include <string>
#include <sstream>

namespace nc {
namespace ir {

using json = nlohmann::json;

/// IRPrinter - serializes IR to JSON format
class IRPrinter {
public:
    /// Print Module to JSON string
    static std::string print(const Module& module, int indent = 2) {
        json j = printOperation(module.op());
        return j.dump(indent);
    }

    /// Print Operation to JSON
    static json printOperation(const Operation* op) {
        json j;
        j["name"] = op->name();
        j["op_type"] = op->opType();

        // Inputs (SSA references)
        if (!op->inputs().empty()) {
            json inputs = json::array();
            for (const auto& input : op->inputs()) {
                inputs.push_back(input->name());
            }
            j["inputs"] = inputs;
        }

        // Attributes
        if (!op->attrs().empty()) {
            j["attrs"] = printAttrs(op->attrs());
        }

        // Output types
        if (!op->outputTypes().empty()) {
            json types = json::array();
            for (const auto& t : op->outputTypes()) {
                types.push_back(printType(t));
            }
            j["output_types"] = types;
        }

        // Regions (body)
        if (op->numRegions() > 0) {
            json body = json::array();
            for (const auto& region : op->regions()) {
                body.push_back(printRegion(region.get()));
            }
            j["body"] = body;
        }

        return j;
    }

    /// Print Region to JSON
    static json printRegion(const Region* region) {
        json j;
        json blocks = json::array();
        for (const auto& block : region->blocks()) {
            blocks.push_back(printBlock(block.get()));
        }
        j["blocks"] = blocks;
        return j;
    }

    /// Print Block to JSON
    static json printBlock(const Block* block) {
        json j;
        j["label"] = block->label();

        // Block arguments
        if (!block->args().empty()) {
            json args = json::array();
            for (const auto& arg : block->args()) {
                args.push_back(printValue(arg.get()));
            }
            j["args"] = args;
        }

        // Operations
        json ops = json::array();
        for (const auto& op : block->ops()) {
            ops.push_back(printOperation(op.get()));
        }
        j["ops"] = ops;

        // Terminator
        if (block->terminator()) {
            j["terminator"] = printOperation(block->terminator());
        }

        return j;
    }

    /// Print Value to JSON
    static json printValue(const Value* value) {
        json j;
        j["name"] = value->name();
        j["type"] = printType(value->type());
        return j;
    }

    /// Print Type to JSON
    static json printType(const Type& type) {
        json j;
        j["dtype"] = dtypeToString(type.dtype());
        j["shape"] = type.shape().dims();
        return j;
    }

    /// Print Attributes to JSON
    static json printAttrs(const Attributes& attrs) {
        json j;
        for (const auto& [key, value] : attrs) {
            j[key] = printAttrValue(value);
        }
        return j;
    }

    /// Print AttrValue to JSON
    static json printAttrValue(const AttrValue& value) {
        return std::visit([](auto&& arg) -> json {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, int64_t>) {
                return arg;
            } else if constexpr (std::is_same_v<T, double>) {
                return arg;
            } else if constexpr (std::is_same_v<T, bool>) {
                return arg;
            } else if constexpr (std::is_same_v<T, std::string>) {
                return arg;
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                return arg;
            } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                return arg;
            }
            return nullptr;
        }, value);
    }
};

} // namespace ir
} // namespace nc
