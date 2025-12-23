#pragma once

#include "ir/Operation.h"
#include "ir/Value.h"
#include <string>
#include <vector>
#include <memory>

namespace nc {
namespace ir {

class Region;

/// Block - a sequence of operations with a label and arguments
/// Block arguments replace phi nodes in SSA
class Block {
public:
    explicit Block(std::string label = "@entry")
        : label_(std::move(label)), parentRegion_(nullptr), terminator_(nullptr) {}

    /// Block label (e.g., "@entry", "@loop_body")
    const std::string& label() const { return label_; }
    void setLabel(std::string label) { label_ = std::move(label); }

    // ========== Block Arguments ==========

    const std::vector<ValuePtr>& args() const { return args_; }

    ValuePtr addArg(std::string name, Type type) {
        auto arg = std::make_shared<Value>(std::move(name), std::move(type));
        args_.push_back(arg);
        return arg;
    }

    // ========== Operations ==========

    const std::vector<OperationPtr>& ops() const { return ops_; }
    std::vector<OperationPtr>& ops() { return ops_; }

    Operation* addOp(OperationPtr op) {
        op->setParentBlock(this);
        ops_.push_back(std::move(op));
        return ops_.back().get();
    }

    size_t numOps() const { return ops_.size(); }

    // ========== Terminator ==========

    Operation* terminator() const { return terminator_.get(); }

    void setTerminator(OperationPtr term) {
        term->setParentBlock(this);
        terminator_ = std::move(term);
    }

    // ========== Parent ==========

    Region* parentRegion() const { return parentRegion_; }
    void setParentRegion(Region* region) { parentRegion_ = region; }

private:
    std::string label_;
    std::vector<ValuePtr> args_;
    std::vector<OperationPtr> ops_;
    OperationPtr terminator_;
    Region* parentRegion_;
};

using BlockPtr = std::unique_ptr<Block>;

} // namespace ir
} // namespace nc
